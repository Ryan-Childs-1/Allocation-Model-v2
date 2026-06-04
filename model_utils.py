"""Runtime scoring utilities for the two-network Keras FLM ranker approach."""
from __future__ import annotations

import numpy as np
import pandas as pd
from base_features import add_features, add_token_features, num


def _item_series(df):
    if "Item" in df.columns:
        return df["Item"].fillna("__missing__").astype(str).reset_index(drop=True)
    return pd.Series(np.arange(len(df)).astype(str))


def _transform(bundle, tok):
    cols = bundle["num_features"] + bundle["cat_features"]
    for c in cols:
        if c not in tok.columns:
            tok[c] = np.nan
    X = bundle["preprocessor"].transform(tok[cols])
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype="float32")


def build_token_table(feat, max_units_per_row=20):
    rows = []
    row_ids = []
    unit_ids = []
    left_units = np.floor(num(feat["f_left_units"]).fillna(0).values).astype(int)
    rec_units = np.ceil(num(feat["f_rec_units"]).fillna(0).values).astype(int)
    gap_units = np.ceil(num(feat["f_gap_aggressive_units"]).fillna(0).values).astype(int)
    for i in range(len(feat)):
        cap = int(max(0, min(max_units_per_row, max(left_units[i], rec_units[i], gap_units[i], 1))))
        if cap <= 0:
            continue
        idx = np.repeat(i, cap)
        units = np.arange(1, cap + 1)
        sub = feat.iloc[idx].reset_index(drop=True)
        tok = add_token_features(sub, units)
        rows.append(tok)
        row_ids.extend(idx.tolist())
        unit_ids.extend(units.tolist())
    if not rows:
        return pd.DataFrame(), np.array([], dtype=int), np.array([], dtype=int)
    return pd.concat(rows, ignore_index=True), np.asarray(row_ids, dtype=int), np.asarray(unit_ids, dtype=int)


def score_token_probs(model, bundle, tok, batch_size=8192):
    if tok.empty:
        return np.array([], dtype=float)
    X = _transform(bundle, tok)
    p = model.predict(X, batch_size=batch_size, verbose=0)
    p = np.asarray(p).reshape(-1)
    return np.nan_to_num(p.astype(float), nan=0, posinf=0, neginf=0).clip(0, 1)


def first_pass_allocation(df, feat, allocation_model, bundle, cutoff=None, max_units_per_row=20):
    tok, row_ids, unit_ids = build_token_table(feat, max_units_per_row=max_units_per_row)
    n = len(feat)
    flm = num(feat["f_flm"]).fillna(1).clip(lower=1).values
    left = num(feat["f_left_dc"]).fillna(0).clip(lower=0).values
    if tok.empty:
        return np.zeros(n), np.zeros(n), tok, row_ids, unit_ids, np.array([])
    probs = score_token_probs(allocation_model, bundle, tok)
    if cutoff is None:
        cutoff = float(bundle.get("allocation_cutoff", 0.50))
    take = probs >= cutoff
    units_by_row = np.zeros(n, dtype=float)
    # Ensure allocated units are contiguous. If FLM 3 is selected, FLMs 1-2 are counted too.
    for r in np.unique(row_ids[take]):
        max_unit = unit_ids[(row_ids == r) & take].max(initial=0)
        units_by_row[r] = max_unit
    # Keep first-pass allocation in whole FLM units. If available DC is less than one FLM, allocate zero.
    units_by_row = np.minimum(units_by_row, np.floor(left / np.maximum(flm, 1)))
    raw_alloc = units_by_row * flm
    return raw_alloc, units_by_row, tok, row_ids, unit_ids, probs


def deallocate_to_dc(df, feat, raw_alloc, allocation_probs, tok, row_ids, unit_ids, deallocation_model, bundle, max_units_per_row=20):
    n = len(feat)
    item = _item_series(df)
    flm = num(feat["f_flm"]).fillna(1).clip(lower=1).values
    left = num(feat["f_left_dc"]).fillna(0).clip(lower=0).values
    final_units = np.floor(raw_alloc / flm).astype(int)
    if tok.empty:
        return raw_alloc.copy(), np.zeros(n), np.zeros(n)

    # Determine group excess after raw allocation.
    group_left = pd.Series(left).groupby(item).transform("max").values
    group_alloc = pd.Series(raw_alloc).groupby(item).transform("sum").values
    group_over = np.maximum(group_alloc - group_left, 0)
    over_by_row = np.maximum(raw_alloc - left, 0) / np.maximum(flm, 1)

    allocated_mask = unit_ids <= final_units[row_ids]
    if not allocated_mask.any():
        return raw_alloc.copy(), np.zeros(n), np.zeros(n)
    dtok = tok.loc[allocated_mask].reset_index(drop=True)
    d_row_ids = row_ids[allocated_mask]
    d_unit_ids = unit_ids[allocated_mask]
    d_alloc_scores = allocation_probs[allocated_mask]
    d_group_over_units = group_over[d_row_ids] / np.maximum(flm[d_row_ids], 1)
    d_row_over_units = over_by_row[d_row_ids]
    dtok = add_token_features(dtok, d_unit_ids, alloc_score=d_alloc_scores, raw_allocated=np.ones(len(dtok)), group_over_units=d_group_over_units, row_over_units=d_row_over_units)
    de_probs = score_token_probs(deallocation_model, bundle, dtok)

    remove = np.zeros(len(dtok), dtype=bool)
    # For each item/DC group, remove highest deallocation-probability tokens until group excess is zero.
    d_item = item.iloc[d_row_ids].reset_index(drop=True)
    for k, sub_idx in pd.Series(np.arange(len(dtok))).groupby(d_item).groups.items():
        idx = np.asarray(list(sub_idx), dtype=int)
        if len(idx) == 0:
            continue
        group_rows = d_row_ids[idx]
        need_units = int(np.ceil(max(0, group_over[group_rows[0]]) / max(flm[group_rows[0]], 1)))
        # Also remove explicit row-over-DC units if any.
        need_units = max(need_units, int(np.ceil(np.nanmax(d_row_over_units[idx]) if len(idx) else 0)))
        if need_units <= 0:
            continue
        order = idx[np.argsort(-de_probs[idx])]
        remove[order[:need_units]] = True

    removed_units_by_row = np.zeros(n, dtype=float)
    for r in np.unique(d_row_ids[remove]):
        removed_units_by_row[r] = (d_row_ids[remove] == r).sum()
    final_units2 = np.maximum(final_units - removed_units_by_row, 0)
    final_alloc = final_units2 * flm

    # Deterministic final safety: if still over DC, remove lowest allocation-score tokens until fixed.
    safety_cut_units = np.zeros(n, dtype=float)
    for k, idx_rows in pd.Series(np.arange(n)).groupby(item).groups.items():
        idx_rows = np.asarray(list(idx_rows), dtype=int)
        group_dc = float(np.nanmax(left[idx_rows])) if len(idx_rows) else 0
        excess = final_alloc[idx_rows].sum() - group_dc
        if excess <= 0:
            continue
        # Build removable tokens for this group, sorted by lowest allocation probability / highest deallocation probability.
        candidates = []
        for r in idx_rows:
            for u in range(1, int(final_alloc[r] // flm[r]) + 1):
                mask = (row_ids == r) & (unit_ids == u)
                score = float(allocation_probs[mask][0]) if mask.any() else 0
                candidates.append((score, r, u))
        candidates.sort(key=lambda x: x[0])
        needed = int(np.ceil(excess / max(np.nanmedian(flm[idx_rows]), 1)))
        for _, r, _ in candidates[:needed]:
            if final_alloc[r] >= flm[r]:
                final_alloc[r] -= flm[r]
                safety_cut_units[r] += 1
                excess -= flm[r]
                if excess <= 0:
                    break
    deallocated = raw_alloc - final_alloc
    return final_alloc, deallocated, safety_cut_units * flm


def score_dataframe_core(df, allocation_model, deallocation_model, bundle):
    orig = df.copy().reset_index(drop=True)
    feat = add_features(orig)
    raw_alloc, raw_units, tok, row_ids, unit_ids, probs = first_pass_allocation(
        orig, feat, allocation_model, bundle,
        cutoff=float(bundle.get("allocation_cutoff", 0.50)),
        max_units_per_row=int(bundle.get("max_units_per_row", 20)),
    )
    final_alloc, deallocated, safety_cut = deallocate_to_dc(
        orig, feat, raw_alloc, probs, tok, row_ids, unit_ids, deallocation_model, bundle,
        max_units_per_row=int(bundle.get("max_units_per_row", 20)),
    )
    item = _item_series(orig)
    left = num(feat["f_left_dc"]).fillna(0).clip(lower=0).values
    group_left = pd.Series(left).groupby(item).transform("max").values
    group_final = pd.Series(final_alloc).groupby(item).transform("sum").values
    ai_left = np.maximum(group_left - group_final, 0)
    out = orig.copy()
    out["AI Starting Left DC"] = np.rint(left).astype(int)
    out["AI Allocation Before Deallocation"] = np.rint(raw_alloc).astype(int)
    out["AI Deallocation Units"] = np.rint(deallocated).astype(int)
    out["AI DC Safety Cut"] = np.rint(safety_cut).astype(int)
    out["AI Final Alloc"] = np.rint(final_alloc).astype(int)
    out["AI Left DC"] = np.rint(ai_left).astype(int)
    out["AI Allocated FLMs Before Deallocation"] = np.rint(raw_units).astype(int)
    out["AI Allocation Cutoff"] = float(bundle.get("allocation_cutoff", 0.50))
    return out
