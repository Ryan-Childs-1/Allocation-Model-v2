"""TensorFlow-free runtime for the Keras FLM ranker models.

The trained .keras networks are exported to compressed NumPy weights. This file
performs identical inference for Dense + BatchNorm + Swish/Sigmoid networks
without importing TensorFlow/Keras.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from base_features import add_features, add_token_features, num


def _sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def _swish(x):
    return x * _sigmoid(x)


class NumpyKerasBinaryClassifier:
    def __init__(self, path: str | Path):
        self.path = str(path)
        data = np.load(self.path, allow_pickle=True)
        self.arrays = {k: data[k] for k in data.files if k != 'metadata_json'}
        try:
            self.metadata = json.loads(str(data['metadata_json'].item()))
        except Exception:
            self.metadata = {'name': Path(path).stem, 'layers': []}
        self.layers = self.metadata.get('layers', [])
        # Input dim from first dense layer.
        self.input_shape = (None, int(self.arrays['dense_0_kernel'].shape[0]))

    def predict(self, X, batch_size: int = 8192, verbose: int = 0):
        X = np.asarray(X, dtype=np.float32)
        outs = []
        bs = max(1, int(batch_size or 8192))
        for start in range(0, len(X), bs):
            outs.append(self._forward(X[start:start+bs]))
        if not outs:
            return np.zeros((0, 1), dtype=np.float32)
        return np.vstack(outs).astype(np.float32)

    def _forward(self, x):
        out = x.astype(np.float32, copy=False)
        for layer in self.layers:
            typ = layer.get('type')
            if typ == 'dense':
                i = int(layer['index'])
                out = out @ self.arrays[f'dense_{i}_kernel'] + self.arrays[f'dense_{i}_bias']
                act = layer.get('activation', 'linear')
                if act == 'swish':
                    out = _swish(out)
                elif act == 'sigmoid':
                    out = _sigmoid(out)
                elif act == 'relu':
                    out = np.maximum(out, 0)
            elif typ == 'batchnorm':
                i = int(layer['index'])
                eps = float(layer.get('epsilon', 0.001))
                gamma = self.arrays[f'bn_{i}_gamma']
                beta = self.arrays[f'bn_{i}_beta']
                mean = self.arrays[f'bn_{i}_mean']
                var = self.arrays[f'bn_{i}_var']
                out = (out - mean) / np.sqrt(var + eps) * gamma + beta
        return np.asarray(out).reshape(len(x), -1)

    def summary_dict(self):
        dense_layers = []
        params = 0
        for layer in self.layers:
            if layer.get('type') == 'dense':
                i = int(layer['index'])
                k = self.arrays[f'dense_{i}_kernel']; b = self.arrays[f'dense_{i}_bias']
                params += int(k.size + b.size)
                dense_layers.append({'layer': f'Dense {i+1}', 'input_dim': int(k.shape[0]), 'units': int(k.shape[1]), 'activation': layer.get('activation','linear'), 'parameters': int(k.size + b.size)})
            elif layer.get('type') == 'batchnorm':
                i = int(layer['index'])
                gamma = self.arrays[f'bn_{i}_gamma']
                params += int(gamma.size * 4)
                dense_layers.append({'layer': f'BatchNorm {i+1}', 'input_dim': int(gamma.size), 'units': int(gamma.size), 'activation': 'inference normalization', 'parameters': int(gamma.size*4)})
        return {'name': self.metadata.get('name', Path(self.path).stem), 'input_dim': self.input_shape[-1], 'parameter_count': params, 'layers': dense_layers}


def load_numpy_model(path: str | Path):
    return NumpyKerasBinaryClassifier(path)


def _item_series(df):
    if 'Item' in df.columns:
        return df['Item'].fillna('__missing__').astype(str).reset_index(drop=True)
    return pd.Series(np.arange(len(df)).astype(str))



def patch_preprocessor_compat(bundle):
    """Patch old sklearn-1.3 pickles so they run under newer sklearn on Streamlit Cloud.

Old SimpleImputer objects may not have _fill_dtype, which newer sklearn expects.
This adds the compatible attribute without changing learned statistics.
    """
    def walk(obj):
        if obj is None:
            return
        if obj.__class__.__name__ == 'SimpleImputer':
            if not hasattr(obj, '_fill_dtype'):
                obj._fill_dtype = getattr(obj, '_fit_dtype', object)
        if hasattr(obj, 'steps'):
            for _, step in obj.steps:
                walk(step)
        if hasattr(obj, 'transformers'):
            for t in obj.transformers:
                if len(t) >= 2:
                    walk(t[1])
        if hasattr(obj, 'transformers_'):
            for t in obj.transformers_:
                if len(t) >= 2:
                    walk(t[1])
    try:
        walk(bundle.get('preprocessor'))
    except Exception:
        pass
    return bundle


def _transform(bundle, tok):
    bundle = patch_preprocessor_compat(bundle)
    cols = bundle['num_features'] + bundle['cat_features']
    for c in cols:
        if c not in tok.columns:
            tok[c] = np.nan
    X = bundle['preprocessor'].transform(tok[cols])
    if hasattr(X, 'toarray'):
        X = X.toarray()
    return np.asarray(X, dtype='float32')


def build_token_table(feat, max_units_per_row=20):
    rows = []
    row_ids = []
    unit_ids = []
    left_units = np.floor(num(feat['f_left_units']).fillna(0).values).astype(int)
    rec_units = np.ceil(num(feat['f_rec_units']).fillna(0).values).astype(int)
    gap_units = np.ceil(num(feat['f_gap_aggressive_units']).fillna(0).values).astype(int)
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
    expected = getattr(model, 'input_shape', (None, X.shape[1]))[-1]
    if X.shape[1] > expected:
        X = X[:, :expected]
    elif X.shape[1] < expected:
        X = np.pad(X, ((0, 0), (0, expected - X.shape[1])), mode='constant')
    p = model.predict(X, batch_size=batch_size, verbose=0)
    p = np.asarray(p).reshape(-1)
    return np.nan_to_num(p.astype(float), nan=0, posinf=0, neginf=0).clip(0, 1)


def first_pass_allocation(df, feat, allocation_model, bundle, cutoff=None, max_units_per_row=20):
    tok, row_ids, unit_ids = build_token_table(feat, max_units_per_row=max_units_per_row)
    n = len(feat)
    flm = num(feat['f_flm']).fillna(1).clip(lower=1).values
    left = num(feat['f_left_dc']).fillna(0).clip(lower=0).values
    if tok.empty:
        return np.zeros(n), np.zeros(n), tok, row_ids, unit_ids, np.array([])
    probs = score_token_probs(allocation_model, bundle, tok)
    if cutoff is None:
        cutoff = float(bundle.get('allocation_cutoff', 0.50))
    take = probs >= cutoff
    units_by_row = np.zeros(n, dtype=float)
    for r in np.unique(row_ids[take]):
        max_unit = unit_ids[(row_ids == r) & take].max(initial=0)
        units_by_row[r] = max_unit
    units_by_row = np.minimum(units_by_row, np.floor(left / np.maximum(flm, 1)))
    raw_alloc = units_by_row * flm
    return raw_alloc, units_by_row, tok, row_ids, unit_ids, probs


def deallocate_to_dc(df, feat, raw_alloc, allocation_probs, tok, row_ids, unit_ids, deallocation_model, bundle, max_units_per_row=20):
    n = len(feat)
    item = _item_series(df)
    flm = num(feat['f_flm']).fillna(1).clip(lower=1).values
    left = num(feat['f_left_dc']).fillna(0).clip(lower=0).values
    final_units = np.floor(raw_alloc / flm).astype(int)
    if tok.empty:
        return raw_alloc.copy(), np.zeros(n), np.zeros(n)

    group_left = pd.Series(left).groupby(item).transform('max').values
    group_alloc = pd.Series(raw_alloc).groupby(item).transform('sum').values
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
    d_item = item.iloc[d_row_ids].reset_index(drop=True)
    for k, sub_idx in pd.Series(np.arange(len(dtok))).groupby(d_item).groups.items():
        idx = np.asarray(list(sub_idx), dtype=int)
        if len(idx) == 0:
            continue
        group_rows = d_row_ids[idx]
        need_units = int(np.ceil(max(0, group_over[group_rows[0]]) / max(flm[group_rows[0]], 1)))
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

    safety_cut_units = np.zeros(n, dtype=float)
    for k, idx_rows in pd.Series(np.arange(n)).groupby(item).groups.items():
        idx_rows = np.asarray(list(idx_rows), dtype=int)
        group_dc = float(np.nanmax(left[idx_rows])) if len(idx_rows) else 0
        excess = final_alloc[idx_rows].sum() - group_dc
        if excess <= 0:
            continue
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
        cutoff=float(bundle.get('allocation_cutoff', 0.50)),
        max_units_per_row=int(bundle.get('max_units_per_row', 20)),
    )
    final_alloc, deallocated, safety_cut = deallocate_to_dc(
        orig, feat, raw_alloc, probs, tok, row_ids, unit_ids, deallocation_model, bundle,
        max_units_per_row=int(bundle.get('max_units_per_row', 20)),
    )
    item = _item_series(orig)
    left = num(feat['f_left_dc']).fillna(0).clip(lower=0).values
    group_left = pd.Series(left).groupby(item).transform('max').values
    group_final = pd.Series(final_alloc).groupby(item).transform('sum').values
    ai_left = np.maximum(group_left - group_final, 0)
    out = orig.copy()
    out['AI Starting Left DC'] = np.rint(left).astype(int)
    out['AI Allocation Before Deallocation'] = np.rint(raw_alloc).astype(int)
    out['AI Deallocation Units'] = np.rint(deallocated).astype(int)
    out['AI DC Safety Cut'] = np.rint(safety_cut).astype(int)
    out['AI Final Alloc'] = np.rint(final_alloc).astype(int)
    out['AI Left DC'] = np.rint(ai_left).astype(int)
    out['AI Allocated FLMs Before Deallocation'] = np.rint(raw_units).astype(int)
    out['AI Allocation Cutoff'] = float(bundle.get('allocation_cutoff', 0.50))
    return out
