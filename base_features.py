"""Allocation AI Keras FLM feature engineering.

This module is shared by the Jupyter trainer and Streamlit runtime.  It is
intentionally pandas/numpy/sklearn only so runtime stays stable.
"""
from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from pyxlsb import open_workbook
except Exception:  # pragma: no cover
    open_workbook = None

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SHEET_NAME = "3.3 Working Table"

NUMERIC_FEATURES_COMPACT = [
    "f_flm", "f_mil", "f_qoh", "f_supply", "f_current_supply", "f_current_with_inbound",
    "f_l30", "f_d30", "f_d60", "f_lw", "f_ttm", "f_proj",
    "f_l30_60", "f_d30_60", "f_lw_60", "f_ttm_60",
    "f_demand_weighted", "f_demand_protective", "f_demand_aggressive",
    "f_demand_mean", "f_demand_max", "f_demand_std", "f_demand_count",
    "f_weekly_rate", "f_gap", "f_gap_aggressive", "f_gap_units", "f_gap_aggressive_units",
    "f_alloc_rec", "f_rec_units", "f_left_dc", "f_left_units", "f_dc_pressure",
    "f_woc_current", "f_woc_after_rec", "f_overstock_current", "f_overstock_units_after_rec",
    "f_understock_units_after_rec", "f_rule_units", "f_rule_alloc", "f_rule_vs_rec_units",
    "f_is_allocate", "f_is_review", "f_is_no_alloc", "f_rescue_rule",
    "f_cost", "f_retail", "f_margin", "f_gm_pct", "f_rank", "f_avg_woc", "f_square_footage",
    "f_supply_to_demand", "f_rec_to_demand", "f_recent_momentum", "f_lw_spike",
    "f_item_rows", "f_item_gap_sum", "f_item_demand_sum", "f_item_rec_sum", "f_item_left_max",
    "f_item_gap_share", "f_item_demand_rank", "f_site_gap_share", "f_site_demand_rank",
    "f_vendor_gap_share", "f_class_gap_share", "f_dept_gap_share",
    # token-level features used by both neural networks
    "t_unit_index", "t_unit_frac_of_left", "t_units_remaining_after", "t_alloc_after_units",
    "t_alloc_after_units_ratio", "t_supply_after", "t_gap_after", "t_over_after",
    "t_woc_after", "t_is_first_flm", "t_is_beyond_rec", "t_is_beyond_gap",
    "t_alloc_score", "t_raw_allocated", "t_group_over_units", "t_row_over_units",
]

CATEGORICAL_FEATURES_COMPACT = [
    "Vendor", "Brand", "Department Id", "Class Id", "Line Id", "Site", "Region", "Zone",
    "Store Size", "Status", "Flag", "__source_id"
]

TEXT_WORDS = ["AMMO", "RIFLE", "PISTOL", "CAMP", "FISH", "ROD", "REEL", "BOOT", "JACKET", "TRAIL", "PACK", "GRILL", "TRAEGER", "SAFE", "OPTIC"]


def make_unique_columns(cols):
    seen = {}
    out = []
    for i, c in enumerate(cols):
        name = str(c).strip() if c is not None and str(c).strip() else f"Unnamed_{i}"
        if name in seen:
            seen[name] += 1
            out.append(f"{name}__dup{seen[name]}")
        else:
            seen[name] = 1
            out.append(name)
    return out


def read_xlsb(path, sheet_name=SHEET_NAME):
    if open_workbook is None:
        raise ImportError("pyxlsb is required to read .xlsb files")
    with open_workbook(str(path)) as wb:
        sh_name = sheet_name if sheet_name in wb.sheets else wb.sheets[0]
        with wb.get_sheet(sh_name) as sh:
            rows = [[cell.v for cell in row] for row in sh.rows()]
    header_idx = 1 if len(rows) > 1 and any(str(x).strip() == "Final Alloc." for x in rows[1]) else 0
    header = make_unique_columns(rows[header_idx])
    df = pd.DataFrame(rows[header_idx + 1:], columns=header).dropna(how="all").reset_index(drop=True)
    return df


def read_allocation_file(path, sheet_name=SHEET_NAME):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".xlsb":
        df = read_xlsb(path, sheet_name)
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=sheet_name, header=1)
        df.columns = make_unique_columns(df.columns)
    elif suffix == ".csv":
        df = pd.read_csv(path)
        df.columns = make_unique_columns(df.columns)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    df["__source_file"] = path.name
    df["__source_id"] = re.sub(r"[^A-Za-z0-9]+", "_", path.stem)[:80]
    return df


def num(s):
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(pd.Series(s), errors="coerce")


def ratio(a, b, cap=999.0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = a / np.where(np.abs(b) < 1e-9, np.nan, b)
    return pd.Series(out).replace([np.inf, -np.inf], np.nan).fillna(0).clip(-cap, cap)


def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan


def add_adjusted_left_dc(df):
    out = df.copy()
    if "Left DC" not in out.columns:
        out["Left DC"] = 0
    base = num(out["Left DC"]).fillna(0).clip(lower=0)
    if "Final Alloc." in out.columns:
        final = num(out["Final Alloc."])
        out["__AI_START_LEFT_DC"] = np.where(final.notna(), base + final.fillna(0).clip(lower=0), base)
    else:
        out["__AI_START_LEFT_DC"] = base
    return out


def add_features(df):
    df = add_adjusted_left_dc(df).copy().reset_index(drop=True)
    needed = [
        "Vendor", "Brand", "Department Id", "Class Id", "Line Id", "Product ID", "Item", "Description", "Pcode Description",
        "Status", "Site", "Region", "Zone", "Store Size", "Rank", "Cost", "Retail", "GM Pct", "Square Footage",
        "MIL", "MIL.1", "FLM", "FLM.1", "L30", "D30", "D60", "LW", "TTM", "Qoh", "Supply", "Intrans",
        "Store Transfer", "Store PO Qty", "Avg. WOC", "Proj. Demand", "Alloc. Rec.", "Flag", "Final Alloc.", "Left DC",
        "__source_id", "__source_file", "__AI_START_LEFT_DC"
    ]
    ensure_cols(df, needed)

    flm = num(df["FLM.1"]).fillna(num(df["FLM"])).replace(0, np.nan).fillna(1).clip(lower=1)
    mil = num(df["MIL.1"]).fillna(num(df["MIL"])).fillna(0).clip(lower=0)
    qoh = num(df["Qoh"]).fillna(0).clip(lower=0)
    supply = num(df["Supply"]).fillna(0).clip(lower=0)
    intrans = num(df["Intrans"]).fillna(0).clip(lower=0)
    transfer = num(df["Store Transfer"]).fillna(0).clip(lower=0)
    storepo = num(df["Store PO Qty"]).fillna(0).clip(lower=0)
    current = qoh + supply
    current_inbound = current + intrans + transfer + storepo

    l30 = num(df["L30"]).fillna(0).clip(lower=0)
    d30 = num(df["D30"]).fillna(0).clip(lower=0)
    d60 = num(df["D60"]).fillna(0).clip(lower=0)
    lw = num(df["LW"]).fillna(0).clip(lower=0)
    ttm = num(df["TTM"]).fillna(0).clip(lower=0)
    proj = num(df["Proj. Demand"]).fillna(0).clip(lower=0)

    l30_60 = l30 * 2.0
    d30_60 = d30 * 2.0
    lw_60 = lw * 8.58
    ttm_60 = ttm / 6.0
    dem = pd.concat([l30_60, d30_60, d60, lw_60, ttm_60, proj], axis=1).replace(0, np.nan)
    dmean = dem.mean(axis=1).fillna(0)
    dmax = dem.max(axis=1).fillna(0)
    dstd = dem.std(axis=1).fillna(0)
    dcount = dem.notna().sum(axis=1)
    dweighted = (0.32*d60 + 0.22*d30_60 + 0.18*l30_60 + 0.12*lw_60 + 0.08*ttm_60 + 0.08*proj).fillna(0)
    dprotect = pd.Series(np.maximum.reduce([dweighted.values, d60.values, proj.values, mil.values]), index=df.index)
    dagg = pd.Series(np.maximum.reduce([dprotect.values, dmax.values, mil.values * 1.10]), index=df.index)
    weekly = pd.Series(np.maximum.reduce([lw.values, l30.values/4.29, d60.values/8.58, ttm.values/52.0, np.ones(len(df))*0.01]), index=df.index)
    left = num(df["__AI_START_LEFT_DC"]).fillna(0).clip(lower=0)
    rec = num(df["Alloc. Rec."]).fillna(0).clip(lower=0)
    flag = df["Flag"].fillna("").astype(str).str.upper()
    is_alloc = flag.str.contains("ALLOC", regex=False).astype(int)
    is_review = flag.str.contains("REVIEW", regex=False).astype(int)
    is_no_alloc = ((is_alloc + is_review) == 0).astype(int)

    gap = (dprotect - current).clip(lower=0)
    gap_aggressive = (dagg - current).clip(lower=0)
    rec_units = ratio(rec, flm, cap=999).round().clip(lower=0)
    left_units = ratio(left, flm, cap=999).clip(lower=0)
    gap_units = np.ceil(gap / np.where(flm <= 0, 1, flm)).clip(lower=0)
    gap_aggressive_units = np.ceil(gap_aggressive / np.where(flm <= 0, 1, flm)).clip(lower=0)
    woc_current = ratio(current, weekly + 0.01).clip(0, 999)
    rescue = ((current <= 0) & (dprotect > 0) & (left >= flm))
    base_rule_units = np.where(rescue, 1, np.where((is_alloc | is_review) & (gap_units > 0), np.minimum(left_units, gap_units), 0))
    base_rule_units = pd.Series(base_rule_units, index=df.index).fillna(0).clip(lower=0)
    overstock_after_rec = (current + rec - (dprotect + flm)).clip(lower=0)
    understock_after_rec = (dprotect - (current + rec)).clip(lower=0)
    cost = num(df["Cost"]).fillna(0)
    retail = num(df["Retail"]).fillna(0)

    F = {
        "f_flm": flm, "f_mil": mil, "f_qoh": qoh, "f_supply": supply, "f_current_supply": current,
        "f_current_with_inbound": current_inbound, "f_l30": l30, "f_d30": d30, "f_d60": d60, "f_lw": lw, "f_ttm": ttm,
        "f_proj": proj, "f_l30_60": l30_60, "f_d30_60": d30_60, "f_lw_60": lw_60, "f_ttm_60": ttm_60,
        "f_demand_weighted": dweighted, "f_demand_protective": dprotect, "f_demand_aggressive": dagg,
        "f_demand_mean": dmean, "f_demand_max": dmax, "f_demand_std": dstd, "f_demand_count": dcount,
        "f_weekly_rate": weekly, "f_gap": gap, "f_gap_aggressive": gap_aggressive, "f_gap_units": gap_units,
        "f_gap_aggressive_units": gap_aggressive_units, "f_alloc_rec": rec, "f_rec_units": rec_units,
        "f_left_dc": left, "f_left_units": left_units, "f_dc_pressure": ratio(rec, left + 1).clip(0, 999),
        "f_woc_current": woc_current, "f_woc_after_rec": ratio(current + rec, weekly + 0.01).clip(0, 999),
        "f_overstock_current": (current - (dprotect + flm)).clip(lower=0),
        "f_overstock_units_after_rec": np.ceil(overstock_after_rec / np.where(flm <= 0, 1, flm)).clip(lower=0),
        "f_understock_units_after_rec": np.ceil(understock_after_rec / np.where(flm <= 0, 1, flm)).clip(lower=0),
        "f_rule_units": base_rule_units, "f_rule_alloc": base_rule_units * flm, "f_rule_vs_rec_units": base_rule_units - rec_units,
        "f_is_allocate": is_alloc, "f_is_review": is_review, "f_is_no_alloc": is_no_alloc, "f_rescue_rule": rescue.astype(int),
        "f_cost": cost, "f_retail": retail, "f_margin": retail - cost, "f_gm_pct": num(df["GM Pct"]).fillna(0),
        "f_rank": num(df["Rank"]).fillna(0), "f_avg_woc": num(df["Avg. WOC"]).fillna(0), "f_square_footage": num(df["Square Footage"]).fillna(0),
        "f_supply_to_demand": ratio(current, dprotect + 1), "f_rec_to_demand": ratio(rec, dprotect + 1),
        "f_recent_momentum": l30_60 - d60, "f_lw_spike": lw_60 - dweighted,
    }
    df = pd.concat([df, pd.DataFrame(F, index=df.index)], axis=1)

    for key, prefix in [("Item", "item"), ("Site", "site"), ("Vendor", "vendor"), ("Class Id", "class"), ("Department Id", "dept")]:
        grp = df.groupby(key, dropna=False, sort=False)
        df[f"f_{prefix}_rows"] = grp["f_flm"].transform("size")
        df[f"f_{prefix}_gap_sum"] = grp["f_gap"].transform("sum")
        df[f"f_{prefix}_demand_sum"] = grp["f_demand_protective"].transform("sum")
        df[f"f_{prefix}_rec_sum"] = grp["f_alloc_rec"].transform("sum")
        df[f"f_{prefix}_left_max"] = grp["f_left_dc"].transform("max")
        df[f"f_{prefix}_gap_share"] = ratio(df["f_gap"], df[f"f_{prefix}_gap_sum"] + 1)
        df[f"f_{prefix}_demand_rank"] = grp["f_demand_protective"].rank(method="average", ascending=False, pct=True).fillna(0)

    text = (df["Description"].fillna("").astype(str) + " " + df["Pcode Description"].fillna("").astype(str)).str.upper()
    for word in TEXT_WORDS:
        df[f"f_text_{word.lower()}"] = text.str.contains(word, regex=False).astype("int8")

    for c in df.columns:
        if c.startswith("f_"):
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    return df


def add_token_features(feat, unit_index, alloc_score=None, raw_allocated=None, group_over_units=None, row_over_units=None):
    """Create per-FLM token features. unit_index is 1-based."""
    tok = feat.copy()
    k = np.asarray(unit_index, dtype=float)
    flm = num(tok["f_flm"]).fillna(1).clip(lower=1).values
    left_units = num(tok["f_left_units"]).fillna(0).values
    rec_units = num(tok["f_rec_units"]).fillna(0).values
    gap_units = num(tok["f_gap_units"]).fillna(0).values
    current = num(tok["f_current_supply"]).fillna(0).values
    dprotect = num(tok["f_demand_protective"]).fillna(0).values
    weekly = num(tok["f_weekly_rate"]).fillna(0.01).values
    alloc_after = k * flm
    supply_after = current + alloc_after

    tok["t_unit_index"] = k.astype("float32")
    tok["t_unit_frac_of_left"] = (k / np.maximum(left_units, 1)).clip(0, 999).astype("float32")
    tok["t_units_remaining_after"] = np.maximum(left_units - k, 0).astype("float32")
    tok["t_alloc_after_units"] = k.astype("float32")
    tok["t_alloc_after_units_ratio"] = (k / np.maximum(gap_units, 1)).clip(0, 999).astype("float32")
    tok["t_supply_after"] = supply_after.astype("float32")
    tok["t_gap_after"] = np.maximum(dprotect - supply_after, 0).astype("float32")
    tok["t_over_after"] = np.maximum(supply_after - (dprotect + flm), 0).astype("float32")
    tok["t_woc_after"] = (supply_after / np.maximum(weekly, 0.01)).clip(0, 999).astype("float32")
    tok["t_is_first_flm"] = (k == 1).astype("float32")
    tok["t_is_beyond_rec"] = (k > rec_units).astype("float32")
    tok["t_is_beyond_gap"] = (k > gap_units).astype("float32")
    tok["t_alloc_score"] = np.zeros(len(tok), dtype="float32") if alloc_score is None else np.asarray(alloc_score, dtype="float32")
    tok["t_raw_allocated"] = np.zeros(len(tok), dtype="float32") if raw_allocated is None else np.asarray(raw_allocated, dtype="float32")
    tok["t_group_over_units"] = np.zeros(len(tok), dtype="float32") if group_over_units is None else np.asarray(group_over_units, dtype="float32")
    tok["t_row_over_units"] = np.zeros(len(tok), dtype="float32") if row_over_units is None else np.asarray(row_over_units, dtype="float32")
    return tok


def feature_lists(feat, max_category_levels=250):
    num_features = [c for c in NUMERIC_FEATURES_COMPACT if c in feat.columns]
    # Include text booleans but still small.
    num_features += [c for c in feat.columns if c.startswith("f_text_")]
    num_features = list(dict.fromkeys(num_features))
    cat_features = []
    for c in CATEGORICAL_FEATURES_COMPACT:
        if c in feat.columns and feat[c].nunique(dropna=False) <= max_category_levels:
            cat_features.append(c)
    return num_features, cat_features


def build_preprocessor(num_features, cat_features, min_frequency=10):
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency, sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency, sparse=False)
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)]), cat_features),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
