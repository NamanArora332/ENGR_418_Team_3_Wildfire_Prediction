"""Turn the raw per-year CSVs into one clean, model-ready table.

Pipeline (each step is logged in build_report.json):

    1. read each year's CSV, keep the chosen states, drop 'drop'-group columns
    2. stack the years
    3. remove duplicate FOD_IDs, rows missing a size, location or date, and rows
       whose coordinates fall outside the chosen region
    4. convert number-like text to numbers, turn -9999-style codes into NaN,
       fill count columns where blank means zero (fire stations nearby)
    5. drop columns that are empty, constant, or more than 95% missing
    6. add the target (size_class, log_fire_size) and calendar features
    7. write fires_clean.parquet + column_policy.csv + build_report.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .columns import classify_columns, is_categorical_hint
import re

from .config import (CLASS_NAMES, FILL_ZERO_PATTERNS, MAX_MISSING_SHARE, MISSING_SENTINELS,
                     SIZE_BINS_ACRES, BuildConfig)

REQUIRED = ("FOD_ID", "FIRE_SIZE", "LATITUDE", "LONGITUDE", "STATE")


# --- 1-2. reading ---------------------------------------------------------------

def read_year(path: Path, states: tuple[str, ...] | None) -> tuple[pd.DataFrame, dict]:
    """Read one year's CSV, keep the chosen states, drop columns marked 'drop'."""
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    stats = {"file": path.name, "rows_raw": len(df), "columns_raw": df.shape[1]}
    if states:
        df = df[df["STATE"].astype(str).str.upper().isin(states)]
    stats["rows_in_states"] = len(df)
    policy = classify_columns(df.columns)
    to_drop = policy.loc[policy["group"] == "drop", "column"].tolist()
    return df.drop(columns=to_drop), stats


# --- 3-5. cleaning --------------------------------------------------------------

def parse_discovery_date(df: pd.DataFrame) -> pd.Series:
    """DISCOVERY_DATE as datetime. Falls back to FIRE_YEAR + DISCOVERY_DOY if unparseable."""
    date = pd.to_datetime(df.get("DISCOVERY_DATE"), errors="coerce", format="mixed")
    if {"FIRE_YEAR", "DISCOVERY_DOY"} <= set(df.columns):
        fallback = pd.to_datetime(
            df["FIRE_YEAR"].astype("Int64").astype(str) + "-" + df["DISCOVERY_DOY"].astype("Int64").astype(str),
            format="%Y-%j", errors="coerce",
        )
        date = date.fillna(fallback)
    return date


def drop_invalid_rows(df: pd.DataFrame, report: dict, bbox: dict | None = None) -> pd.DataFrame:
    n0 = len(df)
    df = df.drop_duplicates(subset="FOD_ID", keep="first")
    report["removed_duplicate_fod_id"] = n0 - len(df)

    df = df.assign(FIRE_SIZE=pd.to_numeric(df["FIRE_SIZE"], errors="coerce"))
    bad_size = df["FIRE_SIZE"].isna() | (df["FIRE_SIZE"] < 0)
    report["removed_missing_or_negative_size"] = int(bad_size.sum())
    df = df[~bad_size]

    lat = pd.to_numeric(df["LATITUDE"], errors="coerce")
    lon = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    bad_loc = lat.isna() | lon.isna() | ~lat.between(18, 72) | ~lon.between(-180, -60)
    report["removed_bad_location"] = int(bad_loc.sum())
    df = df[~bad_loc].assign(LATITUDE=lat[~bad_loc], LONGITUDE=lon[~bad_loc])

    if bbox:
        outside = ~(df["LONGITUDE"].between(*bbox["lon"]) & df["LATITUDE"].between(*bbox["lat"]))
        report["removed_outside_region"] = int(outside.sum())
        df = df[~outside]
    else:
        report["removed_outside_region"] = 0

    date = parse_discovery_date(df)
    report["removed_bad_date"] = int(date.isna().sum())
    df = df[date.notna()].assign(DISCOVERY_DATE=date[date.notna()])
    return df


def fix_types(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Numbers stored as text -> numbers; codes -> category; sentinels -> NaN."""
    protected = {"FOD_ID", "DISCOVERY_DATE"}
    converted, categorical, sentinels = [], [], 0
    for col in df.columns:
        if col in protected:
            continue
        s = df[col]
        if is_categorical_hint(col):
            if pd.api.types.is_float_dtype(s) and (s.dropna() % 1 == 0).all():
                s = s.astype("Int64")  # codes like 101.0 -> "101", not "101.0"
            df[col] = s.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA}).astype("category")
            categorical.append(col)
            continue
        if not pd.api.types.is_numeric_dtype(s):
            num = pd.to_numeric(s, errors="coerce")
            non_null = s.notna().sum()
            if non_null and num.notna().sum() / non_null >= 0.95:
                df[col] = num
                converted.append(col)
            else:
                df[col] = s.astype("string").astype("category")
                categorical.append(col)
                continue
        mask = df[col].isin(MISSING_SENTINELS)
        if mask.any():
            sentinels += int(mask.sum())
            df[col] = df[col].mask(mask)
    filled = {}
    for col in df.columns:
        if any(re.search(p, col) for p in FILL_ZERO_PATTERNS) and pd.api.types.is_numeric_dtype(df[col]):
            n = int(df[col].isna().sum())
            if n:
                df[col] = df[col].fillna(0)
                filled[col] = n
    report["missing_filled_with_zero"] = filled
    report["text_columns_converted_to_numbers"] = converted
    report["categorical_columns"] = categorical
    report["sentinel_values_set_to_nan"] = sentinels
    return df


def drop_useless_columns(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    keep_always = {"FOD_ID", "FIRE_SIZE", "DISCOVERY_DATE", "FIRE_YEAR"}
    empty = [c for c in df.columns if c not in keep_always and df[c].isna().all()]
    constant = [c for c in df.columns if c not in keep_always and c not in empty and df[c].nunique(dropna=True) <= 1]
    mostly_empty = [c for c in df.columns if c not in keep_always and c not in empty + constant
                    and df[c].isna().mean() > MAX_MISSING_SHARE]
    report["dropped_empty_columns"] = empty
    report["dropped_constant_columns"] = constant
    report["dropped_mostly_empty_columns"] = mostly_empty
    return df.drop(columns=empty + constant + mostly_empty)


# --- 6. target and engineered features -----------------------------------------

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    small_max, medium_max = SIZE_BINS_ACRES
    size = df["FIRE_SIZE"]
    df["size_class"] = np.select([size < small_max, size < medium_max], [0, 1], default=2).astype("int8")
    df["log_fire_size"] = np.log10(size + 0.01)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    date = df["DISCOVERY_DATE"]
    days_in_year = np.where(date.dt.is_leap_year, 366.0, 365.0)
    angle = 2 * np.pi * (date.dt.dayofyear - 1) / days_in_year
    df["doy_sin"] = np.sin(angle)
    df["doy_cos"] = np.cos(angle)
    df["month"] = date.dt.month.astype("int8")
    df["weekday"] = date.dt.weekday.astype("int8")
    if "FIRE_YEAR" not in df.columns:
        df["FIRE_YEAR"] = date.dt.year
    df["FIRE_YEAR"] = df["FIRE_YEAR"].astype("int16")
    if "DISCOVERY_TIME" in df.columns:
        t = pd.to_numeric(df["DISCOVERY_TIME"], errors="coerce")
        df["discovery_hour"] = (t // 100).where(t.between(0, 2359)).astype("Int8")
    return df


# --- 7. outputs -----------------------------------------------------------------

def build_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Column policy plus quick stats, one row per column of the clean table."""
    policy = classify_columns(df.columns)
    policy["kind"] = [
        "categorical" if isinstance(df[c].dtype, pd.CategoricalDtype) else
        "datetime" if pd.api.types.is_datetime64_any_dtype(df[c]) else
        "numeric" if pd.api.types.is_numeric_dtype(df[c]) else "text"
        for c in policy["column"]
    ]
    policy["missing_pct"] = [round(100 * df[c].isna().mean(), 2) for c in policy["column"]]
    policy["n_unique"] = [int(df[c].nunique(dropna=True)) for c in policy["column"]]
    order = {"target": 0, "id": 1, "static": 2, "weather": 3, "retrospective": 4, "outcome": 5, "review": 6}
    return policy.sort_values(["group", "column"], key=lambda s: s.map(order) if s.name == "group" else s.str.lower())


def build_dataset(cfg: BuildConfig, write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the whole pipeline. Returns (clean table, column policy, report)."""
    report: dict = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "years": list(cfg.years),
        "states": list(cfg.states) if cfg.states else "all",
        "bbox": cfg.bbox,
        "size_bins_acres": list(SIZE_BINS_ACRES),
        "per_year": [],
    }
    frames = []
    for year in cfg.years:
        path = cfg.raw_path(year)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run the download step first.")
        df_year, stats = read_year(path, cfg.states)
        print(f"  {year}: {stats['rows_raw']:,} fires -> {stats['rows_in_states']:,} in chosen states")
        report["per_year"].append(stats)
        frames.append(df_year)

    df = pd.concat(frames, ignore_index=True)
    report["rows_after_state_filter"] = len(df)
    df = drop_invalid_rows(df, report, cfg.bbox)
    df = fix_types(df, report)
    df = drop_useless_columns(df, report)
    df = add_target(df)
    df = add_calendar_features(df)
    df = df.sort_values(["DISCOVERY_DATE", "FOD_ID"]).reset_index(drop=True)

    policy = build_policy(df)
    report["rows_final"] = len(df)
    report["columns_final"] = df.shape[1]
    report["class_counts"] = {CLASS_NAMES[k]: int(v) for k, v in df["size_class"].value_counts().sort_index().items()}
    report["class_counts_by_year"] = {
        int(y): {CLASS_NAMES[k]: int(v) for k, v in g.value_counts().sort_index().items()}
        for y, g in df.groupby("FIRE_YEAR")["size_class"]
    }
    report["columns_per_group"] = policy["group"].value_counts().to_dict()
    report["columns_needing_review"] = policy.loc[policy["group"] == "review", "column"].tolist()

    if write:
        cfg.clean_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cfg.clean_path, index=False)
        policy.to_csv(cfg.policy_path, index=False)
        cfg.report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return df, policy, report


# --- for teammates ----------------------------------------------------------------

def load_clean(data_root: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (clean table, column policy) written by build_dataset."""
    cfg = BuildConfig(data_root=data_root) if data_root else BuildConfig()
    if not cfg.clean_path.exists():
        raise FileNotFoundError(
            f"{cfg.clean_path} not found. Run notebooks/01_build_dataset.ipynb first, "
            "or point WILDFIRE_DATA_DIR at the shared data folder."
        )
    return pd.read_parquet(cfg.clean_path), pd.read_csv(cfg.policy_path)
