"""ENGR 418 Team 3: wildfire size-class data pipeline (Role A).

Typical use from a modelling notebook:

    from wildfire import load_clean, feature_columns, year_split
    df, policy = load_clean()
    X_cols = feature_columns(policy, tier="at_discovery")
    train, val, test = year_split(df)
"""

from .clean import build_dataset, load_clean
from .columns import FEATURE_TIERS, assert_no_leakage, classify_columns, feature_columns
from .config import CLASS_NAMES, SUGGESTED_SPLIT, BuildConfig
from .splits import year_split

__all__ = [
    "BuildConfig", "CLASS_NAMES", "FEATURE_TIERS", "SUGGESTED_SPLIT",
    "assert_no_leakage", "build_dataset", "classify_columns", "feature_columns",
    "load_clean", "year_split",
]
