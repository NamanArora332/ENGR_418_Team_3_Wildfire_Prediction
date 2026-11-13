"""Checks for the data pipeline. Run with:  python -m pytest -q"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wildfire import BuildConfig, build_dataset, classify_columns, feature_columns, year_split  # noqa: E402
from wildfire.columns import assert_no_leakage, classify_column  # noqa: E402
from wildfire.synthetic import write_synthetic_years  # noqa: E402

YEARS = (2017, 2018, 2019, 2020)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    write_synthetic_years(root / "raw", YEARS, n=800)
    cfg = BuildConfig(years=YEARS, states=("CA", "OR", "WA", "AZ"), data_root=root)
    df, policy, report = build_dataset(cfg)
    return cfg, df, policy, report


@pytest.mark.parametrize("column, group", [
    ("FIRE_SIZE", "target"),
    ("FIRE_SIZE_CLASS", "outcome"),
    ("CONT_DATE", "outcome"),
    ("NWCG_GENERAL_CAUSE", "outcome"),
    ("Evacuation", "outcome"),
    ("MTBS_ID", "drop"),
    ("ICS_209_PLUS_INCIDENT_JOIN_ID", "drop"),
    ("FIRE_NAME", "drop"),
    ("vpd_5D_mean", "retrospective"),
    ("vpd", "weather"),
    ("NDVI-1day", "weather"),
    ("vpd_Normal", "static"),
    ("Elevation", "static"),
    ("Land_Cover", "static"),
    ("Slope_std_deviation", "review"),  # must not be caught by the EVI rule
])
def test_known_columns_are_classified(column, group):
    assert classify_column(column)[0] == group


def test_unknown_column_needs_review():
    assert classify_column("Mystery_attribute")[0] == "review"


def test_size_class_boundaries():
    from wildfire.clean import add_target
    df = add_target(pd.DataFrame({"FIRE_SIZE": [0.0, 9.99, 10.0, 299.99, 300.0, 50000.0]}))
    assert df["size_class"].tolist() == [0, 0, 1, 1, 2, 2]


def test_rows_are_cleaned(built):
    _, df, _, report = built
    assert df["FOD_ID"].is_unique
    assert (df["FIRE_SIZE"] >= 0).all()
    assert set(df["STATE"].unique()) <= {"CA", "OR", "WA", "AZ"}
    assert report["removed_duplicate_fod_id"] > 0
    assert report["removed_missing_or_negative_size"] > 0
    assert (df["Elevation"].dropna() != -9999).all()
    assert pd.api.types.is_numeric_dtype(df["GHM"])  # text numbers converted
    assert isinstance(df["Land_Cover"].dtype, pd.CategoricalDtype)


def test_leaky_columns_never_reach_features(built):
    _, df, policy, _ = built
    assert "MTBS_ID" not in df.columns and "FIRE_NAME" not in df.columns
    for tier in ("static", "at_discovery", "retrospective"):
        cols = feature_columns(policy, tier)
        assert not {"FIRE_SIZE", "size_class", "log_fire_size", "FIRE_SIZE_CLASS", "CONT_DATE",
                    "NWCG_GENERAL_CAUSE", "FOD_ID", "Mystery_attribute"} & set(cols)
    assert "vpd" not in feature_columns(policy, "static")
    assert "vpd" in feature_columns(policy, "at_discovery")
    assert "vpd_5D_mean" not in feature_columns(policy, "at_discovery")
    with pytest.raises(ValueError):
        assert_no_leakage(["vpd", "FIRE_SIZE_CLASS"], policy)


def test_outputs_written_and_reloadable(built):
    cfg, df, _, _ = built
    from wildfire import load_clean
    df2, policy2 = load_clean(cfg.data_root)
    assert len(df2) == len(df)
    assert set(policy2["column"]) == set(df2.columns)


def test_year_split(built):
    _, df, _, _ = built
    train, val, test = year_split(df, {"train": (2017, 2018), "val": (2019, 2019), "test": (2020, 2020)})
    assert train["FIRE_YEAR"].max() < val["FIRE_YEAR"].min() <= val["FIRE_YEAR"].max() < test["FIRE_YEAR"].min()
    assert len(train) + len(val) + len(test) == len(df)


def test_policy_covers_every_column():
    policy = classify_columns(["a", "FIRE_SIZE", "vpd"])
    assert policy["group"].tolist() == ["review", "target", "weather"]
