"""Column policy: which columns may be used as model inputs.

The dataset has roughly 310 columns. Every one is put into exactly one group:

    id             identifiers and dates kept for bookkeeping (never a feature)
    target         the final fire size and labels built from it (never a feature)
    outcome        only known after the fire, e.g. containment date and cause.
                   Kept for EDA, never a feature.
    drop           identifiers, free text, and fields that leak size indirectly.
                   Removed from the clean file.
    static         known before the fire: location, terrain, land cover,
                   people, climate normals, calendar
    weather        conditions on the discovery date (gridMET weather,
                   fire-danger indices, NDVI). Available in practice from
                   same-day observations and forecasts.
    retrospective  multi-day windows (e.g. *_5D_*) that may include days AFTER
                   ignition. Research only.
    review         not matched by any rule. A human must look at it and add a rule.

Feature tiers combine groups for modelling:

    static         static only (most conservative)
    at_discovery   static + weather  <- recommended default
    retrospective  static + weather + retrospective (report as an upper bound only)

The rules are checked in order, so the first match wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

FEATURE_TIERS: dict[str, tuple[str, ...]] = {
    "static": ("static",),
    "at_discovery": ("static", "weather"),
    "retrospective": ("static", "weather", "retrospective"),
}
NEVER_FEATURE_GROUPS = ("id", "target", "outcome", "drop", "review")

# gridMET daily variables (short names used by the dataset)
GRIDMET_DAILY = {
    "pr", "tmmn", "tmmx", "rmin", "rmax", "sph", "srad", "th", "vs",
    "bi", "erc", "fm100", "fm1000", "etr", "pet", "vpd",
}


@dataclass(frozen=True)
class Rule:
    group: str
    pattern: str  # regular expression, matched case-insensitively against the full name
    note: str

    def matches(self, column: str) -> bool:
        return re.search(self.pattern, column, flags=re.IGNORECASE) is not None


def _exact(*names: str) -> str:
    return r"^(" + "|".join(re.escape(n) for n in names) + r")$"


RULES: tuple[Rule, ...] = (
    # --- bookkeeping ---------------------------------------------------------
    Rule("id", _exact("FOD_ID", "FIRE_YEAR", "DISCOVERY_DATE", "DISCOVERY_DOY", "DISCOVERY_TIME"),
         "Identifier / raw date-time. Re-encoded as doy_sin, doy_cos, month, weekday, discovery_hour."),
    Rule("target", _exact("FIRE_SIZE"), "Final burned area in acres: the thing we predict."),
    # --- removed: identifiers, free text, size leaks --------------------------
    Rule("drop", r"^(MTBS_|ICS_209)",
         "LEAK: MTBS and ICS-209 records only exist for large or significant fires."),
    Rule("drop", _exact("FIRE_NAME", "COMPLEX_NAME", "FIRE_CODE"),
         "LEAK: named fires, complexes and fire codes skew toward large incidents."),
    Rule("drop", r"^(FPA_ID|LOCAL_|SOURCE_SYSTEM$|SOURCE_REPORTING_UNIT|NWCG_REPORTING_UNIT)",
         "Record-keeping identifier with no predictive meaning."),
    Rule("drop", _exact("FIPS_NAME", "COUNTY", "OBJECTID", "Shape", "geometry") + r"|^Unnamed",
         "Free text or redundant (use FIPS_CODE / coordinates)."),
    # --- kept for EDA, never a feature ----------------------------------------
    Rule("outcome", r"^(FIRE_SIZE_CLASS|CONT_)", "Only known once the fire is out."),
    Rule("outcome", r"^NWCG_(CAUSE|GENERAL_CAUSE)",
         "Cause is often set by investigation after the fire."),
    Rule("outcome", r"evac", "Evacuations happen because the fire grew."),
    # --- multi-day windows ------------------------------------------------------
    Rule("retrospective", r"5D", "Multi-day window that may include days after ignition."),
    # --- conditions on the discovery date ------------------------------------
    Rule("weather", _exact(*sorted(GRIDMET_DAILY)), "gridMET value on the discovery date."),
    Rule("weather", r"ndvi|(^|_)evi", "Satellite vegetation index near the discovery date."),
    Rule("weather", r"(^|_)PL($|_)|prepared", "Daily preparedness level (administrative)."),
    # --- known before the fire -------------------------------------------------
    Rule("static", _exact("LATITUDE", "LONGITUDE", "STATE", "FIPS_CODE"), "Location at discovery."),
    Rule("static", _exact("NWCG_REPORTING_AGENCY", "OWNER_DESCR", "SOURCE_SYSTEM_TYPE"),
         "Agency / land owner responsible at discovery."),
    Rule("static", _exact("Elevation", "Slope", "Aspect", "TRI", "TPI"), "Terrain."),
    Rule("static", r"land_?cover|^EV[HCT]|^FRG|fuel", "Land cover / vegetation layer. Check its vintage year."),
    Rule("static", r"normal|^annual_|aridity", "Long-term climate normal."),
    Rule("static", r"^GHM$|pop|road|dist|svi|income|housing", "Human footprint / social context."),
    Rule("static", r"mang|gacc|jurisdic|owner|agency|region", "Management / jurisdiction."),
)

# Columns created by the cleaning step
ENGINEERED: dict[str, tuple[str, str]] = {
    "size_class": ("target", "0 = Small (<10 ac), 1 = Medium (10-299 ac), 2 = Large (>=300 ac)."),
    "log_fire_size": ("target", "log10(FIRE_SIZE + 0.01), for optional regression."),
    "doy_sin": ("static", "Day of year on a circle (sine)."),
    "doy_cos": ("static", "Day of year on a circle (cosine)."),
    "month": ("static", "Discovery month, 1-12."),
    "weekday": ("static", "Discovery weekday, 0 = Monday."),
    "discovery_hour": ("static", "Hour of discovery, 0-23, from DISCOVERY_TIME."),
}


# Columns that hold category codes even when they look numeric (e.g. land cover 41, 42 ...)
CATEGORICAL_HINT = r"^(STATE|FIPS_CODE|OWNER_DESCR|NWCG_REPORTING_AGENCY|SOURCE_SYSTEM_TYPE)$|land_?cover|^EVT|^FRG|gacc|mang"


def is_categorical_hint(column: str) -> bool:
    return re.search(CATEGORICAL_HINT, column, flags=re.IGNORECASE) is not None


def classify_column(column: str) -> tuple[str, str]:
    """Return (group, note) for one column name."""
    if column in ENGINEERED:
        return ENGINEERED[column]
    for rule in RULES:
        if rule.matches(column):
            return rule.group, rule.note
    return "review", "No rule matched. Inspect this column, then add a rule in columns.py."


def classify_columns(columns) -> pd.DataFrame:
    """Policy table with one row per column: column, group, note."""
    rows = [(c, *classify_column(c)) for c in columns]
    return pd.DataFrame(rows, columns=["column", "group", "note"])


def feature_columns(policy: pd.DataFrame, tier: str = "at_discovery") -> list[str]:
    """Model input columns allowed by ``tier`` (see FEATURE_TIERS)."""
    if tier not in FEATURE_TIERS:
        raise ValueError(f"Unknown tier {tier!r}. Choose from {list(FEATURE_TIERS)}.")
    allowed = FEATURE_TIERS[tier]
    cols = policy.loc[policy["group"].isin(allowed), "column"].tolist()
    assert_no_leakage(cols, policy)
    return cols


def assert_no_leakage(features, policy: pd.DataFrame) -> None:
    """Raise if any proposed feature is an id, target, outcome, dropped or unreviewed column."""
    groups = dict(zip(policy["column"], policy["group"]))
    bad = {f: groups.get(f, "unknown") for f in features if groups.get(f, "unknown") in NEVER_FEATURE_GROUPS + ("unknown",)}
    if bad:
        raise ValueError(f"These columns must not be model inputs: {bad}")
