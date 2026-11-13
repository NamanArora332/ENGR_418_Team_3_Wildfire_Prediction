"""Project-wide settings for the wildfire data pipeline.

Everything the team might want to change (years, states, size classes, where
files live) is defined here, so the notebooks and tests all agree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --- Source dataset -----------------------------------------------------------
# FPA FOD-Attributes (Pourmohamad et al., 2024, ESSD 16:3045-3060).
# Zenodo record: https://doi.org/10.5281/zenodo.8381129
ZENODO_RECORD_ID = "8381129"
RAW_FILE_TEMPLATE = "{year}_FPA_FOD_cons.csv"  # one CSV per fire year, 1992-2020

# --- Scope --------------------------------------------------------------------
# Western US, where most large fires burn. Change to None for all of CONUS.
WESTERN_STATES: tuple[str, ...] = ("AZ", "CA", "CO", "ID", "MT", "NM", "NV", "OR", "UT", "WA", "WY")
# Box around those states (lon, lat ranges, with a small margin). Fires whose
# coordinates fall outside it have a wrong state or wrong coordinates.
WESTERN_BBOX: dict[str, tuple[float, float]] = {"lon": (-125.5, -101.5), "lat": (30.8, 49.5)}
# Ten most recent years: enough for a train/validation/test split by year
# while keeping the download to roughly 2 GB.
DEFAULT_YEARS: tuple[int, ...] = tuple(range(2011, 2021))

# Suggested split by fire year (Role B owns the final decision).
SUGGESTED_SPLIT: dict[str, tuple[int, int]] = {
    "train": (2011, 2016),
    "val": (2017, 2018),
    "test": (2019, 2020),
}

# --- Target -------------------------------------------------------------------
# Final fire size in acres -> class. Small < 10, Medium 10-299.99, Large >= 300.
# These merge the NWCG size classes A-B, C-D and E-G.
SIZE_BINS_ACRES: tuple[float, float] = (10.0, 300.0)
CLASS_NAMES: tuple[str, ...] = ("Small", "Medium", "Large")

# Values used in the attribute columns to mean "no data".
MISSING_SENTINELS: tuple[float, ...] = (-9999.0, -32768.0)

# Count columns where "missing" really means zero. The fire-station counts never
# contain 0 (their minimum is 1), so a blank means "no station within that distance".
FILL_ZERO_PATTERNS: tuple[str, ...] = (r"^No_FireStation_",)

# Columns that store percentile RANGES as text ("50-70%", ">98%"). Each range is
# replaced by its midpoint (60, 99) so models keep the order.
PERCENTILE_RANGE_PATTERN = r"_Percentile$"

# Drop columns that are missing for more than this share of fires (useless as inputs).
MAX_MISSING_SHARE = 0.95

# --- Locations ----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive/ENGR418_wildfire_data")

# Output file names (inside the data root)
CLEAN_FILE = "processed/fires_clean.parquet"
POLICY_FILE = "processed/column_policy.csv"
REPORT_FILE = "processed/build_report.json"


def default_data_root() -> Path:
    """Where data lives: $WILDFIRE_DATA_DIR, else Google Drive on Colab, else ./data."""
    env = os.environ.get("WILDFIRE_DATA_DIR")
    if env:
        return Path(env)
    if COLAB_DRIVE_ROOT.parent.exists():  # Drive is mounted
        return COLAB_DRIVE_ROOT
    return REPO_ROOT / "data"


@dataclass
class BuildConfig:
    """Settings for one run of the dataset build."""

    years: tuple[int, ...] = DEFAULT_YEARS
    states: tuple[str, ...] | None = WESTERN_STATES
    data_root: Path = field(default_factory=default_data_root)
    raw_dir: Path | None = None  # defaults to <data_root>/raw
    # Coordinate box. "auto" = WESTERN_BBOX when all chosen states are western, else none.
    bbox: dict | str | None = "auto"

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.raw_dir = Path(self.raw_dir) if self.raw_dir else self.data_root / "raw"
        if self.bbox == "auto":
            western = self.states and set(self.states) <= set(WESTERN_STATES)
            self.bbox = WESTERN_BBOX if western else None

    @property
    def clean_path(self) -> Path:
        return self.data_root / CLEAN_FILE

    @property
    def policy_path(self) -> Path:
        return self.data_root / POLICY_FILE

    @property
    def report_path(self) -> Path:
        return self.data_root / REPORT_FILE

    def raw_path(self, year: int) -> Path:
        return self.raw_dir / RAW_FILE_TEMPLATE.format(year=year)
