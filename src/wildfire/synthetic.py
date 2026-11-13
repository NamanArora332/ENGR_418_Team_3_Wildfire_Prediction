"""Fake FPA FOD-Attributes files, used to test the pipeline without the 2 GB download.

The column names copy the real dataset's naming, and the rows include the
problems the cleaner must handle: duplicates, negative sizes, -9999 codes,
numbers stored as text, and leaky columns. The values are random. Never report
results from this data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_FILE_TEMPLATE

STATES = ["CA", "OR", "WA", "AZ", "TX", "FL"]  # includes non-western states to test the filter


def make_year(year: int, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed + year)
    doy = rng.integers(1, 366, n)
    vpd = rng.gamma(2.0, 0.8, n)
    erc = rng.normal(50, 20, n).clip(0, 120)
    slope = rng.gamma(2.0, 5.0, n)
    # Size grows with dryness and fire danger, plus heavy-tailed noise.
    log_size = (-1.5 + 0.6 * vpd + 0.02 * erc + 0.02 * slope + rng.standard_t(3, n)).clip(-2, 5.5)
    size = np.round(10 ** log_size, 2)
    date = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(doy - 1, unit="D")
    df = pd.DataFrame({
        "FOD_ID": np.arange(n) + year * 100_000,
        "FPA_ID": [f"FS-{year}-{i}" for i in range(n)],
        "FIRE_NAME": rng.choice(["", "OAK", "RIDGE", "BIG CREEK"], n),
        "MTBS_ID": np.where(size > 1000, "CA1234", ""),
        "ICS_209_PLUS_INCIDENT_JOIN_ID": np.where(size > 300, "2020_123", ""),
        "FIRE_YEAR": year,
        "DISCOVERY_DATE": date.strftime("%m/%d/%Y"),
        "DISCOVERY_DOY": doy,
        "DISCOVERY_TIME": rng.choice([np.nan, 930, 1415, 1800], n),
        "NWCG_GENERAL_CAUSE": rng.choice(["Natural", "Debris and open burning", "Missing data/not specified/undetermined"], n),
        "CONT_DATE": (date + pd.to_timedelta(rng.integers(0, 20, n), unit="D")).strftime("%m/%d/%Y"),
        "FIRE_SIZE": size,
        "FIRE_SIZE_CLASS": pd.cut(size, [-1, 0.25, 10, 100, 300, 1000, 5000, np.inf], labels=list("ABCDEFG")).astype(str),
        "LATITUDE": rng.uniform(32, 49, n),
        "LONGITUDE": rng.uniform(-124, -103, n),
        "STATE": rng.choice(STATES, n),
        "OWNER_DESCR": rng.choice(["USFS", "PRIVATE", "BLM"], n),
        "Elevation": rng.uniform(0, 3000, n),
        "Slope": slope,
        "Aspect": rng.uniform(0, 360, n),
        "Land_Cover": rng.choice([41, 42, 52, 71], n),
        "EVH": rng.choice([25, 101, 103, 205, 305], n),
        "No_FireStation_5.0km": rng.choice([np.nan, np.nan, 1, 2, 4], n),  # blank = no station
        "road_interstate_dis": np.where(rng.random(n) < 0.01, rng.uniform(0, 5000, n), np.nan),
        "GHM": rng.uniform(0, 1, n),
        "vpd": vpd,
        "erc": erc,
        "pr": rng.exponential(1.0, n),
        "tmmx": rng.normal(300, 8, n),
        "vs": rng.gamma(2, 1.5, n),
        "NDVI-1day": rng.uniform(-0.1, 0.9, n),
        "vpd_Normal": rng.uniform(0.5, 3, n),
        "vpd_5D_mean": vpd + rng.normal(0, 0.2, n),
        "Evacuation": 0,
        "Mystery_attribute": rng.normal(size=n),
    })
    # Realistic mess for the cleaner to fix
    df.loc[df.sample(frac=0.02, random_state=1).index, "Elevation"] = -9999
    df.loc[df.sample(frac=0.01, random_state=2).index, "FIRE_SIZE"] = -1
    df["GHM"] = df["GHM"].round(3).astype(str)  # numbers stored as text
    df.loc[df.index[10:13], "LONGITUDE"] = -77.0  # western state, eastern coordinates
    return pd.concat([df, df.head(5)], ignore_index=True)  # 5 duplicate FOD_IDs


def write_synthetic_years(raw_dir: Path, years, n: int = 2000) -> list[Path]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for year in years:
        path = raw_dir / RAW_FILE_TEMPLATE.format(year=year)
        make_year(year, n).to_csv(path, index=False)
        paths.append(path)
    return paths
