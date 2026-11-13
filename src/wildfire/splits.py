"""Split by fire year, so the model is always tested on fires later than the ones it trained on."""

from __future__ import annotations

import pandas as pd

from .config import SUGGESTED_SPLIT


def year_split(df: pd.DataFrame, split: dict[str, tuple[int, int]] | None = None):
    """Return (train, val, test) using inclusive year ranges, e.g. {"train": (2011, 2016), ...}."""
    split = split or SUGGESTED_SPLIT
    parts = []
    for name in ("train", "val", "test"):
        lo, hi = split[name]
        part = df[df["FIRE_YEAR"].between(lo, hi)]
        if part.empty:
            raise ValueError(f"The {name} split ({lo}-{hi}) has no rows. Check the years you built.")
        parts.append(part)
    return tuple(parts)
