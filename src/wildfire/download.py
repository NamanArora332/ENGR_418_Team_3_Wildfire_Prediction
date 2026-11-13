"""Download the per-year FPA FOD-Attributes CSVs from Zenodo.

Files already on disk are skipped, so re-running is cheap. Downloads go to a
temporary ``.part`` file first, so an interrupted download never leaves a
half-written CSV behind.
"""

from __future__ import annotations

from pathlib import Path

from .config import RAW_FILE_TEMPLATE, ZENODO_RECORD_ID

ZENODO_API = "https://zenodo.org/api/records/{record_id}"
ZENODO_FILE_URL = "https://zenodo.org/records/{record_id}/files/{name}?download=1"


def list_record_files(record_id: str = ZENODO_RECORD_ID) -> dict[str, dict]:
    """Return {file name: {"url": ..., "size": bytes}} for a Zenodo record."""
    import requests

    resp = requests.get(ZENODO_API.format(record_id=record_id), timeout=60)
    resp.raise_for_status()
    files = {}
    for f in resp.json().get("files", []):
        name = f.get("key") or f.get("filename")
        url = (f.get("links") or {}).get("self") or ZENODO_FILE_URL.format(record_id=record_id, name=name)
        files[name] = {"url": url, "size": f.get("size")}
    return files


def _stream_to_file(url: str, dest: Path, expected_size: int | None) -> None:
    import requests

    part = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = expected_size or int(resp.headers.get("content-length", 0)) or None
        done = 0
        next_report = 0.0
        with open(part, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                fh.write(chunk)
                done += len(chunk)
                if total and done / total >= next_report:
                    print(f"    {dest.name}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB")
                    next_report += 0.25
    if expected_size and part.stat().st_size != expected_size:
        raise IOError(f"{dest.name}: got {part.stat().st_size} bytes, expected {expected_size}")
    part.rename(dest)


def download_years(years, raw_dir: Path, record_id: str = ZENODO_RECORD_ID, overwrite: bool = False) -> list[Path]:
    """Make sure the CSV for every year in ``years`` is in ``raw_dir``.

    Only contacts Zenodo if at least one file is missing.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    wanted = {year: raw_dir / RAW_FILE_TEMPLATE.format(year=year) for year in years}
    missing = {y: p for y, p in wanted.items() if overwrite or not p.exists() or p.stat().st_size == 0}

    for year, path in wanted.items():
        if year not in missing:
            print(f"  {path.name}: already downloaded ({path.stat().st_size / 1e6:,.0f} MB)")
    if not missing:
        return list(wanted.values())

    available = list_record_files(record_id)
    for year, path in missing.items():
        info = available.get(path.name)
        if info is None:
            raise FileNotFoundError(
                f"{path.name} is not in Zenodo record {record_id}. "
                f"Files there: {sorted(available)[:5]} ..."
            )
        print(f"  downloading {path.name} ...")
        _stream_to_file(info["url"], path, info.get("size"))
    return list(wanted.values())
