# Apply this patch to your existing GitHub feature branch

This archive is an **additive repository-completion update**, not a replacement for the already-uploaded `train_wildfire_model.py`. It includes a new `README.md`, `.gitignore`, `src/`, `data/`, `notebooks/`, `results/`, `results_retrospective/`, `report/`, `tests/test_reporting.py`, and `.github/workflows/ci.yml`.

## Preferred: Git on your computer (PowerShell)

1. Download and unzip this archive.
2. Clone the repository and check out the existing branch `feature/advanced-wildfire-model` (which already contains the advanced script).
3. Copy the extracted contents into the repository root. Existing `README.md` should be replaced. Do not overwrite `train_wildfire_model.py` with an older script; this package deliberately omits it.
4. Run `python -m unittest discover -s tests -v`. If you have the dataset locally, run `python -m src.reproduce --data data/2020_FPA_FOD_cons.csv`.
5. Commit on `feature/advanced-wildfire-model`, push, then have a teammate review the updated Pull Request #3.

For browser-only GitHub: switch to `feature/advanced-wildfire-model` first; edit the root `README.md`, then use the GitHub web file editor or Codespaces to create the nested paths. **Do not upload this ZIP file as one file**; it must be extracted so GitHub can display the folders and files.

GitHub ignores empty directories, but this package provides README/scripts in every required directory.
