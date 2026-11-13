# Data pipeline (Role A)

Builds **one clean, leakage-checked table** of western-US wildfires from FPA FOD-Attributes. Everyone on the team trains on it.
Raw and processed data are **not** stored in Git. They live in a shared Google Drive folder.

## Quick start (Google Colab)

1. Open [`notebooks/01_build_dataset.ipynb`](../notebooks/01_build_dataset.ipynb) in Colab (use the badge at the top) and click **Runtime → Run all**.
   It mounts your Google Drive, downloads 10 years of data from Zenodo (about 2 GB, roughly 10 minutes), and writes the outputs below.
2. Open [`notebooks/02_eda.ipynb`](../notebooks/02_eda.ipynb) for the charts and tables for the report's Data section.

To test the code in seconds without downloading anything, set `DRY_RUN = True` in notebook 01. It uses **fake** data.

## Sharing the data with teammates

Notebook 01 writes to `MyDrive/ENGR418_wildfire_data/`. To share it:

1. In Google Drive, share the `ENGR418_wildfire_data` folder with both teammates (Viewer access is enough).
2. Each teammate opens **Shared with me**, right-clicks the folder, and chooses **Organize → Add shortcut → My Drive**.
3. The same path now works in everyone's Colab, so `load_clean()` finds the data with no changes.

## Outputs

| File (in `ENGR418_wildfire_data/`) | What it is |
| --- | --- |
| `processed/fires_clean.parquet` | One row per fire: features, target, and bookkeeping columns |
| `processed/column_policy.csv` | Every column with its group, type, missing % and the reason for its group |
| `processed/build_report.json` | Rows removed at each step, class counts per year, columns needing review |
| `figures/*.png` | EDA charts from notebook 02 |

## Using the clean data (Roles B and C)

```python
from wildfire import load_clean, feature_columns, year_split

df, policy = load_clean()                              # finds the shared Drive folder on Colab
X_cols = feature_columns(policy, tier="at_discovery")  # leakage-safe default
train, val, test = year_split(df)                      # 2011-16 / 2017-18 / 2019-20

y_train = train["size_class"]   # 0 = Small (<10 ac), 1 = Medium (10-299 ac), 2 = Large (>=300 ac)
```

Notes for modelling:
- Categorical columns (state, land cover, owner) have the pandas `category` dtype. One-hot encode them, or use XGBoost or LightGBM with `enable_categorical=True`.
- Missing values are left as `NaN` on purpose. Impute inside your pipeline, fitted on the training split only.
- `log_fire_size` is available as an optional regression target.

## What the cleaning does

| Step | Rule |
| --- | --- |
| Scope | Years 2011–2020, states AZ CA CO ID MT NM NV OR UT WA WY (change in `src/wildfire/config.py`) |
| Duplicates | Keep the first row per `FOD_ID` |
| Invalid rows | Remove rows with missing or negative `FIRE_SIZE`, missing or impossible coordinates, or an unparseable discovery date |
| Types | Text that holds numbers becomes numbers; code columns (state, land cover …) become categories |
| Missing codes | `-9999` and `-32768` become `NaN` |
| Useless columns | Drop columns that are completely empty or constant |
| Target | `size_class` from final acres: under 10 Small, 10–299.99 Medium, 300+ Large (NWCG classes A–B, C–D, E–G) |
| New features | `doy_sin`, `doy_cos`, `month`, `weekday`, `discovery_hour` |

## Leakage policy

A model may only use what is known **when the fire is reported**. Every column is put into a group by rules in [`src/wildfire/columns.py`](../src/wildfire/columns.py):

| Group | Used as a feature? | Examples |
| --- | --- | --- |
| `static` | yes | location, terrain, land cover, human footprint, climate normals, calendar |
| `weather` | yes, in the `at_discovery` tier | gridMET `vpd`, `erc`, `bi`, `fm100`, wind, `NDVI` on the discovery date |
| `retrospective` | only in the `retrospective` tier | `*_5D_*` windows that may include days **after** ignition |
| `outcome` | **never** (kept for EDA) | `FIRE_SIZE_CLASS`, `CONT_*` (containment), `NWCG_*CAUSE*`, evacuations |
| `drop` | **never** (removed) | `MTBS_*` and `ICS_209*` (these exist only for big fires: a hidden size leak), fire names, record IDs |
| `review` | **never** until a rule is added | anything the rules don't recognise |

Feature tiers: `static` · `at_discovery` (static + weather, **recommended**) · `retrospective` (adds 5-day windows; report only as an upper bound).
`feature_columns()` calls `assert_no_leakage()`, so a leaky column cannot slip into a feature list unnoticed.

**After the first real run:** check the `review` list in notebook 01 (section 2) and add a rule for each column worth keeping.

## Running the tests

```bash
pip install -r requirements.txt
python -m pytest -q
```

The tests build the dataset from small fake files and check the cleaning, the size-class boundaries, the leakage rules and the year split.

## Source

Pourmohamad, Y. et al. (2024). *Physical, social, and biological attributes for improved understanding and prediction of wildfires: FPA FOD-Attributes dataset.* Earth System Science Data, 16, 3045–3060. Data: https://doi.org/10.5281/zenodo.8381129
