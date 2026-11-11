# ENGR 418 Team 3 — Wildfire Size Prediction

> **Status:** Course research prototype (not an operational wildfire warning system).  
> **Course:** ENGR 418, Applied Machine Learning for Engineers, UBC Okanagan.  
> **Research goal:** Predict a reported wildfire's **final size class** using candidate features that could be available at discovery, while comparing understandable baselines against more complex classifiers.

## Research objective and scope

We investigate whether more complex machine-learning classifiers improve detection of **large fires (≥300 acres)** compared with simpler models, and whether those gains are meaningful once class imbalance and time-dependent generalization are considered.

This prototype evaluates **three team-defined classes**, which are **not** the original Coffield et al. (2019) class boundaries:

| Label | Final burned area |
| --- | --- |
| Small | `<10` acres |
| Medium | `10` to `<300` acres |
| Large | `≥300` acres |

### Important limitations

- **Dataset:** Only the 2020 portion of FPA FOD-Attributes supplied to the team (`2020_FPA_FOD_cons.csv`), not a multi-year validation set.
- **Prediction-time validity:** The default models exclude known post-ignition and retrospectively aggregated fields. Exact vintages of static datasets, including land cover and human modification, still require final source-level verification. The optional retrospective mode **must not** be reported as a real-time predictor.
- **Reproduction claim:** The four-leaf tree is a **Coffield-inspired proxy**, not a faithful replication of the original Coffield study (different data, predictors, and target thresholds).
- **Utility:** Model scores are preliminary; these models must not be used for operational fire response.

## Dataset and how to obtain it

- **Original source:** Pourmohamad et al. (2024), FPA FOD-Attributes, *Earth System Science Data*, DOI: https://doi.org/10.5194/essd-16-3045-2024
- **Team input:** `2020_FPA_FOD_cons.csv` (2020 subset supplied for the project).
- **Availability:** The exact team subset is **not bundled here**. Locate the dataset from its official distribution or obtain the prepared subset from an authorized teammate.
- Put the file at `data/2020_FPA_FOD_cons.csv`. Its contents are intentionally excluded from Git.
- See [`data/README.md`](data/README.md) for input and reproducibility notes; the optional `data/download_data.py` downloader requires a verified direct CSV URL.

The provided file contains **73,221 records** with 308 columns. The evaluation deduplicates by `FOD_ID` and retains **73,213 fires**, all from 2020. The train / validation / test sets are split by entire discovery dates (roughly 60/20/20), rather than randomly mixing dates.

## Repository structure

```text
.
├── README.md                       # Main project documentation
├── CONTRIBUTING.md                 # Issues, branches, reviews, contribution expectations
├── MODEL_README.md                 # Technical model and lecture-level details
├── requirements.txt                # Python dependencies
├── train_wildfire_model.py         # Annotated primary training entry point
├── feature_availability_audit.csv  # Feature-vintage / possible leakage audit
├── data/                           # Download instructions and input helper (NO raw CSV in Git)
├── notebooks/                      # Exploratory notebook
├── src/                            # Reproduction and results-table utilities
├── tests/                          # Automated smoke and safety tests
├── results/                        # Measured conservative results (.csv, .json, .png)
├── results_retrospective/          # Optional clearly-labeled non-operational experiment
├── report/                         # Draft report outline and figure selection notes
└── .github/workflows/              # Lightweight continuous integration
```

## Quick start (Windows / macOS / Linux)

Use Python 3.11 or 3.12 and work in the **repository root**:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

**One command to regenerate the model results table and graphs** (after placing the CSV under `data/`):

```bash
python -m src.reproduce --data data/2020_FPA_FOD_cons.csv
```

This calls `train_wildfire_model.py` with the default feature policy and seed **42**, then writes `results/model_metrics.csv`, `results/summary.md`, metadata and diagnostic figures. It also saves a local `.joblib` model that Git will ignore. To run the original script directly:

```bash
python train_wildfire_model.py --data data/2020_FPA_FOD_cons.csv --output results --seed 42 --trees 120
```

**Optional research-only retrospective experiment:**

```bash
python -m src.reproduce --data data/2020_FPA_FOD_cons.csv --retrospective-features
```

The optional mode adds features that may not have existed when the fires were reported. **Do not compare its scores as an operational forecast** without a feature availability investigation.

### Model methods (ENGR 418 lecture connections)

| Component | Course connection | This project |
| --- | --- | --- |
| Data and `X`, `y` | Lecture 3 | pandas, cleaning, target classes |
| Loss / objective | Lectures 4 and 7 | cross-entropy for multinomial logistic regression; tree criteria for tree models |
| Classification / evaluation | Lectures 6 and 7 | confusion matrix, balanced accuracy, macro-F1, precision and recall |
| Feature engineering | Lecture 8 | seasonal sin/cos, one-hot encoding, missing-data imputation |
| Dimensionality reduction | Lecture 9 | discussed but intentionally not needed for the interpretable feature set |
| Advanced extension | Beyond supplied notes | XGBoost gradient-boosted trees |

See [`MODEL_README.md`](MODEL_README.md) and in-line comments in `train_wildfire_model.py` for implementation rationale.

## Recorded results from the supplied 2020 data

The figures below are **initial measured test-set results**, NOT expected results or claimed evidence of operational forecasting performance. CSV files and images are in [`results/`](results/).

| Model | Test macro-F1 | Large-fire recall | Large-fire precision |
| --- | ---: | ---: | ---: |
| Majority class | 0.314 | 0.0% | 0.0% |
| Four-leaf terrain tree | 0.077 | 71.3% | 2.3% |
| Multiclass logistic regression | **0.374** | **58.0%** | 3.4% |
| XGBoost (validation-tuned large-fire threshold) | 0.337 | 49.0% | 5.1% |

**Key finding:** On this single-year chronological test, the multinomial logistic model has higher macro-F1 and large-fire recall than the tuned XGBoost model. The four-leaf tree gets high recall by triggering excessive false alarms. Large-fire **precision remains low for every model** because large fires are rare; it is important to show recall *and* precision in the report. This result does **not** establish that logistic regression is generally superior across years or regions.

![Large-fire precision-recall curves](results/large_fire_precision_recall.png)

Explore the [`confusion matrix`](results/confusion_matrix.png), [`feature importance chart`](results/feature_importance.png), [`split counts`](results/split_class_counts.csv), and exact [`model metrics`](results/model_metrics.csv). Generated model artifacts (`*.joblib`) are intentionally not committed.

## Team roles and contribution record

| Role | Assigned person | Responsibilities | Evidence |
| --- | --- | --- | --- |
| Data and quality lead | **To confirm with team** | Data extraction, schema, leakage audit | Linked issues / commits / PRs |
| Baseline and evaluation lead | **To confirm with team** | Baselines, splits, metrics, figures | Linked issues / commits / PRs |
| Advanced models and documentation lead | **To confirm with team** | XGBoost, evaluation extensions, README | Linked issues / commits / PRs |

**Replace the placeholders only when assignments are agreed.** All three members should create meaningful commits on their own GitHub accounts and request reviews on pull requests. Do not invent contributions or assign credit solely from the existence of this starter code. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Reproduction and grading checklist

- [x] Executable model script and dependency specification
- [x] Fixed random seed 42 and date-grouped partitions
- [x] One-command reproduction that exports CSV results and PNG figures
- [x] Recorded initial results and an explicitly labeled feature audit
- [x] Data, notebooks, src, results, report folders (populated with real files)
- [ ] Each member independently verifies the code and records contributions
- [ ] Final feature availability audit and instructor feedback
- [ ] Multi-year / held-out-region validation (if data and time permit)
- [ ] Final 4–6-page technical report, three-minute video, and individual GenAI declarations

## References, academic integrity and AI disclosure

1. Pourmohamad et al. (2024), *Physical, social, and biological attributes for improved understanding and prediction of wildfires: FPA FOD-Attributes dataset*. https://doi.org/10.5194/essd-16-3045-2024
2. Coffield et al. (2019), *Machine learning to predict final fire size at the time of ignition*. Refer to the publication for its exact feature windows, targets and evaluation.
3. UBC Okanagan, ENGR 418 lecture materials by Dr. Ling Bai (course materials; **do not** publish lecture PDFs or practice solutions without permission).

**Generative AI use:** Initial training code, documentation and experiment scaffolding were developed with ChatGPT assistance using the supplied dataset and lecture materials. Team members must independently review and validate all outputs, document edits and checks, and complete their **individual** Generative AI Use Declarations and Reflections. See [`report/README.md`](report/README.md).
