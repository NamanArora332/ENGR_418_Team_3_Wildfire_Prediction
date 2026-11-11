# ENGR 418 — Wildfire size prediction: annotated benchmark

## What it does

This project uses the supplied **`2020_FPA_FOD_cons.csv`** to predict a wildfire's **final size class** from location, discovery date, geography, land cover and, optionally, retrospective environmental attributes. This is **retrospective research**, not an operational wildfire forecasting system.

**Classes** (team-defined, not Coffield's original thresholds):

| Class | Acres |
|---|---|
| Small | `0 <= FIRE_SIZE < 10` |
| Medium | `10 <= FIRE_SIZE < 300` |
| Large | `FIRE_SIZE >= 300` |

The attached 2020 CSV contains **73,221 records** and **308 columns**. The script keeps one row per `FOD_ID`, leaving **73,213** distinct fires. All records belong to **2020**. The 1992–2014 / 2015–2017 / 2018–2020 split in our broader project plan **cannot be run on this year-only file**.

## Five stages and references to ENGR 418 lecture notes

The Python script has explanatory comments next to essentially every executable statement, with course-topic references. The references below are to Dr. Ling Bai's provided lecture PDFs and practice notebooks, *not* copied code from the lecture files.

| Stage | How this script uses it | Lecture connection |
|---|---|---|
| 1 — Data | Read data, check identifiers/targets, prepare feature matrix `X` and target `y`, check class imbalance | **Lecture 3:** samples, features, targets, Pandas; Lecture 3 practice solution |
| 2 — Model functions | Majority reference, four-leaf decision tree, multinomial logistic regression, XGBoost | **Lecture 7:** multiclass logistic/softmax; XGBoost is **beyond** the uploaded slides |
| 3 — Loss / objective | Logistic-regression cross-entropy; XGBoost multinomial log-loss, tree regularization; selection by macro-F1/F2 | **Lectures 4 and 7:** objective minimization and multiclass cross-entropy |
| 4 — Optimization/training | Fit on training dates only; tune the XGBoost warning threshold on validation only | **Lecture 4:** optimization; **Lecture 8:** feature encoding/scaling |
| 5 — Evaluation | Chronological holdout, confusion matrix, macro-F1, balanced accuracy, large-fire precision/recall, average precision | **Lectures 6 and 7:** evaluation, confusion matrix, precision, recall and F1 |

**Lecture 9** covers PCA, but PCA was **not** used because retaining interpretable terrain/land-cover variables is more useful for this comparison. The XGBoost implementation is an original extension beyond the supplied introductory lecture methods.

## Data availability and leakage safeguards

**Default policy (conservative)**: latitude, longitude, discovery day (as sine/cosine), elevation, slope, aspect, TRI, TPI, pre-2020 human modification index, state, and land-cover codes (the source's 2019 land-cover layer). This is a *candidate* pre-ignition feature set; geographic and source-vintage assumptions should still be verified and documented before asserting first-report availability.

**Optional policy (retrospective)**: adds EVH/EVC vegetation fields, annual climate indicators and gridMET `*_Normal` fields. The FPA FOD-Attributes documentation indicates that some normals use a **1990–2020** reference period and vegetation layers can include **2020**, so these fields are **not guaranteed to have been known when an individual 2020 fire started**. Do not label this mode as real-time or leakage-free.

The program excludes the final-size class, containment date, estimated costs, after-ignition multi-day weather (`*_5D_*`), unverified same-day observational weather and cause fields. It fits imputation, scaling and one-hot encoding **only on the training split** using scikit-learn `Pipeline` and `ColumnTransformer`.

**This is not an exact Coffield et al. replication.** The conservative four-leaf tree uses `Elevation` and `Slope`. The optional retrospective tree uses VPD normal and vegetation height as *proxies*. Coffield's original spruce-forest share and prediction-time weather setup are not reproduced here. Always label comparisons accordingly.

## How to run on Windows, macOS or Linux

1. Download or clone the team's GitHub repository and put this script, `requirements.txt`, and the small `MODEL_README.md` in its root.
2. Put `2020_FPA_FOD_cons.csv` into the `data/` folder on your own machine. **Do not commit the approximately 160 MB dataset to GitHub**; `.gitignore` excludes it. Alternatively, pass an absolute CSV path.
3. In a terminal at the repository root, run:

```bash
python -m pip install -r requirements.txt
python train_wildfire_model.py --data data/2020_FPA_FOD_cons.csv --output results
```

To evaluate the **optional retrospective features separately**:

```bash
python train_wildfire_model.py --data data/2020_FPA_FOD_cons.csv --output results_retrospective --retrospective-features
```

From the environment where we tested the project, the CSV also works with:

```bash
python train_wildfire_model.py --data "/mnt/data/2020_FPA_FOD_cons.csv"
```

To run basic checks:

```bash
python -m unittest discover -s tests -v
```

## Splits and methodology

The code groups rows by their `DISCOVERY_DATE`, then picks cut points nearest to **60% training, 20% validation and 20% test** by cumulative number of fires, never placing the same day in more than one group. Actual dates in the uploaded data:

| Split | Dates | Fires | Large fires |
|---|---|---:|---:|
| Train | Jan 1–Jul 18, 2020 | 43,989 | 494 |
| Validation | Jul 19–Sep 10, 2020 | 14,619 | 445 |
| Test | Sep 11–Dec 31, 2020 | 14,605 | 143 |

Temporal distributions differ strongly within a wildfire season, so this is a challenging holdout. A full multi-year evaluation and regional holdout would be stronger future work. **Do not treat the 2020 findings as proof of deployment accuracy in future years.**

All models fit on the **training** set only. The XGBoost large-fire warning threshold is chosen using **validation F2** (large-fire recall has more weight than precision). Final test metrics are then computed once using that chosen threshold. Metrics report how the decision threshold trades off false alarms and missed large fires. The experiment should be repeated across additional years before making real-world claims.

## Verified results on this uploaded CSV (120 boosting trees; random seed 42)

The following results are **measured**, not expected or invented. They are supplied for team review and are initial baselines, not final conclusions.

**Conservative feature policy (default):**

| Model | Test macro-F1 | Large-fire recall | Large-fire precision |
|---|---:|---:|---:|
| Majority class | 0.3137 | 0.0% | 0.0% |
| Four-leaf terrain tree | 0.0775 | 71.3% | 2.3% |
| Multinomial logistic regression | **0.3738** | **58.0%** | 3.4% |
| XGBoost, validation-tuned warning | 0.3373 | 49.0% | 5.1% |

**Optional retrospective feature policy (not first-report-safe):**

| Model | Test macro-F1 | Large-fire recall | Large-fire precision |
|---|---:|---:|---:|
| Majority class | 0.3137 | 0.0% | 0.0% |
| Four-leaf proxy tree | 0.3448 | 4.9% | 11.1% |
| Multinomial logistic regression | **0.3998** | **44.1%** | 4.3% |
| XGBoost, validation-tuned warning | 0.3442 | 32.2% | 6.0% |

The cost of high recall is a high false-alarm rate, as shown by low large-fire precision. The complex model **has not outperformed** multinomial logistic regression here. Improvements should be tested using **validation metrics**, not repeated peeking at the held-out test period.

## Generated outputs

- `results/model_metrics.csv` and `results_retrospective/model_metrics.csv`: one row per model and split.
- `split_class_counts.csv`: inspect major changes in the class distribution over time.
- `experiment_metadata.json`: data, features, split dates, number of trees and tuned threshold.
- `confusion_matrix.png`: count mistakes by predicted and actual size class.
- `large_fire_precision_recall.png`: large-fire detection vs. false alarms.
- `feature_importance.png`: XGBoost split-based feature importance (not causal inference).
- `xgboost_model.joblib`: optional trained model artifact. **Not** committed; regenerated on demand.

## Academic integrity and credits

The team must independently read, test, modify and explain all code and results. The code was prepared with ChatGPT assistance from the attached lecture files, user-provided CSV and public dataset documentation, then tested locally. Record this assistance in each team member's **individual Generative AI Use Declaration and Reflection** along with examples of verification and modifications. Lecture slides and solutions themselves should not be redistributed publicly without permission.

**External dataset reference:** Pourmohamad et al. (2024), *Physical, social, and biological attributes for improved understanding and prediction of wildfires: FPA FOD-Attributes dataset*, Earth System Science Data, https://doi.org/10.5194/essd-16-3045-2024 .

**Related baseline:** Coffield et al. (2019), *Machine learning to predict final fire size at the time of ignition*. Our tree is an adaptation/proxy, not an exact reproducibility claim.
