"""ENGR 418 Team 3: annotated wildfire size-class prediction benchmark.

Lecture map (source: uploaded Dr. Ling Bai ENGR 418 notes):
  L3: inspect samples, features X, labels y, missing values, and Pandas.
  L4: parameters, optimization, minimizing a cost/objective function.
  L6: classification, decision rules, validation, confusion matrix.
  L7: three-class softmax regression, cross-entropy, precision/recall/F1.
  L8: feature selection, feature transformations, one-hot encoding.
  L9: PCA is discussed in class but intentionally NOT applied here.

XGBoost / tree boosting is an extension BEYOND the supplied lectures.
Caution: climate-normal feature vintage must be verified before operational use.
"""

from __future__ import annotations  # Allow forward-compatible type annotations.

import argparse  # Read command-line settings such as dataset location.
import json  # Serialize experiment metadata into a reproducible JSON record.
from pathlib import Path  # Manipulate file paths across Windows, macOS, and Linux.

import joblib  # Save fitted models so results can be reproduced or inspected.
import matplotlib  # Configure figure generation without requiring a desktop GUI.
matplotlib.use("Agg")  # Select the non-interactive backend before pyplot is imported.
import matplotlib.pyplot as plt  # Save the evaluation charts required in the report.
import numpy as np  # Perform numerical and array calculations (Lecture 3).
import pandas as pd  # Read tabular CSV data into a DataFrame (Lecture 3).
from sklearn.compose import ColumnTransformer  # Apply different preprocessing by feature type (Lecture 8).
from sklearn.dummy import DummyClassifier  # Establish a majority-class reference prediction.
from sklearn.impute import SimpleImputer  # Learn missing-value replacements from TRAINING DATA ONLY.
from sklearn.linear_model import LogisticRegression  # Lecture 7: multinomial logistic-regression baseline.
from sklearn.metrics import (  # Lecture 7: evaluate classifiers with several complementary metrics.
    ConfusionMatrixDisplay,  # Display correct and incorrect class predictions.
    accuracy_score,  # Record accuracy, not as the sole score on imbalanced data.
    average_precision_score,  # Evaluate precision-recall quality for rare large fires.
    balanced_accuracy_score,  # Balance performance across the three fire-size classes.
    confusion_matrix,  # Count predicted vs. actual classes.
    f1_score,  # Use macro F1 as the primary model-comparison metric.
    fbeta_score,  # Weight recall more than precision for a large-fire decision threshold.
    precision_recall_curve,  # Calculate the curve for detecting large fires.
    precision_score,  # Quantify false alarms among predicted large fires.
    recall_score,  # Quantify missed large fires.
)
from sklearn.pipeline import Pipeline  # Combine preprocessing and model fitting safely.
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Lecture 8: encode categories and scale features.
from sklearn.tree import DecisionTreeClassifier  # Create a four-leaf, Coffield-inspired proxy model.
from xgboost import XGBClassifier  # Advanced extension: gradient-boosted decision trees.

SEED = 42  # Fix stochastic behavior for comparable experiments.
CLASS_NAMES = ["Small (<10 acres)", "Medium (10–299 acres)", "Large (>=300 acres)"]  # Declare output classes.
LABELS = [0, 1, 2]  # Encode the three wildfire-size targets as integers.

# Lecture 8: use a deliberately small, auditable set of candidate predictors.
NUMERIC_FEATURES = [  # Do not include a feature unless its availability can be justified.
    "LATITUDE",  # Fire location known at or near first report.
    "LONGITUDE",  # Fire location known at or near first report.
    "discovery_doy_sin",  # Cyclical seasonal feature engineered from discovery date (Lecture 8).
    "discovery_doy_cos",  # Second seasonal feature removes a discontinuity at New Year.
    "Elevation",  # Static topographic elevation.
    "Slope",  # Static terrain steepness.
    "Aspect",  # Static slope orientation; retained as a simple baseline feature.
    "TRI",  # Terrain ruggedness index.
    "TPI",  # Topographic position index.
    "GHM",  # Global human-modification index (pre-2020 source); verify provenance.
]  # Conservative input policy excludes 2020-vintage or retrospectively aggregated fields.
RETROSPECTIVE_NUMERIC = [  # OPTIONAL: not demonstrably available at first report in 2020.
    "EVH",  # Vegetation height: a 2020-vintage layer could be future-looking.
    "EVC",  # Vegetation cover: a 2020-vintage layer could be future-looking.
    "Annual_precipitation",  # Annual climate aggregate: check its reference years.
    "Aridity_index",  # Climate aggregate: check the reference period.
    "vpd_Normal",  # Dataset normals may summarize 1990-2020 retrospectively.
    "tmmx_Normal",  # Temperature normal subject to retrospective source timing.
    "pr_Normal",  # Precipitation normal subject to retrospective source timing.
    "erc_Normal",  # Fire-danger normal subject to retrospective source timing.
]  # Explore only as a distinct retrospective feature-set experiment.
CATEGORICAL_FEATURES = ["STATE", "Land_Cover"]  # Location group and land-cover code (Lecture 8).
BASELINE_NUMERIC_FEATURES = ["Elevation", "Slope"]  # Conservative 2-feature tree, NOT Coffield reproduction.
RETROSPECTIVE_BASELINE_FEATURES = ["vpd_Normal", "EVH"]  # Retrospective Coffield-inspired proxy (not exact).
REQUIRED_COLUMNS = sorted(set(NUMERIC_FEATURES + RETROSPECTIVE_NUMERIC + CATEGORICAL_FEATURES + BASELINE_NUMERIC_FEATURES)  # Read necessary features only.
                          - {"discovery_doy_sin", "discovery_doy_cos"}  # These seasonal features are created later.
                          | {"DISCOVERY_DATE", "FIRE_SIZE", "FOD_ID"})  # Read labels, discovery dates, and identifiers.

# Important leakage guard: NEVER add post-fire measurements to the candidate list.
FORBIDDEN_EXACT = {"FIRE_SIZE", "FIRE_SIZE_CLASS", "CONT_DATE", "CONT_DOY", "CONT_TIME",  # Target and containment outcomes.
                   "DISCOVERY_TIME", "NWCG_GENERAL_CAUSE", "NWCG_CAUSE_CLASSIFICATION", "Evacuation",  # Untested timing/availability.
                   "pr", "vpd", "tmmn", "tmmx", "rmin", "rmax", "erc", "bi", "NDVI-1day"}  # Potentially later observations.


def audit_feature_names(numeric_features: list[str]) -> None:
    """Fail early if a feature violates the approved first-report design."""  # Guard against accidental leakage.
    selected = numeric_features + CATEGORICAL_FEATURES  # Collect every model predictor.
    assert len(selected) == len(set(selected)), "Duplicate feature names were selected."  # Prevent ambiguous preprocessing.
    for name in selected:  # Inspect every feature one at a time.
        if name in FORBIDDEN_EXACT or "_5D_" in name or name.startswith("CONT_"):  # Identify banned future targets.
            raise ValueError(f"Potential target/time leakage in selected feature: {name}")  # Stop before training.


def load_and_prepare(csv_path: Path, numeric_features: list[str]) -> pd.DataFrame:
    """Read the supplied 2020 FPA FOD CSV and construct the three class labels."""  # Lecture 3: X and y.
    if not csv_path.is_file():  # Validate the file path before attempting a long read.
        raise FileNotFoundError(f"Dataset not found: {csv_path}")  # Report an actionable problem.
    df = pd.read_csv(csv_path, usecols=lambda column: column in REQUIRED_COLUMNS, low_memory=False)  # Load just the audited columns.
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))  # Compare CSV headers with the required schema.
    if missing:  # The original dataset must contain all selected features.
        raise ValueError(f"Missing required columns in CSV: {missing}")  # Avoid silent mismatches.
    df["DISCOVERY_DATE"] = pd.to_datetime(df["DISCOVERY_DATE"], errors="coerce")  # Convert dates (Lecture 3).
    df["FIRE_SIZE"] = pd.to_numeric(df["FIRE_SIZE"], errors="coerce")  # Convert target fire area to acres.
    df = df.dropna(subset=["DISCOVERY_DATE", "FIRE_SIZE"]).copy()  # Exclude rows without time or target.
    df = df.loc[df["FIRE_SIZE"] >= 0].copy()  # Reject physically invalid negative fire sizes.
    df = df.drop_duplicates(subset=["FOD_ID"], keep="first")  # Keep one record per original fire identifier.
    df["target"] = np.select([df["FIRE_SIZE"] < 10, df["FIRE_SIZE"] < 300], [0, 1], default=2).astype(int)  # Assign classes.
    day = df["DISCOVERY_DATE"].dt.dayofyear.astype(float)  # Extract calendar information available at discovery.
    df["discovery_doy_sin"] = np.sin(2.0 * np.pi * day / 366.0)  # Encode the 2020 leap-year seasonal cycle.
    df["discovery_doy_cos"] = np.cos(2.0 * np.pi * day / 366.0)  # Complete the cyclical day representation.
    for column in numeric_features:  # Ensure numerical predictors truly contain numbers.
        df[column] = pd.to_numeric(df[column], errors="coerce")  # Turn invalid entries into train-imputable NaN.
    df["Land_Cover"] = df["Land_Cover"].astype("Int64").astype("string").fillna("Unknown")  # Treat cover codes as categories.
    df["STATE"] = df["STATE"].astype("string").fillna("Unknown")  # Keep state codes categorical.
    if len(df) < 100:  # Ensure there are enough records for meaningful splits.
        raise ValueError("Too few usable wildfire records after cleaning.")  # Avoid misleading training.
    return df  # Return the annotated feature table and labels.


def chronological_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Create 60/20/20 time-ordered partitions with no shared discovery date."""  # Honest 2020-only validation.
    counts = df.groupby("DISCOVERY_DATE", sort=True).size()  # Count records for each distinct discovery day.
    cumulative = counts.cumsum()  # Determine where the requested sample fractions are reached.
    train_end = cumulative.index[np.searchsorted(cumulative.to_numpy(), 0.60 * len(df))]  # End training at about 60%.
    valid_end = cumulative.index[np.searchsorted(cumulative.to_numpy(), 0.80 * len(df))]  # End validation at about 80%.
    if train_end >= valid_end:  # Require time periods to be strictly ordered.
        raise ValueError("Not enough distinct dates for chronological validation.")  # Explain why splitting failed.
    train = df.loc[df["DISCOVERY_DATE"] <= train_end].copy()  # Earlier fires are the training cases.
    valid = df.loc[(df["DISCOVERY_DATE"] > train_end) & (df["DISCOVERY_DATE"] <= valid_end)].copy()  # Later validation.
    test = df.loc[df["DISCOVERY_DATE"] > valid_end].copy()  # Latest fires are reserved for FINAL evaluation.
    for name, frame in [("train", train), ("validation", valid), ("test", test)]:  # Inspect each temporal group.
        if len(frame) == 0 or frame["target"].nunique() < 3:  # Every group needs all classes for these metrics.
            raise ValueError(f"{name} split lacks samples or a fire-size class.")  # Avoid invalid evaluation.
    assert train["DISCOVERY_DATE"].max() < valid["DISCOVERY_DATE"].min()  # Stop same-day leakage between train and val.
    assert valid["DISCOVERY_DATE"].max() < test["DISCOVERY_DATE"].min()  # Stop same-day leakage between val and test.
    intervals = {"train_end": str(train_end.date()), "validation_end": str(valid_end.date()),  # Describe experiment periods.
                 "test_start": str(test["DISCOVERY_DATE"].min().date())}  # Record when the held-out test begins.
    return train, valid, test, intervals  # Supply disjoint partitions and their boundaries.


def make_preprocessor(numerical: list[str], categorical: list[str]) -> ColumnTransformer:
    """Fit missing-value handling, scaling and encoding ONLY on training rows."""  # Lecture 8.
    num_steps = Pipeline([("imputer", SimpleImputer(strategy="median")),  # Replace missing numbers using training medians.
                          ("scaler", StandardScaler())])  # Standardize numbers for logistic regression.
    if categorical:  # Some baselines intentionally contain no categorical predictors.
        cat_steps = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),  # Fill unknown categories from training.
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))])  # Encode nominal labels (Lecture 8).
        transformers = [("num", num_steps, numerical), ("cat", cat_steps, categorical)]  # Combine numerical and categorical.
    else:  # The two-feature proxy baseline uses only numerical variables.
        transformers = [("num", num_steps, numerical)]  # Avoid a transformer with zero input columns.
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)  # Exclude unapproved raw fields.


def build_models(seed: int, trees: int, numeric_features: list[str], proxy_features: list[str]) -> dict[str, Pipeline]:
    """Make lecture-aligned baselines and one advanced boosted-tree model."""  # Stage 2: model functions.
    proxy_prep = make_preprocessor(proxy_features, [])  # Fit proxy preprocessing within the model pipeline.
    full_prep = lambda: make_preprocessor(numeric_features, CATEGORICAL_FEATURES)  # Independent preprocessing per model.
    models = {  # Keep training and evaluation identical for fair model comparison.
        "majority": Pipeline([("prep", full_prep()), ("model", DummyClassifier(strategy="prior"))]),  # Majority benchmark.
        "four_leaf_proxy": Pipeline([("prep", proxy_prep),  # Two-input analogue, NOT exact Coffield replication.
                                     ("model", DecisionTreeClassifier(max_leaf_nodes=4, class_weight="balanced",  # 4-leaf tree.
                                                                       random_state=seed))]),  # Fix random tie-breaking.
        "multinomial_logistic": Pipeline([("prep", full_prep()),  # Lecture 7: learned three-class softmax model.
                                          ("model", LogisticRegression(max_iter=500, class_weight="balanced",  # Rare classes matter.
                                                                        random_state=seed))]),  # Reproducible solver.
        "xgboost": Pipeline([("prep", full_prep()),  # Advanced, nonlinear gradient-boosted tree model.
                              ("model", XGBClassifier(objective="multi:softprob", num_class=3,  # Three-class probabilities.
                                                      n_estimators=trees, max_depth=5, learning_rate=0.06,  # Conservative trees.
                                                      subsample=0.8, colsample_bytree=0.85,  # Regularize ensembles.
                                                      reg_lambda=2.0, min_child_weight=4, tree_method="hist",  # CPU-friendly.
                                                      n_jobs=4, random_state=seed, eval_metric="mlogloss"))]),  # Multiclass loss.
    }  # The lecture covers softmax/cross-entropy, while XGBoost's tree mechanism is an extension.
    return models  # Return unfitted pipelines ready for the training stage.


def adjusted_predictions(probabilities: np.ndarray, large_threshold: float | None = None) -> np.ndarray:
    """Predict all classes; optionally use a tuned large-fire warning threshold."""  # Lecture 7: decision boundaries.
    if large_threshold is None:  # Ordinary multiclass decision rule: select highest predicted probability.
        return probabilities.argmax(axis=1)  # Output class indices 0, 1, or 2.
    predicted_nonlarge = probabilities[:, :2].argmax(axis=1)  # Choose best small-vs-medium class first.
    return np.where(probabilities[:, 2] >= large_threshold, 2, predicted_nonlarge)  # Override when large risk is high.


def tune_large_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Use validation labels ONLY to maximize F2 for rare large-fire detection."""  # No test-set tuning.
    candidates = np.linspace(0.02, 0.70, 69)  # Search practical warning probabilities from 2% to 70%.
    scores = [fbeta_score(y_true == 2, probabilities[:, 2] >= value, beta=2, zero_division=0)  # F2 favors recall.
              for value in candidates]  # Compare warning thresholds only on validation data.
    best_index = int(np.argmax(scores))  # Choose the threshold with the highest validation F2.
    return float(candidates[best_index]), float(scores[best_index])  # Save selection and its justification.


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float | None = None) -> dict:
    """Measure whole-model quality and detection of rare large fires."""  # Lecture 7 evaluation.
    predictions = adjusted_predictions(probabilities, threshold)  # Apply the selected classification rule.
    return {  # Record multiple metrics to avoid misleading majority-class accuracy.
        "accuracy": float(accuracy_score(y_true, predictions)),  # Overall correct classification fraction.
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),  # Average recall by class.
        "macro_f1": float(f1_score(y_true, predictions, labels=LABELS, average="macro", zero_division=0)),  # Main metric.
        "large_recall": float(recall_score(y_true == 2, predictions == 2, zero_division=0)),  # Large fires found.
        "large_precision": float(precision_score(y_true == 2, predictions == 2, zero_division=0)),  # Warning reliability.
        "large_average_precision": float(average_precision_score(y_true == 2, probabilities[:, 2])),  # Ranking quality.
    }  # A strong model should report the entire quality/false-alarm trade-off.


def save_charts(model: Pipeline, y_true: np.ndarray, proba: np.ndarray, threshold: float, output: Path) -> None:
    """Create report-ready confusion matrix, precision-recall curve and feature importance."""  # Stage 5.
    predictions = adjusted_predictions(proba, threshold)  # Use the threshold chosen from validation only.
    counts = confusion_matrix(y_true, predictions, labels=LABELS)  # Get true-vs-predicted class counts.
    fig, ax = plt.subplots(figsize=(7, 5))  # Start a single readable confusion-matrix figure.
    ConfusionMatrixDisplay(counts, display_labels=["Small", "Medium", "Large"]).plot(ax=ax, cmap="Blues", colorbar=False)  # Plot.
    ax.set_title("XGBoost: 2020 chronological holdout")  # Explain which model/test this plot represents.
    fig.tight_layout()  # Reduce label overlap before saving.
    fig.savefig(output / "confusion_matrix.png", dpi=170)  # Save output for the ENGR 418 report.
    plt.close(fig)  # Release memory in a headless training run.
    precision, recall, _ = precision_recall_curve(y_true == 2, proba[:, 2])  # Large-fire probability ranking.
    fig, ax = plt.subplots(figsize=(7, 5))  # Start the precision-recall visualization.
    ax.plot(recall, precision, label="XGBoost (large-fire probability)")  # Show warning quality across thresholds.
    ax.axhline(np.mean(y_true == 2), linestyle="--", label="Large-fire prevalence")  # Draw no-skill reference.
    ax.set(xlabel="Large-fire recall", ylabel="Large-fire precision", title="Large-fire detection (test period)")  # Label axes.
    ax.legend(loc="best")  # Describe both lines on the graph.
    fig.tight_layout()  # Prevent axis titles from being cut off.
    fig.savefig(output / "large_fire_precision_recall.png", dpi=170)  # Save a second report figure.
    plt.close(fig)  # Close the precision-recall figure.
    importance = model.named_steps["model"].feature_importances_  # Extract relative tree-based feature importance.
    names = model.named_steps["prep"].get_feature_names_out()  # Match expanded feature names to model inputs.
    top = np.argsort(importance)[-15:]  # Select the 15 most important encoded features.
    fig, ax = plt.subplots(figsize=(9, 6))  # Start a separate feature-importance figure.
    ax.barh(np.asarray(names)[top], importance[top])  # Show model importance, NOT a causal effect.
    ax.set(xlabel="XGBoost feature importance", title="Top 15 encoded features")  # Clarify interpretation.
    fig.tight_layout()  # Make long feature names readable.
    fig.savefig(output / "feature_importance.png", dpi=170)  # Save for the Methods/Results report section.
    plt.close(fig)  # Clean up figure resources.


def parse_args() -> argparse.Namespace:
    """Define the user-facing command-line interface."""  # Make code easy to reproduce from GitHub.
    parser = argparse.ArgumentParser(description="ENGR 418 wildfire-size prediction benchmark")  # CLI help text.
    parser.add_argument("--data", type=Path, default=Path("data/2020_FPA_FOD_cons.csv"))  # Input CSV location.
    parser.add_argument("--output", type=Path, default=Path("results"))  # Folder for metrics and figures.
    parser.add_argument("--trees", type=int, default=120)  # Boosting iterations; reduce for quick experiments.
    parser.add_argument("--seed", type=int, default=SEED)  # Optional reproducibility seed override.
    parser.add_argument("--retrospective-features", action="store_true")  # Explicitly opt in to 2020-vintage-risk features.
    return parser.parse_args()  # Return parsed settings to main().


def main() -> None:
    """Execute the five stages of the project, with an untouched temporal test set."""  # End-to-end pipeline.
    args = parse_args()  # Read dataset and output options supplied by the user.
    numeric_features = NUMERIC_FEATURES + RETROSPECTIVE_NUMERIC if args.retrospective_features else NUMERIC_FEATURES  # Policy.
    proxy_features = RETROSPECTIVE_BASELINE_FEATURES if args.retrospective_features else BASELINE_NUMERIC_FEATURES  # Baseline.
    audit_feature_names(numeric_features)  # Check predictor names for prohibited post-fire data.
    if args.retrospective_features:  # Warn about the optional research-only feature policy.
        print("WARNING: Retrospective features may include information unavailable at fire discovery.")  # Clarify.
    args.output.mkdir(parents=True, exist_ok=True)  # Create output folder before saving files.
    df = load_and_prepare(args.data, numeric_features)  # Lecture 3: inspect and clean the supplied 2020 wildfire records.
    years = sorted(df["DISCOVERY_DATE"].dt.year.unique().tolist())  # Record which years are actually available.
    print(f"Loaded {len(df):,} distinct fires from years {years}.")  # Display genuine dataset coverage.
    train, valid, test, times = chronological_splits(df)  # Partition fire records by discovery date.
    print(f"Train: {len(train):,}; validation: {len(valid):,}; test: {len(test):,}; dates: {times}")  # Log proportions.
    for title, frame in [("train", train), ("validation", valid), ("test", test)]:  # Audit each partition's class balance.
        print(f"{title} class counts: {frame['target'].value_counts().sort_index().to_dict()}")  # Reveal rare-fire prevalence.
    xtrain, ytrain = train, train["target"].to_numpy()  # Assemble training features and numeric targets (Lecture 3).
    xvalid, yvalid = valid, valid["target"].to_numpy()  # Reserve a later validation period for tuning.
    xtest, ytest = test, test["target"].to_numpy()  # Hold out the final period until all choices are fixed.
    models = build_models(args.seed, args.trees, numeric_features, proxy_features)  # Stage 2: instantiate the four competing model functions.
    rows = []  # Accumulate model-comparison metrics as rows in a report table.
    chosen_threshold = None  # The threshold can be learned only after fitting the XGBoost model.
    chosen_validation_f2 = None  # Save threshold selection evidence in the metadata.
    for name, model in models.items():  # Train each model on identical rows for fair comparison.
        print(f"Training {name}...")  # Clearly identify the current training algorithm.
        columns = proxy_features if name == "four_leaf_proxy" else numeric_features + CATEGORICAL_FEATURES  # Inputs.
        model.fit(xtrain[columns], ytrain)  # Stage 4: training optimizes parameters using training examples ONLY.
        valid_proba = model.predict_proba(xvalid[columns])  # Evaluate on validation rows without refitting.
        test_proba = model.predict_proba(xtest[columns])  # Predict untouched future test rows.
        if name == "xgboost":  # Advanced model may use an explicit threshold for rare fires.
            chosen_threshold, chosen_validation_f2 = tune_large_threshold(yvalid, valid_proba)  # Tune on validation ONLY.
            print(f"Selected large-fire warning threshold: {chosen_threshold:.2f}, validation F2: {chosen_validation_f2:.4f}")  # Explain tuning.
        threshold = chosen_threshold if name == "xgboost" else None  # Keep baseline rules unchanged.
        for part, actual, prob in [("validation", yvalid, valid_proba), ("test", ytest, test_proba)]:  # Two comparisons.
            values = calculate_metrics(actual, prob, threshold)  # Compute the objective performance measures.
            rows.append({"model": name, "split": part, "threshold": threshold, **values})  # Save all scores.
            print(f"{name} {part}: macro-F1={values['macro_f1']:.4f}; large recall={values['large_recall']:.4f}; large precision={values['large_precision']:.4f}")  # Explain.
        if name == "xgboost":  # Only generate advanced-model diagnostic figures and model artifact.
            save_charts(model, ytest, test_proba, chosen_threshold, args.output)  # Make report-ready figures.
            joblib.dump(model, args.output / "xgboost_model.joblib")  # Archive the fitted preprocessing + model.
    pd.DataFrame(rows).to_csv(args.output / "model_metrics.csv", index=False)  # Save honest train/validation-independent scores.
    pd.DataFrame([{"split": name, "total": len(frame), "small": int((frame.target == 0).sum()),  # Class counts.
                   "medium": int((frame.target == 1).sum()), "large": int((frame.target == 2).sum())}  # Class counts.
                  for name, frame in [("train", train), ("validation", valid), ("test", test)]]).to_csv(  # Split table.
                      args.output / "split_class_counts.csv", index=False)  # Make temporal imbalance visible.
    metadata = {"data_file": args.data.name, "years": years, "rows": len(df), "split_dates": times,  # Experiment provenance.
                "classes": CLASS_NAMES, "features": numeric_features + CATEGORICAL_FEATURES,  # Document the model input.
                "excluded": "All *_5D_* future windows, same-day observations, outcomes, post-incident details, and causes",  # Leakage audit.
                "warning": "2020-only temporal evaluation; verify every input source vintage before operational deployment",
                "feature_policy": "retrospective" if args.retrospective_features else "conservative",  # Limitation.
                "large_threshold": chosen_threshold, "validation_f2": chosen_validation_f2,  # Validation result.
                "seed": args.seed, "xgb_n_estimators": args.trees}  # Reproducible training settings.
    (args.output / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")  # Save provenance.
    print(f"Done. Read model_metrics.csv and plots in: {args.output.resolve()}")  # Tell the user where results are.


if __name__ == "__main__":  # Execute training only when invoked as a script, not on import.
    main()  # Start the reproducible ENGR 418 experimental pipeline.
