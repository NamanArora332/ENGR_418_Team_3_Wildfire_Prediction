
# ENGR 418 - Team 3
# Wildfire Size Prediction - Starter Model Training
#The following trains a Random Forest classifier on wildfire records, using latitude and longitude as an initial baseline. 
#It assumes your team will provide a cleaned CSV at data/wildfires.csv with the columns FIRE_YEAR, FIRE_SIZE, LATITUDE, and LONGITUDE
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

print("Hello, World!")
print("Starting wildfire model training...")

DATA_PATH = Path("data/wildfires.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Add the cleaned wildfire dataset to data/wildfires.csv"
    )

# Stage 1: Load data
columns = ["FIRE_YEAR", "FIRE_SIZE", "LATITUDE", "LONGITUDE"]
df = pd.read_csv(DATA_PATH, usecols=columns).dropna()

# Project-defined size classes (acres)
df = df[df["FIRE_SIZE"] >= 0].copy()

df["target"] = pd.cut(
    df["FIRE_SIZE"],
    bins=[-1, 10, 300, float("inf")],
    labels=["Small", "Medium", "Large"],
    right=False,
)

# Temporal data splitting
train = df[df["FIRE_YEAR"] <= 2014]
val = df[df["FIRE_YEAR"].between(2015, 2017)]
test = df[df["FIRE_YEAR"].between(2018, 2020)]

if min(len(train), len(val), len(test)) == 0:
    raise ValueError("One or more time splits are empty.")

# Stage 2: Model function
# Initial location-only baseline.
# Future features must pass the ignition-time leakage audit.
features = ["LATITUDE", "LONGITUDE"]

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced_subsample",
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1,
)

# Stage 3: The classifier uses an internal split criterion
# (not a neural-network cross-entropy training objective).

# Stage 4: Train the model
model.fit(train[features], train["target"])

# Stage 5: Evaluate
for name, split in [("Validation", val), ("Test", test)]:
    predictions = model.predict(split[features])

    print(f"\n{name} Results:")
    print("Macro F1:", f1_score(
        split["target"],
        predictions,
        labels=["Small", "Medium", "Large"],
        average="macro",
        zero_division=0,
    ))

    print(classification_report(
        split["target"],
        predictions,
        labels=["Small", "Medium", "Large"],
        zero_division=0,
    ))

print("Training and evaluation completed!")
