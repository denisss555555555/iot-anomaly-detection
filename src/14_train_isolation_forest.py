import os
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

TRAIN_FILE = "data/processed/train_selected.csv"
TEST_FILE = "data/processed/test_selected.csv"

OUTPUT_DIR = "results/isolation_forest"
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

train_df["binary_label"] = train_df["label"].apply(
    lambda x: 0 if x == "BenignTraffic" else 1
)

test_df["binary_label"] = test_df["label"].apply(
    lambda x: 0 if x == "BenignTraffic" else 1
)

normal_train = train_df[
    train_df["binary_label"] == 0
]

X_train = normal_train.drop(
    columns=["label", "binary_label"]
)

X_test = test_df.drop(
    columns=["label", "binary_label"]
)

y_test = test_df["binary_label"]

model = IsolationForest(
    contamination=0.1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train)

predictions = model.predict(X_test)

predictions = [0 if x == 1 else 1 for x in predictions]

report = classification_report(
    y_test,
    predictions
)

print(report)

report_path = os.path.join(
    OUTPUT_DIR,
    "isolation_forest_report.txt"
)

with open(report_path, "w") as file:
    file.write(report)

print("Isolation Forest evaluation е готов.")