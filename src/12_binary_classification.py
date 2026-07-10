import os
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier

TRAIN_FILE = "data/processed/train_selected.csv"
TEST_FILE = "data/processed/test_selected.csv"

OUTPUT_DIR = "results/binary_classification"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

train_df["binary_label"] = train_df["label"].apply(
    lambda x: "Normal" if x == "BenignTraffic" else "Attack"
)

test_df["binary_label"] = test_df["label"].apply(
    lambda x: "Normal" if x == "BenignTraffic" else "Attack"
)

X_train = train_df.drop(columns=["label", "binary_label"])
y_train = train_df["binary_label"]

X_test = test_df.drop(columns=["label", "binary_label"])
y_test = test_df["binary_label"]

encoder = LabelEncoder()

y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

model = LGBMClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train_encoded)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test_encoded, y_pred)

print(f"\nBinary Accuracy: {accuracy:.4f}")

report = classification_report(
    y_test_encoded,
    y_pred,
    target_names=encoder.classes_
)

print(report)

report_path = os.path.join(
    OUTPUT_DIR,
    "binary_classification_report.txt"
)

with open(report_path, "w") as file:
    file.write(report)

joblib.dump(
    model,
    "models/binary_lightgbm_model.pkl"
)

print("Binary classification е готов.")