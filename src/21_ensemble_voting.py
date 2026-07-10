import os
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

TRAIN_FILE = "data/processed/train_selected.csv"
TEST_FILE = "data/processed/test_selected.csv"

OUTPUT_DIR = "results/ensemble_voting"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    objective="multi:softprob",
    num_class=len(encoder.classes_),
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

lgbm = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=64,
    objective="multiclass",
    random_state=42,
    n_jobs=-1
)

ensemble = VotingClassifier(
    estimators=[
        ("rf", rf),
        ("xgb", xgb),
        ("lgbm", lgbm)
    ],
    voting="soft",
    n_jobs=-1
)

print("Обучение на Ensemble Voting модел...")
ensemble.fit(X_train, y_train_encoded)

print("Prediction...")
y_pred = ensemble.predict(X_test)

accuracy = accuracy_score(y_test_encoded, y_pred)

report = classification_report(
    y_test_encoded,
    y_pred,
    target_names=encoder.classes_
)

print(f"Ensemble Accuracy: {accuracy:.4f}")
print(report)

with open(os.path.join(OUTPUT_DIR, "ensemble_classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

joblib.dump(
    ensemble,
    "models/ensemble_voting_model.pkl"
)

print("Ensemble моделът е готов.")