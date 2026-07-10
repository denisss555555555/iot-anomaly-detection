import os
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from lightgbm import LGBMClassifier

TRAIN_FILE = "data/processed/train_selected.csv"
TEST_FILE = "data/processed/test_selected.csv"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 70)
print("Зареждане на dataset-и")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

LABEL_COLUMN = "label"

X_train = train_df.drop(columns=[LABEL_COLUMN])
y_train = train_df[LABEL_COLUMN]

X_test = test_df.drop(columns=[LABEL_COLUMN])
y_test = test_df[LABEL_COLUMN]

print("Encoding labels...")

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("Обучение на LightGBM...")

model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=64,
    objective="multiclass",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train_encoded)

print("Prediction...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test_encoded, y_pred)

print("\n" + "=" * 70)
print("LIGHTGBM РЕЗУЛТАТИ")
print("=" * 70)

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test_encoded,
        y_pred,
        target_names=label_encoder.classes_
    )
)

print("\nЗаписване на модела...")

joblib.dump(model, "models/lightgbm_model.pkl")

print("Готово.")