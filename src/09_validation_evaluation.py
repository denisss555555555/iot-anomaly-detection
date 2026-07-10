import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

VALIDATION_FILE = "data/processed/validation_selected.csv"

MODEL_FILE = "models/lightgbm_model.pkl"
ENCODER_FILE = "models/label_encoder.pkl"

OUTPUT_DIR = "results/validation_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("Зареждане на validation dataset")
print("=" * 70)

df = pd.read_csv(VALIDATION_FILE)

X_validation = df.drop(columns=["label"])
y_validation = df["label"]

print("Зареждане на модела...")

model = joblib.load(MODEL_FILE)
label_encoder = joblib.load(ENCODER_FILE)

y_validation_encoded = label_encoder.transform(y_validation)

print("Prediction...")

y_pred = model.predict(X_validation)

accuracy = accuracy_score(
    y_validation_encoded,
    y_pred
)

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(f"Validation Accuracy: {accuracy:.4f}")

report = classification_report(
    y_validation_encoded,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

report_path = os.path.join(
    OUTPUT_DIR,
    "validation_classification_report.csv"
)

report_df.to_csv(report_path)

print(f"\nЗаписан report: {report_path}")

print("\nГенериране на confusion matrix...")

plt.figure(figsize=(14, 14))

ConfusionMatrixDisplay.from_predictions(
    y_validation_encoded,
    y_pred,
    display_labels=label_encoder.classes_,
    xticks_rotation=90,
    cmap="Blues",
    values_format="d"
)

plt.title("LightGBM - Validation Confusion Matrix")

plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "validation_confusion_matrix.png"
)

plt.savefig(cm_path, dpi=300)

plt.close()

print(f"Записана матрица: {cm_path}")

print("\nГотово.")