import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay
)

TEST_FILE = "data/processed/test_selected.csv"
MODEL_FILE = "models/random_forest_model.pkl"
ENCODER_FILE = "models/label_encoder.pkl"

FIGURES_DIR = "results/figures"
METRICS_DIR = "results/metrics"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

print("Зареждане на test dataset...")

test_df = pd.read_csv(TEST_FILE)

LABEL_COLUMN = "label"

X_test = test_df.drop(columns=[LABEL_COLUMN])
y_test = test_df[LABEL_COLUMN]

print("Зареждане на модела...")

model = joblib.load(MODEL_FILE)
label_encoder = joblib.load(ENCODER_FILE)

y_test_encoded = label_encoder.transform(y_test)
y_pred = model.predict(X_test)

print("Генериране на classification report...")

report_dict = classification_report(
    y_test_encoded,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True
)

report_df = pd.DataFrame(report_dict).transpose()

report_path = os.path.join(METRICS_DIR, "random_forest_classification_report.csv")
report_df.to_csv(report_path)

print(f"Записан report: {report_path}")

print("Генериране на bar chart за F1-score...")

class_report_df = report_df.iloc[:len(label_encoder.classes_)]

plt.figure(figsize=(14, 7))
plt.bar(class_report_df.index, class_report_df["f1-score"])
plt.xticks(rotation=90)
plt.ylim(0, 1.05)
plt.title("Random Forest - F1-score по класове")
plt.xlabel("Клас")
plt.ylabel("F1-score")
plt.tight_layout()

f1_path = os.path.join(FIGURES_DIR, "random_forest_f1_scores.png")
plt.savefig(f1_path, dpi=300)
plt.close()

print(f"Записана графика: {f1_path}")

print("Генериране на confusion matrix...")

plt.figure(figsize=(14, 14))
ConfusionMatrixDisplay.from_predictions(
    y_test_encoded,
    y_pred,
    display_labels=label_encoder.classes_,
    xticks_rotation=90,
    cmap="Blues",
    values_format="d"
)

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

cm_path = os.path.join(FIGURES_DIR, "random_forest_confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()

print(f"Записана матрица: {cm_path}")

print("\nГотово.")