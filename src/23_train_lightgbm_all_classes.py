import os
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier

TRAIN_FILE = "data/processed_all_classes/train_all_classes.csv"
TEST_FILE = "data/processed_all_classes/test_all_classes.csv"

OUTPUT_DIR = "results/all_classes_experiment"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 70)
print("Зареждане на всички класове")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

LABEL_COLUMN = "label"

X_train = train_df.drop(columns=[LABEL_COLUMN])
y_train = train_df[LABEL_COLUMN]

X_test = test_df.drop(columns=[LABEL_COLUMN])
y_test = test_df[LABEL_COLUMN]

encoder = LabelEncoder()

y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

print("Брой класове:", len(encoder.classes_))
print("Класове:")
for class_name in encoder.classes_:
    print("-", class_name)

model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=64,
    objective="multiclass",
    random_state=42,
    n_jobs=-1
)

print("\nОбучение на LightGBM върху всички класове...")

model.fit(X_train, y_train_encoded)

print("\nPrediction...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test_encoded, y_pred)

report_text = classification_report(
    y_test_encoded,
    y_pred,
    target_names=encoder.classes_
)

report_dict = classification_report(
    y_test_encoded,
    y_pred,
    target_names=encoder.classes_,
    output_dict=True
)

report_df = pd.DataFrame(report_dict).transpose()

report_csv_path = os.path.join(
    OUTPUT_DIR,
    "lightgbm_all_classes_report.csv"
)

report_txt_path = os.path.join(
    OUTPUT_DIR,
    "lightgbm_all_classes_report.txt"
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "lightgbm_all_classes_summary.txt"
)

report_df.to_csv(report_csv_path)

with open(report_txt_path, "w", encoding="utf-8") as file:
    file.write(report_text)

with open(summary_path, "w", encoding="utf-8") as file:
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Number of classes: {len(encoder.classes_)}\n")

joblib.dump(model, "models/lightgbm_all_classes_model.pkl")
joblib.dump(encoder, "models/label_encoder_all_classes.pkl")

print("=" * 70)
print("ALL CLASSES RESULTS")
print("=" * 70)

print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n")
print(report_text)

print("\nЗаписани файлове:")
print(f"- {report_csv_path}")
print(f"- {report_txt_path}")
print(f"- {summary_path}")
print("- models/lightgbm_all_classes_model.pkl")
print("- models/label_encoder_all_classes.pkl")