import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder

TEST_FILE = "data/processed/test_selected.csv"

MODEL_FILE = "models/binary_lightgbm_model.pkl"

OUTPUT_DIR = "results/roc_auc"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(TEST_FILE)

df["binary_label"] = df["label"].apply(
    lambda x: "Normal" if x == "BenignTraffic" else "Attack"
)

X_test = df.drop(columns=["label", "binary_label"])
y_test = df["binary_label"]

encoder = LabelEncoder()

y_test_encoded = encoder.fit_transform(y_test)

model = joblib.load(MODEL_FILE)

y_probs = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test_encoded, y_probs)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Binary Classification")

plt.legend()

plt.grid(True)

plot_path = os.path.join(
    OUTPUT_DIR,
    "roc_auc_curve.png"
)

plt.savefig(plot_path, dpi=300)

plt.close()

print(f"ROC-AUC: {roc_auc:.4f}")