import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

TEST_FILE = "data/processed/test_selected.csv"
MODEL_FILE = "models/lightgbm_model.pkl"

OUTPUT_DIR = "results/shap_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Зареждане на test dataset...")
df = pd.read_csv(TEST_FILE)

X = df.drop(columns=["label"])

# SHAP е тежък, затова вземаме sample
X_sample = X.sample(n=3000, random_state=42)

print("Зареждане на LightGBM модела...")
model = joblib.load(MODEL_FILE)

print("Създаване на SHAP explainer...")
explainer = shap.TreeExplainer(model)

print("Изчисляване на SHAP стойности...")
shap_values = explainer.shap_values(X_sample)

print("Генериране на SHAP summary plot...")

plt.figure()
shap.summary_plot(
    shap_values,
    X_sample,
    show=False,
    max_display=20
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "lightgbm_shap_summary.png"
)

plt.savefig(summary_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Записана SHAP summary графика: {summary_path}")

print("Генериране на SHAP bar plot...")

plt.figure()
shap.summary_plot(
    shap_values,
    X_sample,
    plot_type="bar",
    show=False,
    max_display=20
)

bar_path = os.path.join(
    OUTPUT_DIR,
    "lightgbm_shap_bar.png"
)

plt.savefig(bar_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Записана SHAP bar графика: {bar_path}")

print("Готово.")