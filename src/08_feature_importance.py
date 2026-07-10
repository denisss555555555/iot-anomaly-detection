import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

TRAIN_FILE = "data/processed/train_selected.csv"
MODEL_FILE = "models/lightgbm_model.pkl"

OUTPUT_DIR = "results/feature_importance"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(TRAIN_FILE)

X = df.drop(columns=["label"])

model = joblib.load(MODEL_FILE)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

csv_path = os.path.join(OUTPUT_DIR, "lightgbm_feature_importance.csv")
importance_df.to_csv(csv_path, index=False)

top_features = importance_df.head(20)

plt.figure(figsize=(12, 8))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.gca().invert_yaxis()
plt.title("LightGBM - Top 20 важни характеристики")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, "lightgbm_top20_features.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print("Feature importance е готов.")
print(f"CSV: {csv_path}")
print(f"Графика: {plot_path}")

print("\nTop 20 features:")
print(top_features)