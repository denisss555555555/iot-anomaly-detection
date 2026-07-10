import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results/model_comparison"

os.makedirs(RESULTS_DIR, exist_ok=True)

models = ["Random Forest", "XGBoost", "LightGBM"]

accuracy = [0.9735, 0.9748, 0.9763]
macro_f1 = [0.96, 0.97, 0.97]
weighted_f1 = [0.97, 0.97, 0.98]

comparison_df = pd.DataFrame({
    "Model": models,
    "Accuracy": accuracy,
    "Macro_F1": macro_f1,
    "Weighted_F1": weighted_f1
})

csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
comparison_df.to_csv(csv_path, index=False)

print(comparison_df)

plt.figure(figsize=(10, 6))

x = range(len(models))

plt.plot(x, accuracy, marker="o", label="Accuracy")
plt.plot(x, macro_f1, marker="o", label="Macro F1")
plt.plot(x, weighted_f1, marker="o", label="Weighted F1")

plt.xticks(x, models)

plt.ylim(0.94, 1.00)

plt.title("Сравнение на ML моделите")
plt.xlabel("Модел")
plt.ylabel("Стойност")
plt.legend()

plt.grid(True)

plot_path = os.path.join(
    RESULTS_DIR,
    "model_comparison_plot.png"
)

plt.savefig(plot_path, dpi=300)
plt.close()

print(f"\nЗаписана таблица: {csv_path}")
print(f"Записана графика: {plot_path}")