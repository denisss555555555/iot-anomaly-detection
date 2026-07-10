import os
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "results/main_vs_all_classes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAIN_ACCURACY = 0.9763
MAIN_MACRO_F1 = 0.97
MAIN_WEIGHTED_F1 = 0.98

ALL_REPORT_FILE = "results/all_classes_experiment/lightgbm_all_classes_report.csv"

all_report = pd.read_csv(ALL_REPORT_FILE, index_col=0)

all_accuracy = float(all_report.loc["accuracy", "f1-score"])
all_macro_f1 = float(all_report.loc["macro avg", "f1-score"])
all_weighted_f1 = float(all_report.loc["weighted avg", "f1-score"])

comparison_df = pd.DataFrame({
    "Experiment": [
        "15 атаки + BenignTraffic",
        "Всички класове"
    ],
    "Accuracy": [
        MAIN_ACCURACY,
        all_accuracy
    ],
    "Macro F1": [
        MAIN_MACRO_F1,
        all_macro_f1
    ],
    "Weighted F1": [
        MAIN_WEIGHTED_F1,
        all_weighted_f1
    ]
})

csv_path = os.path.join(OUTPUT_DIR, "main_vs_all_classes.csv")
comparison_df.to_csv(csv_path, index=False)

print(comparison_df)

x = range(len(comparison_df))

plt.figure(figsize=(10, 6))

plt.plot(x, comparison_df["Accuracy"], marker="o", label="Accuracy")
plt.plot(x, comparison_df["Macro F1"], marker="o", label="Macro F1")
plt.plot(x, comparison_df["Weighted F1"], marker="o", label="Weighted F1")

plt.xticks(x, comparison_df["Experiment"])
plt.ylim(0.80, 1.00)

plt.title("Сравнение: основен експеримент срещу всички класове")
plt.xlabel("Експеримент")
plt.ylabel("Стойност")

plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, "main_vs_all_classes_plot.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Записана таблица: {csv_path}")
print(f"Записана графика: {plot_path}")