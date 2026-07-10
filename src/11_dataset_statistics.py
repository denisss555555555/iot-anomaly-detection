import os
import pandas as pd
import matplotlib.pyplot as plt

DATASET_FILE = "data/processed/train_selected.csv"

OUTPUT_DIR = "results/dataset_statistics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("DATASET STATISTICS")
print("=" * 70)

df = pd.read_csv(DATASET_FILE)

print("Dataset shape:", df.shape)

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

label_counts = df["label"].value_counts()

print("\nClass distribution:")
print(label_counts)

label_counts.to_csv(
    os.path.join(OUTPUT_DIR, "class_distribution.csv")
)

# ============================================================
# CLASS DISTRIBUTION FIGURE
# ============================================================

plt.figure(figsize=(18, 8))

label_counts.plot(kind="bar")

plt.title(
    "Class Distribution in CICIoT2023 Dataset",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel("Classes", fontsize=14)
plt.ylabel("Number of Samples", fontsize=14)

plt.xticks(rotation=90)

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "class_distribution.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FEATURE STATISTICS
# ============================================================

stats_df = df.describe()

stats_df.to_csv(
    os.path.join(OUTPUT_DIR, "feature_statistics.csv")
)

print("\nFeature statistics:")
print(stats_df)

print("\nГотово.")