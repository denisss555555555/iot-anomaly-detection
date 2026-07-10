import os
import matplotlib.pyplot as plt

OUTPUT_DIR = "results/preprocessing_pipeline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 12))

ax.axis("off")

steps = [
    "Raw CICIoT2023 Dataset",
    "Data Cleaning\n(Remove NaN / inf)",
    "Class Sampling\n(Balancing)",
    "Label Encoding",
    "Train / Test / Validation Split",
    "Feature Processing",
    "Model Training",
    "Evaluation & Visualization"
]

y_positions = list(range(len(steps), 0, -1))

for y, step in zip(y_positions, steps):

    ax.text(
        0.5,
        y,
        step,
        ha="center",
        va="center",
        fontsize=14,
        bbox=dict(
            boxstyle="round,pad=0.5",
            edgecolor="black",
            facecolor="#d9eaf7"
        )
    )

for i in range(len(y_positions) - 1):

    ax.annotate(
        "",
        xy=(0.5, y_positions[i + 1] + 0.3),
        xytext=(0.5, y_positions[i] - 0.3),
        arrowprops=dict(
            arrowstyle="->",
            lw=2
        )
    )

plt.title(
    "IoT IDS Data Preprocessing Pipeline",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "preprocessing_pipeline.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Pipeline diagram generated successfully.")