import os
import pandas as pd
import numpy as np

# ============================================================
# INPUT FILES
# ============================================================

RAW_DIR = "data/raw/CICIOT23"

TRAIN_FILE = os.path.join(RAW_DIR, "train", "train.csv")
TEST_FILE = os.path.join(RAW_DIR, "test", "test.csv")
VALIDATION_FILE = os.path.join(RAW_DIR, "validation", "validation.csv")

# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILES = {
    "train": os.path.join(OUTPUT_DIR, "train_selected.csv"),
    "test": os.path.join(OUTPUT_DIR, "test_selected.csv"),
    "validation": os.path.join(OUTPUT_DIR, "validation_selected.csv")
}

# ============================================================
# SETTINGS
# ============================================================

LABEL_COLUMN = "label"

MAX_ROWS_PER_CLASS = {
    "train": 30000,
    "test": 8000,
    "validation": 8000
}

RANDOM_STATE = 42

# ============================================================
# FUNCTIONS
# ============================================================

def prepare_dataset(input_file, output_file, split_name):
    print("=" * 70)
    print(f"Обработка на {split_name} dataset")
    print("=" * 70)

    df = pd.read_csv(input_file)

    print("Първоначален размер:", df.shape)

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Колоната '{LABEL_COLUMN}' не е намерена.")

    # Премахване на невалидни стойности
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    print("Размер след почистване:", df.shape)

    classes = sorted(df[LABEL_COLUMN].unique())

    print(f"Открити класове: {len(classes)}")
    for cls in classes:
        print("-", cls)

    sampled_parts = []

    max_rows = MAX_ROWS_PER_CLASS[split_name]

    for class_name in classes:
        class_df = df[df[LABEL_COLUMN] == class_name]

        sample_size = min(len(class_df), max_rows)

        sampled_class_df = class_df.sample(
            n=sample_size,
            random_state=RANDOM_STATE
        )

        sampled_parts.append(sampled_class_df)

        print(f"{class_name}: {sample_size} записа")

    final_df = pd.concat(sampled_parts, axis=0)

    final_df = final_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    final_df.to_csv(output_file, index=False)

    print("Финален размер:", final_df.shape)
    print("Записан файл:", output_file)
    print()


# ============================================================
# MAIN
# ============================================================

prepare_dataset(TRAIN_FILE, OUTPUT_FILES["train"], "train")
prepare_dataset(TEST_FILE, OUTPUT_FILES["test"], "test")
prepare_dataset(VALIDATION_FILE, OUTPUT_FILES["validation"], "validation")

print("=" * 70)
print("Готово. Данните са подготвени с ВСИЧКИ класове.")
print("=" * 70)