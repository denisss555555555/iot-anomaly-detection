import os
import pandas as pd
import numpy as np

INPUT_FILES = {
    "train": "data/raw/CICIOT23/train/train.csv",
    "test": "data/raw/CICIOT23/test/test.csv",
    "validation": "data/raw/CICIOT23/validation/validation.csv"
}

OUTPUT_DIR = "data/processed_all_classes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_ROWS = {
    "train": 30000,
    "test": 8000,
    "validation": 8000
}


def prepare_file(split_name, input_path, output_path):
    print("=" * 70)
    print(f"Обработка: {input_path}")
    print("=" * 70)

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    label_col = "label"

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    sampled_parts = []

    for class_name in sorted(df[label_col].unique()):
        class_df = df[df[label_col] == class_name]
        sample_size = min(len(class_df), MAX_ROWS[split_name])

        sampled_parts.append(
            class_df.sample(n=sample_size, random_state=42)
        )

    final_df = pd.concat(sampled_parts, ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Краен размер:", final_df.shape)
    print(final_df[label_col].value_counts())

    final_df.to_csv(output_path, index=False)
    print(f"Записан файл: {output_path}")


def main():
    for split_name, input_path in INPUT_FILES.items():
        output_path = os.path.join(OUTPUT_DIR, f"{split_name}_all_classes.csv")
        prepare_file(split_name, input_path, output_path)


if __name__ == "__main__":
    main()