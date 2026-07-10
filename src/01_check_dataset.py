import pandas as pd

FILES = [
    "data/raw/CICIOT23/train/train.csv",
    "data/raw/CICIOT23/test/test.csv",
    "data/raw/CICIOT23/validation/validation.csv"
]

for file in FILES:
    print("\n" + "=" * 60)
    print(file)
    print("=" * 60)

    df = pd.read_csv(file)

    print("Размер:", df.shape)
    print("Колони:", df.shape[1])

    label_col = "label" if "label" in df.columns else "Label"

    print("\nКласове:")
    print(df[label_col].value_counts())