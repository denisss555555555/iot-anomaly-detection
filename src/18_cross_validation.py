import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

TRAIN_FILE = "data/processed/train_selected.csv"

OUTPUT_DIR = "results/cross_validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(TRAIN_FILE)

X = df.drop(columns=["label"])
y = df["label"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

model = LGBMClassifier(
    n_estimators=150,
    learning_rate=0.05,
    num_leaves=64,
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y_encoded,
    cv=cv,
    scoring="f1_weighted",
    n_jobs=-1
)

results = pd.DataFrame({
    "fold": list(range(1, 6)),
    "weighted_f1": scores
})

results.loc[len(results)] = ["mean", np.mean(scores)]
results.loc[len(results)] = ["std", np.std(scores)]

results.to_csv(
    os.path.join(OUTPUT_DIR, "lightgbm_cross_validation.csv"),
    index=False
)

print(results)