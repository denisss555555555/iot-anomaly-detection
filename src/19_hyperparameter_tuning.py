import os
import joblib
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

TRAIN_FILE = "data/processed/train_selected.csv"

OUTPUT_DIR = "results/hyperparameter_tuning"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(TRAIN_FILE)

# За да не е прекалено тежко
df_sample = df.sample(n=min(120000, len(df)), random_state=42)

X = df_sample.drop(columns=["label"])
y = df_sample["label"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

model = LGBMClassifier(
    objective="multiclass",
    random_state=42,
    n_jobs=-1
)

param_grid = {
    "n_estimators": [100, 150, 200],
    "learning_rate": [0.03, 0.05, 0.1],
    "num_leaves": [31, 64, 128],
    "max_depth": [-1, 10, 20],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=10,
    scoring="f1_weighted",
    cv=3,
    random_state=42,
    verbose=2,
    n_jobs=-1
)

search.fit(X, y_encoded)

results_df = pd.DataFrame(search.cv_results_)
results_df.to_csv(
    os.path.join(OUTPUT_DIR, "lightgbm_tuning_results.csv"),
    index=False
)

joblib.dump(
    search.best_estimator_,
    "models/lightgbm_tuned_model.pkl"
)

with open(os.path.join(OUTPUT_DIR, "best_params.txt"), "w", encoding="utf-8") as f:
    f.write(str(search.best_params_))
    f.write(f"\nBest score: {search.best_score_}")

print("Най-добри параметри:")
print(search.best_params_)
print("Best score:", search.best_score_)