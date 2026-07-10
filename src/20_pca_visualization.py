import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

TEST_FILE = "data/processed/test_selected.csv"

OUTPUT_DIR = "results/pca_visualization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(TEST_FILE)

df_sample = df.sample(n=min(20000, len(df)), random_state=42)

X = df_sample.drop(columns=["label"])
y = df_sample["label"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y_encoded,
    s=5,
    alpha=0.7
)

plt.title("PCA визуализация на IoT мрежовия трафик")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(scatter)

plot_path = os.path.join(OUTPUT_DIR, "pca_2d_visualization.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

explained = pd.DataFrame({
    "component": ["PC1", "PC2"],
    "explained_variance_ratio": pca.explained_variance_ratio_
})

explained.to_csv(
    os.path.join(OUTPUT_DIR, "pca_explained_variance.csv"),
    index=False
)

print("PCA визуализацията е готова.")