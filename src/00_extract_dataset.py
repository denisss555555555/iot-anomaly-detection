import zipfile
import os

ZIP_PATH = "archive (1).zip"
OUTPUT_DIR = "data/raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(OUTPUT_DIR)

print("Dataset-ът е разархивиран успешно.")