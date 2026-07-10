import os
import time
import joblib
import pandas as pd

TEST_FILE = "data/processed/test_selected.csv"
MODEL_FILE = "models/lightgbm_model.pkl"
ENCODER_FILE = "models/label_encoder.pkl"

OUTPUT_DIR = "results/stream_simulation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 1000

df = pd.read_csv(TEST_FILE)
model = joblib.load(MODEL_FILE)
encoder = joblib.load(ENCODER_FILE)

X = df.drop(columns=["label"])
y = df["label"]

results = []

start_total = time.time()

for start in range(0, len(X), BATCH_SIZE):
    end = start + BATCH_SIZE

    X_batch = X.iloc[start:end]
    y_batch = y.iloc[start:end]

    start_time = time.time()
    predictions = model.predict(X_batch)
    end_time = time.time()

    latency = end_time - start_time
    throughput = len(X_batch) / latency

    decoded_predictions = encoder.inverse_transform(predictions)

    attack_count = sum(decoded_predictions != "BenignTraffic")

    results.append({
        "batch_start": start,
        "batch_end": end,
        "records": len(X_batch),
        "latency_seconds": latency,
        "throughput_records_per_second": throughput,
        "detected_attacks": attack_count
    })

    print(f"Batch {start}-{end} | Latency: {latency:.4f}s | Throughput: {throughput:.2f} rec/s | Attacks: {attack_count}")

total_time = time.time() - start_total

results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "stream_simulation_results.csv"),
    index=False
)

summary = {
    "total_records": len(X),
    "total_time_seconds": total_time,
    "average_latency_seconds": results_df["latency_seconds"].mean(),
    "average_throughput_records_per_second": results_df["throughput_records_per_second"].mean()
}

pd.DataFrame([summary]).to_csv(
    os.path.join(OUTPUT_DIR, "stream_simulation_summary.csv"),
    index=False
)

print("\nStream simulation готова.")