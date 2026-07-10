import time
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

DATA_FILE = "data/processed/test_selected.csv"
MODEL_FILE = "models/lightgbm_model.pkl"
ENCODER_FILE = "models/label_encoder.pkl"

st.set_page_config(
    page_title="IoT IDS Dashboard",
    layout="wide"
)

st.title("IoT Anomaly Detection Dashboard")
st.write("Симулация на откриване на атаки в IoT мрежов трафик.")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    return model, encoder

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

model, encoder = load_model()
df = load_data()

batch_size = st.sidebar.slider(
    "Размер на batch",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100
)

start_index = st.sidebar.slider(
    "Начален ред",
    min_value=0,
    max_value=max(0, len(df) - batch_size),
    value=0,
    step=batch_size
)

batch = df.iloc[start_index:start_index + batch_size]

X_batch = batch.drop(columns=["label"])
y_true = batch["label"]

start_time = time.time()
predictions = model.predict(X_batch)
latency = time.time() - start_time

y_pred = encoder.inverse_transform(predictions)

result_df = pd.DataFrame({
    "True Label": y_true.values,
    "Predicted Label": y_pred
})

attack_count = sum(result_df["Predicted Label"] != "BenignTraffic")
normal_count = sum(result_df["Predicted Label"] == "BenignTraffic")
throughput = len(batch) / latency

col1, col2, col3, col4 = st.columns(4)

col1.metric("Обработени записи", len(batch))
col2.metric("Открити атаки", attack_count)
col3.metric("Нормален трафик", normal_count)
col4.metric("Throughput", f"{throughput:.2f} rec/s")

st.metric("Latency", f"{latency:.4f} sec")

st.subheader("Разпределение на предсказаните класове")

class_counts = result_df["Predicted Label"].value_counts()

fig, ax = plt.subplots(figsize=(12, 6))
class_counts.plot(kind="bar", ax=ax)
ax.set_xlabel("Клас")
ax.set_ylabel("Брой")
ax.set_title("Предсказани класове")
plt.xticks(rotation=90)
st.pyplot(fig)

st.subheader("Последни предсказания")
st.dataframe(result_df.head(100))