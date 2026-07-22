import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "data/raw/helmet_data.csv"
MODEL_PATH = "models/helmet_real_rf_model.joblib"
OUTPUT_PATH = "data/processed/real_predictions.csv"
TIMELINE_PATH = "data/processed/real_prediction_timeline.png"


SENSOR_COLUMNS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acc_mag",
    "gyro_mag"
]


def rms(values):
    return np.sqrt(np.mean(np.square(values)))


def clean_data(df):
    expected_columns = [
        "time_ms",
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "temp_c"
    ]

    missing_columns = [col for col in expected_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Eksik sütunlar var: {missing_columns}")

    # Dosya içinde tekrar başlık satırı varsa temizler
    df = df[df["time_ms"].astype(str) != "time_ms"].copy()

    numeric_columns = [
        "time_ms",
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "temp_c"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_columns).copy()

    df["acc_mag"] = np.sqrt(
        df["acc_x"] ** 2 +
        df["acc_y"] ** 2 +
        df["acc_z"] ** 2
    )

    df["gyro_mag"] = np.sqrt(
        df["gyro_x"] ** 2 +
        df["gyro_y"] ** 2 +
        df["gyro_z"] ** 2
    )

    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip()

    return df


def extract_window_features(window):
    features = {}

    for col in SENSOR_COLUMNS:
        values = window[col].values

        features[f"{col}_mean"] = np.mean(values)
        features[f"{col}_std"] = np.std(values)
        features[f"{col}_min"] = np.min(values)
        features[f"{col}_max"] = np.max(values)
        features[f"{col}_range"] = np.max(values) - np.min(values)
        features[f"{col}_rms"] = rms(values)

    return features


def create_prediction_windows(df, window_size, step_size):
    rows = []

    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        window = df.iloc[start:end]

        features = extract_window_features(window)

        row = {
            "start_index": start,
            "end_index": end - 1,
            "start_time_ms": window["time_ms"].iloc[0],
            "end_time_ms": window["time_ms"].iloc[-1],
        }

        # Elimizde gerçek label varsa sadece kontrol amaçlı ekliyoruz.
        # Gerçek kullanımda label olmayabilir.
        if "label" in window.columns:
            row["real_label_for_check"] = window["label"].value_counts().index[0]

        row.update(features)
        rows.append(row)

    return pd.DataFrame(rows)


def plot_prediction_timeline(predictions_df):
    label_to_number = {
        "On Head": 1,
        "On Belt": 2,
        "In Hand": 3,
        "On Surface": 4
    }

    y_values = predictions_df["predicted_label"].map(label_to_number)

    plt.figure(figsize=(12, 5))
    plt.plot(predictions_df["start_time_ms"], y_values, marker="o")
    plt.yticks(
        [1, 2, 3, 4],
        ["On Head", "On Belt", "In Hand", "On Surface"]
    )
    plt.xlabel("Zaman (ms)")
    plt.ylabel("Tahmin Edilen Durum")
    plt.title("Akıllı Baret Gerçek Veri Tahmin Zaman Çizelgesi")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(TIMELINE_PATH)
    plt.close()

    print(f"Tahmin zaman grafiği kaydedildi: {TIMELINE_PATH}")


def main():
    os.makedirs("data/processed", exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CSV bulunamadı: {DATA_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")

    saved_data = joblib.load(MODEL_PATH)

    model = saved_data["model"]
    feature_columns = saved_data["feature_columns"]
    window_size = saved_data["window_size"]
    step_size = saved_data["step_size"]

    print("Model yüklendi.")
    print("Window size:", window_size)
    print("Step size:", step_size)

    df = pd.read_csv(DATA_PATH)
    print("Okunan veri boyutu:", df.shape)

    df = clean_data(df)
    print("Temiz veri boyutu:", df.shape)

    prediction_windows = create_prediction_windows(df, window_size, step_size)

    print("Tahmin yapılacak pencere sayısı:", len(prediction_windows))

    X = prediction_windows[feature_columns]

    predicted_labels = model.predict(X)
    predicted_probs = model.predict_proba(X)

    confidence_values = np.max(predicted_probs, axis=1)

    result_columns = [
        "start_index",
        "end_index",
        "start_time_ms",
        "end_time_ms"
    ]

    if "real_label_for_check" in prediction_windows.columns:
        result_columns.append("real_label_for_check")

    results = prediction_windows[result_columns].copy()
    results["predicted_label"] = predicted_labels
    results["confidence"] = confidence_values

    results.to_csv(OUTPUT_PATH, index=False)

    print(f"\nTahminler kaydedildi: {OUTPUT_PATH}")

    print("\nTahmin dağılımı:")
    print(results["predicted_label"].value_counts())

    print("\nSon 10 tahmin:")
    print(results.tail(10))

    if "real_label_for_check" in results.columns:
        correct = results["real_label_for_check"] == results["predicted_label"]
        accuracy = correct.mean()

        print("\nKontrol amaçlı doğruluk:")
        print(f"{accuracy:.4f}")

        print("\nGerçek label - Tahmin karşılaştırması:")
        print(pd.crosstab(
            results["real_label_for_check"],
            results["predicted_label"],
            rownames=["Gerçek"],
            colnames=["Tahmin"]
        ))

    plot_prediction_timeline(results)


if __name__ == "__main__":
    main()