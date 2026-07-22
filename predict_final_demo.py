import os
import sys
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================================================
# FINAL DEMO AYARLARI
# ======================================================

MODEL_PATH = "models/helmet_rf_model_v3_final.joblib"
OUTPUT_DIR = "data/demo_output"

SENSOR_COLUMNS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acc_mag",
    "gyro_mag",
]

LABEL_ORDER = [
    "On Head",
    "On Belt",
    "In Hand",
    "On Surface",
]


def rms(values):
    return np.sqrt(np.mean(np.square(values)))


def clean_sensor_data(df):
    required_columns = [
        "time_ms",
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "temp_c",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"CSV dosyasinda eksik sutunlar var: {missing_columns}"
        )

    # Dosyada tekrar yazilmis baslik satiri varsa temizle
    df = df[df["time_ms"].astype(str) != "time_ms"].copy()

    numeric_columns = [
        "time_ms",
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "temp_c",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=numeric_columns).copy()

    # Model tahmin yaparken label sutununu kullanmaz.
    if "label" in df.columns:
        print("Not: CSV icinde label var, ancak AI tahmininde kullanilmiyor.")

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

    return df


def extract_window_features(window):
    features = {}

    for column in SENSOR_COLUMNS:
        values = window[column].values

        features[f"{column}_mean"] = np.mean(values)
        features[f"{column}_std"] = np.std(values)
        features[f"{column}_min"] = np.min(values)
        features[f"{column}_max"] = np.max(values)
        features[f"{column}_range"] = np.max(values) - np.min(values)
        features[f"{column}_rms"] = rms(values)

    return features


def create_prediction_windows(df, window_size, step_size):
    feature_rows = []
    metadata_rows = []

    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        window = df.iloc[start:end]

        features = extract_window_features(window)

        metadata = {
            "start_index": start,
            "end_index": end - 1,
            "start_time_ms": int(window["time_ms"].iloc[0]),
            "end_time_ms": int(window["time_ms"].iloc[-1]),
        }

        feature_rows.append(features)
        metadata_rows.append(metadata)

    X = pd.DataFrame(feature_rows)
    metadata_df = pd.DataFrame(metadata_rows)

    return X, metadata_df


def save_prediction_timeline(results, output_path):
    label_to_number = {
        "On Head": 1,
        "On Belt": 2,
        "In Hand": 3,
        "On Surface": 4,
    }

    time_seconds = (
        results["start_time_ms"] - results["start_time_ms"].iloc[0]
    ) / 1000.0

    y_values = results["predicted_label"].map(label_to_number)

    plt.figure(figsize=(12, 5))
    plt.plot(time_seconds, y_values, marker="o", markersize=3)
    plt.yticks(
        [1, 2, 3, 4],
        ["On Head", "On Belt", "In Hand", "On Surface"]
    )
    plt.xlabel("Zaman (saniye)")
    plt.ylabel("AI Tahmini")
    plt.title("Akilli Baret - V3 Final Model Tahmin Zaman Cizelgesi")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Kullanim:")
        print("python ml/predict_final_demo.py data\\test2\\test2_on_belt_01.csv")
        return

    input_path = sys.argv[1]

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Girdi CSV dosyasi bulunamadi: {input_path}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Final model bulunamadi: {MODEL_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    file_stem = os.path.splitext(os.path.basename(input_path))[0]

    predictions_path = os.path.join(
        OUTPUT_DIR,
        f"{file_stem}_predictions.csv"
    )

    timeline_path = os.path.join(
        OUTPUT_DIR,
        f"{file_stem}_timeline.png"
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        f"{file_stem}_summary.txt"
    )

    saved_model = joblib.load(MODEL_PATH)

    model = saved_model["model"]
    feature_columns = saved_model["feature_columns"]
    window_size = saved_model["window_size"]
    step_size = saved_model["step_size"]

    print("======================================")
    print("AKILLI BARET - V3 FINAL DEMO TAHMINI")
    print("======================================")
    print(f"Model: {MODEL_PATH}")
    print(f"Girdi dosyasi: {input_path}")
    print(f"Window size: {window_size}")
    print(f"Step size: {step_size}")

    df = pd.read_csv(input_path)
    print(f"\nHam veri satir sayisi: {len(df)}")

    df = clean_sensor_data(df)
    print(f"Temiz veri satir sayisi: {len(df)}")

    X, results = create_prediction_windows(
        df,
        window_size,
        step_size
    )

    if len(X) == 0:
        raise ValueError(
            "Tahmin icin yeterli veri yok. "
            "En az 40 sensor satiri gerekli."
        )

    X = X[feature_columns]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    results["predicted_label"] = predictions
    results["confidence"] = np.max(probabilities, axis=1)

    for index, class_name in enumerate(model.classes_):
        results[f"prob_{class_name.replace(' ', '_').lower()}"] = probabilities[:, index]

    prediction_counts = results["predicted_label"].value_counts()
    prediction_percentages = (
        results["predicted_label"].value_counts(normalize=True) * 100
    )

    dominant_label = prediction_counts.index[0]
    dominant_percentage = prediction_percentages.iloc[0]

    last_window_count = min(5, len(results))
    last_windows = results.tail(last_window_count)

    stable_last_label = last_windows["predicted_label"].mode().iloc[0]
    stable_last_confidence = last_windows["confidence"].mean() * 100

    average_confidence = results["confidence"].mean() * 100

    results.to_csv(predictions_path, index=False)
    save_prediction_timeline(results, timeline_path)

    print("\n======================================")
    print("DEMO TAHMIN SONUCU")
    print("======================================")
    print(f"Toplam tahmin penceresi: {len(results)}")
    print(f"AI ana karari: {dominant_label}")
    print(f"Ana karar pencere orani: %{dominant_percentage:.2f}")
    print(f"Son durum tahmini: {stable_last_label}")
    print(f"Son durum ortalama guveni: %{stable_last_confidence:.2f}")
    print(f"Genel ortalama guven: %{average_confidence:.2f}")

    print("\nTahmin dagilimi:")
    for label in LABEL_ORDER:
        count = int(prediction_counts.get(label, 0))
        percentage = float(prediction_percentages.get(label, 0.0))

        print(f"{label:12} : {count:4} pencere  (%{percentage:.2f})")

    summary_text = ""
    summary_text += "AKILLI BARET - V3 FINAL DEMO TAHMIN OZETI\n"
    summary_text += "=========================================\n\n"
    summary_text += f"Model: {MODEL_PATH}\n"
    summary_text += f"Girdi dosyasi: {input_path}\n"
    summary_text += f"Toplam tahmin penceresi: {len(results)}\n\n"
    summary_text += f"AI ana karari: {dominant_label}\n"
    summary_text += f"Ana karar pencere orani: %{dominant_percentage:.2f}\n"
    summary_text += f"Son durum tahmini: {stable_last_label}\n"
    summary_text += f"Son durum ortalama guveni: %{stable_last_confidence:.2f}\n"
    summary_text += f"Genel ortalama guven: %{average_confidence:.2f}\n\n"
    summary_text += "Tahmin dagilimi:\n"

    for label in LABEL_ORDER:
        count = int(prediction_counts.get(label, 0))
        percentage = float(prediction_percentages.get(label, 0.0))

        summary_text += f"{label:12} : {count:4} pencere  (%{percentage:.2f})\n"

    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(summary_text)

    print("\nCiktilar:")
    print(f"Tahmin CSV: {predictions_path}")
    print(f"Tahmin grafigi: {timeline_path}")
    print(f"Ozet rapor: {summary_path}")


if __name__ == "__main__":
    main()