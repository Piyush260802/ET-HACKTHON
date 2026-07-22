import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


MODEL_PATH = "models/helmet_rf_model_v2.joblib"

TEST_DATA_DIR = "data/test2"

TEST_FILES = [
    ("test2_on_head_01.csv", "On Head"),
    ("test2_on_belt_01.csv", "On Belt"),
    ("test2_in_hand_01.csv", "In Hand"),
    ("test2_on_surface_01.csv", "On Surface"),
]

PREDICTIONS_PATH = "data/processed/v2_test2_predictions.csv"
CONFUSION_MATRIX_PATH = "data/processed/v2_test2_confusion_matrix.png"
REPORT_PATH = "data/processed/v2_test2_report.txt"

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


def rms(values):
    return np.sqrt(np.mean(np.square(values)))


def clean_test_file(df, expected_label, file_name):
    expected_columns = [
        "time_ms",
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "temp_c",
        "label",
    ]

    missing_columns = [
        column for column in expected_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{file_name} dosyasinda eksik sutunlar var: {missing_columns}"
        )

    df = df[expected_columns].copy()

    # Dosya icinde tekrar baslik satiri varsa temizler
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

    df["label"] = df["label"].astype(str).str.strip()

    df = df.dropna(subset=numeric_columns + ["label"]).copy()

    unique_labels = df["label"].unique().tolist()

    if unique_labels != [expected_label]:
        raise ValueError(
            f"{file_name} label hatasi. "
            f"Beklenen: {expected_label}, bulunan: {unique_labels}"
        )

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


def create_test_windows(df, expected_label, file_name, window_size, step_size):
    feature_rows = []
    metadata_rows = []

    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        window = df.iloc[start:end]

        features = extract_window_features(window)
        feature_rows.append(features)

        metadata_rows.append(
            {
                "source_file": file_name,
                "start_index": start,
                "end_index": end - 1,
                "start_time_ms": window["time_ms"].iloc[0],
                "end_time_ms": window["time_ms"].iloc[-1],
                "real_label": expected_label,
            }
        )

    X = pd.DataFrame(feature_rows)
    metadata = pd.DataFrame(metadata_rows)

    return X, metadata


def save_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("V2 Model - Test2 Bagimsiz Confusion Matrix")
    plt.xlabel("Tahmin")
    plt.ylabel("Gercek")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()


def main():
    os.makedirs("data/processed", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model bulunamadi: {MODEL_PATH}")

    saved_model = joblib.load(MODEL_PATH)

    model = saved_model["model"]
    feature_columns = saved_model["feature_columns"]
    labels = saved_model["labels"]
    window_size = saved_model["window_size"]
    step_size = saved_model["step_size"]

    print("======================================")
    print("V2 MODELI - TEST2 BAGIMSIZ DEGERLENDIRMESI")
    print("======================================")
    print(f"Model: {MODEL_PATH}")
    print(f"Window size: {window_size}")
    print(f"Step size: {step_size}")

    all_features = []
    all_metadata = []

    for file_name, expected_label in TEST_FILES:
        file_path = os.path.join(TEST_DATA_DIR, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Test dosyasi bulunamadi: {file_path}")

        df = pd.read_csv(file_path)
        df = clean_test_file(df, expected_label, file_name)

        X_file, metadata_file = create_test_windows(
            df,
            expected_label,
            file_name,
            window_size,
            step_size,
        )

        print()
        print(f"Dosya: {file_name}")
        print(f"Label: {expected_label}")
        print(f"Ham satir sayisi: {len(df)}")
        print(f"Test pencere sayisi: {len(X_file)}")

        all_features.append(X_file)
        all_metadata.append(metadata_file)

    X_test = pd.concat(all_features, ignore_index=True)
    results = pd.concat(all_metadata, ignore_index=True)

    X_test = X_test[feature_columns]

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    results["predicted_label"] = predictions
    results["confidence"] = np.max(probabilities, axis=1)
    results["correct"] = results["real_label"] == results["predicted_label"]

    y_true = results["real_label"]
    y_pred = results["predicted_label"]

    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    results.to_csv(PREDICTIONS_PATH, index=False)
    save_confusion_matrix(cm, labels)

    class_results = results.groupby("real_label")["correct"].agg(["sum", "count"])
    class_results["accuracy"] = class_results["sum"] / class_results["count"]

    print()
    print("======================================")
    print("V2 TEST2 BAGIMSIZ SONUCLARI")
    print("======================================")
    print(f"Toplam test pencere sayisi: {len(results)}")
    print(f"Bagimsiz test accuracy: {accuracy:.4f}")
    print(f"Bagimsiz test accuracy yuzde: %{accuracy * 100:.2f}")

    print()
    print("Classification Report:")
    print(report)

    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    print()
    print("Sinif bazli dogruluk:")
    print(class_results)

    print()
    print("Ortalama guven degerleri:")
    print(results.groupby("real_label")["confidence"].mean())

    report_text = ""
    report_text += "AKILLI BARET - BAGIMSIZ TEST DEGERLENDIRMESI\n"
    report_text += "============================================\n\n"
    report_text += f"Toplam test pencere sayisi: {len(results)}\n"
    report_text += f"Bagimsiz test accuracy: {accuracy:.4f}\n"
    report_text += f"Bagimsiz test accuracy yuzde: %{accuracy * 100:.2f}\n\n"
    report_text += "Classification Report:\n"
    report_text += report
    report_text += "\nConfusion Matrix:\n"
    report_text += pd.DataFrame(cm, index=labels, columns=labels).to_string()
    report_text += "\n\nSinif bazli dogruluk:\n"
    report_text += class_results.to_string()
    report_text += "\n\nOrtalama guven degerleri:\n"
    report_text += results.groupby("real_label")["confidence"].mean().to_string()

    with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
        report_file.write(report_text)

    print()
    print(f"Tahminler kaydedildi: {PREDICTIONS_PATH}")
    print(f"Confusion matrix kaydedildi: {CONFUSION_MATRIX_PATH}")
    print(f"Rapor kaydedildi: {REPORT_PATH}")


if __name__ == "__main__":
    main()