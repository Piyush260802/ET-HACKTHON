import os
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Grafik pencere açmadan dosyaya kaydetmek için
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# ======================================================
# V3 FINAL MODEL AYARLARI
# ======================================================

MODEL_PATH = "models/helmet_rf_model_v3_final.joblib"
CONFUSION_MATRIX_PATH = "data/processed/v3_final_internal_confusion_matrix.png"
REPORT_PATH = "data/processed/v3_final_internal_report.txt"

WINDOW_SIZE = 40   # 20 Hz veri icin yaklasik 2 saniye
STEP_SIZE = 20     # 1 saniye kaydirma

TRAINING_FILES = [
    # Ilk 10 dakikalik gercek egitim verileri
    ("data/raw/on_head_01.csv", "On Head"),
    ("data/raw/on_belt_01.csv", "On Belt"),
    ("data/raw/in_hand_01.csv", "In Hand"),
    ("data/raw/on_surface_01.csv", "On Surface"),

    # Ikinci 2 dakikalik gercek veriler
    ("data/training_extra/on_head_02.csv", "On Head"),
    ("data/training_extra/on_belt_02.csv", "On Belt"),
    ("data/training_extra/in_hand_02.csv", "In Hand"),
    ("data/training_extra/on_surface_02.csv", "On Surface"),

    # Yeni 5 dakikalik veriler
    # On Belt kaydinda baret sol tarafta tasindi.
    ("data/test2/test2_on_head_01.csv", "On Head"),
    ("data/test2/test2_on_belt_01.csv", "On Belt"),
    ("data/test2/test2_in_hand_01.csv", "In Hand"),
    ("data/test2/test2_on_surface_01.csv", "On Surface"),
]

EXPECTED_LABELS = [
    "On Head",
    "On Belt",
    "In Hand",
    "On Surface",
]

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


def clean_file(df, expected_label, file_path):
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
            f"{file_path} dosyasinda eksik sutunlar var: {missing_columns}"
        )

    df = df[expected_columns].copy()

    # Dosya icinde tekrar baslik satiri varsa temizle
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
            f"{file_path} label hatasi. "
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


def create_windows_from_single_file(df, label, source_file):
    feature_rows = []
    label_rows = []
    source_rows = []

    for start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        window = df.iloc[start:end]

        features = extract_window_features(window)

        feature_rows.append(features)
        label_rows.append(label)
        source_rows.append(source_file)

    X = pd.DataFrame(feature_rows)
    y = pd.Series(label_rows, name="label")
    sources = pd.Series(source_rows, name="source_file")

    return X, y, sources


def save_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("V3 Final Model - Ic Kontrol Confusion Matrix")
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
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    all_features = []
    all_labels = []
    all_sources = []

    total_raw_rows = 0

    print("======================================")
    print("AKILLI BARET - V3 FINAL MODEL EGITIMI")
    print("======================================")
    print()
    print("Bu model tum mevcut gercek verilerle egitilecek")
    print("Sag ve sol kemer tasima varyasyonlari modele dahil edildi.")
    print()

    for file_path, expected_label in TRAINING_FILES:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dosya bulunamadi: {file_path}")

        df = pd.read_csv(file_path)
        df = clean_file(df, expected_label, file_path)

        X_file, y_file, source_file = create_windows_from_single_file(
            df,
            expected_label,
            file_path
        )

        total_raw_rows += len(df)

        print(f"Dosya: {file_path}")
        print(f"Label: {expected_label}")
        print(f"Ham satir: {len(df)}")
        print(f"Pencere: {len(X_file)}")
        print("--------------------------------------")

        all_features.append(X_file)
        all_labels.append(y_file)
        all_sources.append(source_file)

    X = pd.concat(all_features, ignore_index=True)
    y = pd.concat(all_labels, ignore_index=True)
    sources = pd.concat(all_sources, ignore_index=True)

    print()
    print("======================================")
    print("V3 FINAL EGITIM VERISI OZETI")
    print("======================================")
    print(f"Toplam ham satir sayisi: {total_raw_rows}")
    print(f"Toplam pencere sayisi: {len(X)}")
    print(f"Feature sayisi: {X.shape[1]}")

    print()
    print("Pencere label dagilimi:")
    print(y.value_counts())

    # ======================================================
    # IC KONTROL
    # Bu sonuc final bagimsiz basari degildir.
    # Birbirine yakin pencereler ayni kayit oturumundan gelebilir.
    # ======================================================

    X_train, X_check, y_train, y_check = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    check_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=1
    )

    check_model.fit(X_train, y_train)

    y_pred = check_model.predict(X_check)

    accuracy = accuracy_score(y_check, y_pred)

    labels = EXPECTED_LABELS

    report = classification_report(
        y_check,
        y_pred,
        labels=labels,
        zero_division=0
    )

    cm = confusion_matrix(
        y_check,
        y_pred,
        labels=labels
    )

    print()
    print("======================================")
    print("V3 FINAL IC KONTROL SONUCLARI")
    print("======================================")
    print("Not: Bu final bagimsiz test sonucu degildir.")
    print(f"Ic kontrol accuracy: {accuracy:.4f}")
    print(f"Ic kontrol accuracy yuzde: %{accuracy * 100:.2f}")

    print()
    print("Classification Report:")
    print(report)

    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    save_confusion_matrix(cm, labels)

    # ======================================================
    # NIHAI V3 FINAL MODEL
    # Tum mevcut egitim verisi kullanilir.
    # Bu model canli sistemde kullanilacak final modeldir.
    # ======================================================

    print()
    print("Nihai V3 final model tum mevcut gercek verilerle egitiliyor...")

    final_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=1
    )

    final_model.fit(X, y)

    joblib.dump(
        {
            "model": final_model,
            "feature_columns": list(X.columns),
            "labels": labels,
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "model_version": "V3_FINAL",
            "training_raw_rows": total_raw_rows,
            "training_window_count": len(X),
            "training_files": TRAINING_FILES,
        },
        MODEL_PATH
    )

    report_text = ""
    report_text += "AKILLI BARET - V3 FINAL MODEL IC KONTROL RAPORU\n"
    report_text += "================================================\n\n"
    report_text += f"Toplam ham egitim satiri: {total_raw_rows}\n"
    report_text += f"Toplam pencere sayisi: {len(X)}\n"
    report_text += f"Feature sayisi: {X.shape[1]}\n\n"
    report_text += "NOT: Asagidaki sonuc final bagimsiz test sonucu degildir.\n"
    report_text += "Bu sonuc yalnizca V3 Final Model ic kontrol amaciyla uretilmistir.\n\n"
    report_text += f"Ic kontrol accuracy: {accuracy:.4f}\n"
    report_text += f"Ic kontrol accuracy yuzde: %{accuracy * 100:.2f}\n\n"
    report_text += "Classification Report:\n"
    report_text += report
    report_text += "\nConfusion Matrix:\n"
    report_text += pd.DataFrame(cm, index=labels, columns=labels).to_string()

    with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
        report_file.write(report_text)

    print()
    print(f"V3 final model kaydedildi: {MODEL_PATH}")
    print(f"V3 ic kontrol grafigi kaydedildi: {CONFUSION_MATRIX_PATH}")
    print(f"V3 ic kontrol raporu kaydedildi: {REPORT_PATH}")

    print()
    print("SONRAKI ADIM:")
    print("Bu model prototipin final demo ve tahmin modeli olarak kullanilacak.")


if __name__ == "__main__":
    main()
