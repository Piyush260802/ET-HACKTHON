import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


DATA_PATH = "data/raw/helmet_data.csv"
MODEL_PATH = "models/helmet_real_rf_model.joblib"
CONFUSION_MATRIX_PATH = "data/processed/real_confusion_matrix.png"

WINDOW_SIZE = 40   # 20 Hz veri için yaklaşık 2 saniye
STEP_SIZE = 20     # 1 saniye kaydırma

EXPECTED_LABELS = [
    "On Head",
    "On Belt",
    "In Hand",
    "On Surface"
]

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
    print("Ham veri boyutu:", df.shape)

    expected_columns = [
        "time_ms",
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "temp_c",
        "label"
    ]

    missing_columns = [col for col in expected_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Eksik sütunlar var: {missing_columns}")

    # Tekrar yazılmış CSV başlık satırları varsa temizle
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

    df["label"] = df["label"].astype(str).str.strip()

    # Sadece beklenen 4 sınıfı al
    df = df[df["label"].isin(EXPECTED_LABELS)].copy()

    # Bozuk satırları temizle
    df = df.dropna(subset=numeric_columns + ["label"])

    # Ek özellikler
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

    print("Temiz veri boyutu:", df.shape)
    print("\nLabel dağılımı:")
    print(df["label"].value_counts())

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


def create_feature_table(df):
    feature_rows = []
    labels = []

    print("\nHer sinif kendi icinde pencereleniyor...")

    for label in EXPECTED_LABELS:
        class_df = df[df["label"] == label].reset_index(drop=True)

        print(f"{label}: {len(class_df)} ham satir")

        class_window_count = 0

        for start in range(0, len(class_df) - WINDOW_SIZE + 1, STEP_SIZE):
            end = start + WINDOW_SIZE
            window = class_df.iloc[start:end]

            features = extract_window_features(window)

            feature_rows.append(features)
            labels.append(label)

            class_window_count += 1

        print(f"{label}: {class_window_count} pencere")

    X = pd.DataFrame(feature_rows)
    y = pd.Series(labels, name="label")

    print("\nOzellik tablosu boyutu:", X.shape)
    print("Pencere label dagilimi:")
    print(y.value_counts())

    return X, y


def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Gerçek Veri Confusion Matrix")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

    print(f"\nConfusion matrix grafiği kaydedildi: {CONFUSION_MATRIX_PATH}")


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CSV bulunamadı: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = clean_data(df)

    if df["label"].nunique() < 2:
        raise ValueError("Model eğitimi için en az 2 farklı label gerekli.")

    X, y = create_feature_table(df)

    if len(X) < 10:
        raise ValueError("Yeterli pencere oluşmadı. Daha fazla veri topla.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n==============================")
    print("MODEL SONUÇLARI")
    print("==============================")
    print("Accuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    plot_confusion_matrix(cm, labels)

    print("\nNihai model tum gercek egitim verisiyle yeniden egitiliyor...")
    model.fit(X, y)
    
    joblib.dump(
        {
            "model": model,
            "feature_columns": list(X.columns),
            "labels": labels,
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE
        },
        MODEL_PATH
    )

    print(f"\nModel kaydedildi: {MODEL_PATH}")


if __name__ == "__main__":
    main()