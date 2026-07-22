import os
import pandas as pd


RAW_DATA_DIR = "data/raw"

INPUT_FILES = [
    ("on_head_01.csv", "On Head"),
    ("on_belt_01.csv", "On Belt"),
    ("in_hand_01.csv", "In Hand"),
    ("on_surface_01.csv", "On Surface"),
]

OUTPUT_PATH = os.path.join(RAW_DATA_DIR, "helmet_data.csv")

EXPECTED_COLUMNS = [
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


def main():
    combined_frames = []

    print("======================================")
    print("GERCEK VERI DOSYALARI BIRLESTIRILIYOR")
    print("======================================")

    for file_name, expected_label in INPUT_FILES:
        file_path = os.path.join(RAW_DATA_DIR, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dosya bulunamadi: {file_path}")

        df = pd.read_csv(file_path)

        missing_columns = [
            column for column in EXPECTED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{file_name} dosyasinda eksik sutunlar var: {missing_columns}"
            )

        df = df[EXPECTED_COLUMNS].copy()

        # Label bosluklarini temizle
        df["label"] = df["label"].astype(str).str.strip()

        # Sayisal kolonlari kontrol et
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

        broken_rows = df.isna().any(axis=1).sum()

        if broken_rows > 0:
            raise ValueError(
                f"{file_name} dosyasinda {broken_rows} bozuk/bos satir var."
            )

        unique_labels = df["label"].unique().tolist()

        if unique_labels != [expected_label]:
            raise ValueError(
                f"{file_name} label hatasi. "
                f"Beklenen: {expected_label}, bulunan: {unique_labels}"
            )

        print()
        print(f"Dosya: {file_name}")
        print(f"Label: {expected_label}")
        print(f"Satir sayisi: {len(df)}")
        print(f"Kayit suresi: {df['time_ms'].iloc[-1] / 1000:.2f} saniye")

        combined_frames.append(df)

    combined_df = pd.concat(combined_frames, ignore_index=True)

    combined_df.to_csv(OUTPUT_PATH, index=False)

    print()
    print("======================================")
    print("BIRLESTIRME TAMAMLANDI")
    print("======================================")
    print(f"Yeni dosya: {OUTPUT_PATH}")
    print(f"Toplam satir sayisi: {len(combined_df)}")

    print()
    print("Label dagilimi:")
    print(combined_df["label"].value_counts())

    print()
    print("Artik model egitimi icin hazir.")


if __name__ == "__main__":
    main()