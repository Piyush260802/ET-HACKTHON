import os
import pandas as pd


# ======================================================
# ETIKETSIZ JURI DEMO SENARYOSU
# ======================================================
# Bu dosya bilimsel bagimsiz test icin degil,
# panelde dört durumun ardışık gösterimi icin oluşturulur.
# Cikista label sutunu bulunmaz.
# ======================================================

OUTPUT_PATH = "data/demo_input/demo_unlabeled_scenario_01.csv"

# Her durumdan 30 saniye veri alinacak.
# 20 Hz veri hizinda: 30 x 20 = 600 satir
SEGMENT_ROWS = 600

SOURCE_FILES = [
    {
        "path": "data/test2/test2_on_head_01.csv",
        "name": "On Head",
    },
    {
        "path": "data/test2/test2_on_belt_01.csv",
        "name": "On Belt",
    },
    {
        "path": "data/test2/test2_in_hand_01.csv",
        "name": "In Hand",
    },
    {
        "path": "data/test2/test2_on_surface_01.csv",
        "name": "On Surface",
    },
]

OUTPUT_COLUMNS = [
    "time_ms",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "temp_c",
]


def main():
    os.makedirs("data/demo_input", exist_ok=True)

    combined_parts = []

    print("======================================")
    print("ETIKETSIZ JURI DEMO SENARYOSU")
    print("======================================")
    print()
    print("Not: Bu dosya görsel demo içindir; bağımsız test sonucu değildir.")
    print()

    for source in SOURCE_FILES:
        file_path = source["path"]
        class_name = source["name"]

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dosya bulunamadi: {file_path}")

        df = pd.read_csv(file_path)

        missing_columns = [
            column for column in OUTPUT_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{file_path} dosyasinda eksik sutunlar var: {missing_columns}"
            )

        if len(df) < SEGMENT_ROWS:
            raise ValueError(
                f"{file_path} dosyasinda yeterli satir yok. "
                f"Gerekli: {SEGMENT_ROWS}, bulunan: {len(df)}"
            )

        # Kaydin ortasindan 30 saniyelik bolum alinir.
        # Boylece baslangic/bitis hazirlik etkileri azalir.
        start_index = (len(df) - SEGMENT_ROWS) // 2
        end_index = start_index + SEGMENT_ROWS

        segment = df.iloc[start_index:end_index][OUTPUT_COLUMNS].copy()

        combined_parts.append(segment)

        print(f"{class_name:12} -> {SEGMENT_ROWS} satir eklendi.")

    combined_df = pd.concat(combined_parts, ignore_index=True)

    # Farkli dosyalarin zamanlari yeniden basladigi icin,
    # tek ve surekli demo zaman cizelgesi olusturulur.
    combined_df["time_ms"] = range(0, len(combined_df) * 50, 50)

    # Cikista label sutunu bilincli olarak YOKTUR.
    combined_df.to_csv(OUTPUT_PATH, index=False)

    print()
    print("======================================")
    print("DEMO DOSYASI HAZIR")
    print("======================================")
    print(f"Dosya: {OUTPUT_PATH}")
    print(f"Toplam satir: {len(combined_df)}")
    print(f"Toplam sure: {len(combined_df) * 50 / 1000:.0f} saniye")
    print()
    print("Senaryo sirasi:")
    print("0 - 30 saniye    : On Head verisi")
    print("30 - 60 saniye   : On Belt verisi")
    print("60 - 90 saniye   : In Hand verisi")
    print("90 - 120 saniye  : On Surface verisi")
    print()
    print("Cikista label sutunu bulunmamaktadir.")


if __name__ == "__main__":
    main()