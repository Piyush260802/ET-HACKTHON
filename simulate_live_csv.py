import argparse
import time
from pathlib import Path

import pandas as pd
import requests


# ======================================================
# AKILLI BARET - CSV CANLI AKIS SIMULASYONU
# ======================================================

DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
DEFAULT_HELMET_ID = "Baret-01"
DEFAULT_CSV_PATH = "data/demo_input/demo_unlabeled_scenario_01.csv"

BATCH_SIZE = 20  # Sunucunun her pakette beklediği satır sayısı


def parse_args():
    parser = argparse.ArgumentParser(
        description="CSV sensör verisini canlı ESP32 akışı gibi sunucuya gönderir."
    )

    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV_PATH,
        help="Gönderilecek etiketsiz CSV dosyası.",
    )

    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER_URL,
        help="Canlı AI sunucu adresi.",
    )

    parser.add_argument(
        "--helmet-id",
        default=DEFAULT_HELMET_ID,
        help="Baret kimliği.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Her 20 satırlık paket arasındaki bekleme süresi. "
             "Gerçek zaman için 1.0, hızlı test için 0.05 kullan.",
    )

    return parser.parse_args()


def prepare_dataframe(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV dosyası bulunamadı: {csv_path}")

    df = pd.read_csv(path)

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
            f"CSV dosyasında eksik sütunlar var: {missing_columns}"
        )

    df = df[required_columns].copy()

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    usable_row_count = (len(df) // BATCH_SIZE) * BATCH_SIZE
    df = df.iloc[:usable_row_count].copy()

    return df


def reset_live_status(server_url: str, helmet_id: str):
    response = requests.post(
        f"{server_url}/api/reset/{helmet_id}",
        timeout=10,
    )

    response.raise_for_status()

    print(f"Canlı durum sıfırlandı: {helmet_id}")


def send_batch(server_url: str, helmet_id: str, batch_df: pd.DataFrame):
    samples = []

    for _, row in batch_df.iterrows():
        samples.append(
            {
                "time_ms": int(row["time_ms"]),
                "acc_x": float(row["acc_x"]),
                "acc_y": float(row["acc_y"]),
                "acc_z": float(row["acc_z"]),
                "gyro_x": float(row["gyro_x"]),
                "gyro_y": float(row["gyro_y"]),
                "gyro_z": float(row["gyro_z"]),
                "temp_c": float(row["temp_c"]),
            }
        )

    response = requests.post(
        f"{server_url}/api/samples/batch",
        json={
            "helmet_id": helmet_id,
            "samples": samples,
        },
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


def main():
    args = parse_args()

    df = prepare_dataframe(args.csv)

    total_batches = len(df) // BATCH_SIZE
    total_seconds = total_batches

    print("======================================")
    print("AKILLI BARET - CANLI CSV SIMULASYONU")
    print("======================================")
    print(f"CSV dosyası: {args.csv}")
    print(f"Baret ID: {args.helmet_id}")
    print(f"Sunucu: {args.server}")
    print(f"Kullanılacak satır: {len(df)}")
    print(f"Gönderilecek paket: {total_batches}")
    print(f"Temsil edilen sensör süresi: yaklaşık {total_seconds} saniye")
    print(f"Paket bekleme süresi: {args.delay} saniye")
    print()

    reset_live_status(args.server, args.helmet_id)

    for batch_no in range(total_batches):
        start = batch_no * BATCH_SIZE
        end = start + BATCH_SIZE

        batch_df = df.iloc[start:end]

        status = send_batch(
            args.server,
            args.helmet_id,
            batch_df,
        )

        if status["ready"]:
            print(
                f"Paket {batch_no + 1:3}/{total_batches} | "
                f"Durum: {status['current_label_tr']:<22} | "
                f"Güven: %{status['current_confidence']:>6.2f} | "
                f"Tahmin: {status['prediction_count']}"
            )
        else:
            print(
                f"Paket {batch_no + 1:3}/{total_batches} | "
                "İlk tahmin için veri birikiyor..."
            )

        time.sleep(args.delay)

    print()
    print("======================================")
    print("CANLI AKIS TAMAMLANDI")
    print("======================================")
    print(f"Son durum: {status['current_label_tr']}")
    print(f"Son güven: %{status['current_confidence']:.2f}")
    print(f"Toplam tahmin: {status['prediction_count']}")
    print()
    print("Durum süreleri:")

    for label, seconds in status["state_seconds"].items():
        print(f"{label:12}: {seconds:.1f} saniye")


if __name__ == "__main__":
    main()