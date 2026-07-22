import os
import pandas as pd


FIXES = [
    {
        "input_path": "data/test2/test2_on_belt_01.csv",
        "output_path": "data/test2/test2_on_belt_01.csv",
        "new_label": "On Belt",
    },
    {
        "input_path": "data/test2/test2_on_hand_01.csv",
        "output_path": "data/test2/test2_in_hand_01.csv",
        "new_label": "In Hand",
    },
]


def main():
    for item in FIXES:
        input_path = item["input_path"]
        output_path = item["output_path"]
        new_label = item["new_label"]

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Dosya bulunamadi: {input_path}")

        df = pd.read_csv(input_path)

        print(f"\nDosya: {input_path}")
        print("Eski label dagilimi:")
        print(df["label"].value_counts())

        df["label"] = new_label
        df.to_csv(output_path, index=False)

        print(f"Yeni dosya kaydedildi: {output_path}")
        print("Yeni label dagilimi:")
        print(df["label"].value_counts())

    print("\nDuzeltme tamamlandi.")
    print("test2_on_hand_01.csv artik kullanilmayacak.")
    print("Yeni dogru dosya: test2_in_hand_01.csv")


if __name__ == "__main__":
    main()