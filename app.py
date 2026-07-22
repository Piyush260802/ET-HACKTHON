import os
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================================================
# AKILLI BARET AI PANELI - V3 FINAL MODEL
# ======================================================

MODEL_PATH = "models/helmet_rf_model_v3_final.joblib"

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

LABEL_TR = {
    "On Head": "Baret Kafada",
    "On Belt": "Baret Kemerde / Belde",
    "In Hand": "Baret Elde",
    "On Surface": "Baret Yüzeyde",
}

STATUS_EMOJI = {
    "On Head": "✅",
    "On Belt": "⚠️",
    "In Hand": "⚠️",
    "On Surface": "❌",
}


def rms(values):
    return np.sqrt(np.mean(np.square(values)))


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Final model bulunamadı: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


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
            f"CSV dosyasında eksik sütunlar var: {missing_columns}"
        )

    # Dosya içinde tekrar yazılmış başlık satırı varsa temizler.
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

        feature_rows.append(extract_window_features(window))

        metadata_rows.append(
            {
                "start_index": start,
                "end_index": end - 1,
                "start_time_ms": int(window["time_ms"].iloc[0]),
                "end_time_ms": int(window["time_ms"].iloc[-1]),
            }
        )

    return pd.DataFrame(feature_rows), pd.DataFrame(metadata_rows)


def prediction_timeline_chart(results):
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

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(time_seconds, y_values, marker="o", markersize=4)

    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(
        [
            "Baret Kafada",
            "Baret Kemerde / Belde",
            "Baret Elde",
            "Baret Yüzeyde",
        ]
    )

    ax.set_xlabel("Zaman (saniye)")
    ax.set_ylabel("Tahmin Edilen Durum")
    ax.set_title("Zaman İçindeki Baret Durumu Tahminleri")
    ax.grid(True)
    fig.tight_layout()

    return fig


def distribution_chart(results):
    counts = (
        results["predicted_label"]
        .value_counts()
        .reindex(LABEL_ORDER, fill_value=0)
    )

    turkish_labels = [
        LABEL_TR[label] for label in LABEL_ORDER
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(turkish_labels, counts.values)
    ax.set_xlabel("Baret Durumu")
    ax.set_ylabel("Tahmin Penceresi Sayısı")
    ax.set_title("Tahmin Dağılımı")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()

    return fig


def process_csv(df, saved_model):
    model = saved_model["model"]
    feature_columns = saved_model["feature_columns"]
    window_size = saved_model["window_size"]
    step_size = saved_model["step_size"]

    clean_df = clean_sensor_data(df)

    X, results = create_prediction_windows(
        clean_df,
        window_size,
        step_size
    )

    if len(X) == 0:
        raise ValueError(
            "Tahmin için yeterli sensör satırı yok. "
            "En az 40 satır veri gerekli."
        )

    X = X[feature_columns]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    results["predicted_label"] = predictions
    results["confidence"] = np.max(probabilities, axis=1)

    for index, class_name in enumerate(model.classes_):
        safe_name = class_name.replace(" ", "_").lower()
        results[f"prob_{safe_name}"] = probabilities[:, index]

    prediction_counts = results["predicted_label"].value_counts()
    prediction_percentages = (
        results["predicted_label"].value_counts(normalize=True) * 100
    )

    dominant_label = prediction_counts.index[0]
    dominant_percentage = float(prediction_percentages.iloc[0])

    last_windows = results.tail(min(5, len(results)))
    last_label = last_windows["predicted_label"].mode().iloc[0]
    last_confidence = float(last_windows["confidence"].mean() * 100)

    average_confidence = float(results["confidence"].mean() * 100)

    return {
        "clean_df": clean_df,
        "results": results,
        "dominant_label": dominant_label,
        "dominant_percentage": dominant_percentage,
        "last_label": last_label,
        "last_confidence": last_confidence,
        "average_confidence": average_confidence,
    }


# ======================================================
# STREAMLIT ARAYÜZÜ
# ======================================================

st.set_page_config(
    page_title="Akıllı Baret AI Paneli",
    page_icon="⛑️",
    layout="wide",
)

st.title("⛑️ Akıllı Baret AI Tahmin Paneli")
st.caption(
    "ESP32 + IMU sensör verilerini kullanarak baretin kullanım durumunu "
    "sınıflandıran V3 Final Model demosu."
)

with st.sidebar:
    st.header("Model Bilgisi")
    st.write("**Model:** V3 Final Random Forest")
    st.write("**Algılanabilen Durumlar:**")
    st.write("- **Baret Kafada** (`On Head`)")
    st.write("- **Baret Kemerde / Belde** (`On Belt`)")
    st.write("- **Baret Elde** (`In Hand`)")
    st.write("- **Baret Yüzeyde** (`On Surface`)")
    st.write("**Pencere:** 2 saniye")
    st.write("**Kaydırma:** 1 saniye")
    st.info(
        "Panel, CSV içindeki label sütununu tahmin için kullanmaz. "
        "Etiketsiz CSV dosyalarıyla da çalışır."
    )

try:
    saved_model = load_model()
    st.success("V3 Final Model başarıyla yüklendi.")
except Exception as error:
    st.error(str(error))
    st.stop()

uploaded_file = st.file_uploader(
    "Sensör CSV dosyasını yükle",
    type=["csv"],
    help="Örneğin demo_unlabeled_surface_01.csv dosyasını yükleyebilirsin.",
)

if uploaded_file is None:
    st.info(
        "Başlamak için etiketsiz demo CSV dosyanı yükle: "
        "`data/demo_input/demo_unlabeled_surface_01.csv`"
    )
    st.stop()

try:
    input_df = pd.read_csv(uploaded_file)

    has_label = "label" in input_df.columns

    if has_label:
        st.warning(
            "Yüklenen CSV içinde label sütunu bulunuyor. "
            "Ancak AI tahmin yaparken bu sütunu kullanmıyor."
        )
    else:
        st.success(
            "Yüklenen CSV etiketsizdir. Tahmin yalnızca sensör verisine göre yapılacaktır."
        )

    output = process_csv(input_df, saved_model)
    results = output["results"]

except Exception as error:
    st.error(f"Dosya işlenirken hata oluştu: {error}")
    st.stop()

dominant_label = output["dominant_label"]
last_label = output["last_label"]

prediction_percentages = (
    results["predicted_label"].value_counts(normalize=True) * 100
)

# Bir sınıf tahminlerin en az %10'unda görülmüşse algılanmış kabul edilir.
detected_labels = [
    label for label in LABEL_ORDER
    if float(prediction_percentages.get(label, 0.0)) >= 10.0
]

# Tek bir baskın durum yoksa ve birden fazla durum belirginse,
# dosya çoklu durum senaryosu olarak gösterilir.
is_multi_state = (
    len(detected_labels) >= 2
    and output["dominant_percentage"] < 70.0
)

st.divider()

st.subheader("AI Tahmin Sonucu")

col1, col2, col3, col4 = st.columns(4)

if is_multi_state:
    with col1:
        st.metric(
            "Senaryo",
            "🔄 Çoklu Durum"
        )

    with col2:
        st.metric(
            "Algılanan Durum",
            f"{len(detected_labels)} sınıf"
        )

    with col3:
        st.metric(
            "Son Durum",
            LABEL_TR[last_label]
        )

    with col4:
        st.metric(
            "Ortalama Güven",
            f"%{output['average_confidence']:.2f}"
        )

    detected_text = ", ".join(
        LABEL_TR[label] for label in detected_labels
    )

    st.info(
        "Çoklu durum senaryosu algılandı. "
        f"Algılanan durumlar: {detected_text}. "
        "Durumların zaman içindeki sırası aşağıdaki grafikte gösterilmektedir."
    )

else:
    with col1:
        st.metric(
            "Ana Karar",
            f"{STATUS_EMOJI[dominant_label]} {LABEL_TR[dominant_label]}"
        )

    with col2:
        st.metric(
            "Karar Oranı",
            f"%{output['dominant_percentage']:.2f}"
        )

    with col3:
        st.metric(
            "Son Durum",
            LABEL_TR[last_label]
        )

    with col4:
        st.metric(
            "Ortalama Güven",
            f"%{output['average_confidence']:.2f}"
        )

    if dominant_label == "On Head":
        st.success(
            "Baret doğru kullanım durumunda: çalışanın kafasında görünüyor."
        )
    elif dominant_label == "On Surface":
        st.error(
            "Baret çalışanın kafasında değil; sabit bir yüzeyde görünüyor."
        )
    else:
        st.warning(
            f"Baret doğru kullanım durumunda değil: {LABEL_TR[dominant_label]}."
        )

st.divider()

left_chart, right_chart = st.columns(2)

with left_chart:
    st.subheader("Tahmin Dağılımı")
    st.pyplot(distribution_chart(results))

with right_chart:
    st.subheader("Zaman İçindeki Tahmin")
    st.pyplot(prediction_timeline_chart(results))

st.divider()

st.subheader("Pencere Bazlı Sonuçlar")

display_results = results.copy()
display_results["confidence_percent"] = (
    display_results["confidence"] * 100
).round(2)

st.dataframe(
    display_results[
        [
            "start_time_ms",
            "end_time_ms",
            "predicted_label",
            "confidence_percent",
        ]
    ],
    width="stretch",
)

csv_output = results.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="Tahmin Sonuçlarını CSV Olarak İndir",
    data=csv_output,
    file_name="ai_prediction_results.csv",
    mime="text/csv",
)

st.caption(
    "Bilimsel not: Bu panel, V3 final modelinin demo tahmin aracıdır. "
    "V3 için %95,89 değeri iç kontroldür; bağımsız final test sonucu değildir."
)
