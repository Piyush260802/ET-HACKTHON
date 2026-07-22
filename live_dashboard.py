from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# ======================================================
# AKILLI BARET - CANLI YONETICI PANELI
# ======================================================

SERVER_URL = "http://127.0.0.1:8000"
HELMET_ID = "Baret-01"

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

LABEL_NUMBER = {
    "On Head": 1,
    "On Belt": 2,
    "In Hand": 3,
    "On Surface": 4,
}

STATUS_CONFIG = {
    "safe": {
        "icon": "✅",
        "text": "Güvenli Kullanım",
        "message": "Baret çalışanın kafasında.",
    },
    "warning": {
        "icon": "⚠️",
        "text": "Uyarı",
        "message": "Baret çalışanın kafasında değil.",
    },
    "danger": {
        "icon": "❌",
        "text": "Riskli Durum",
        "message": "Baret çalışanın kafasında değil; yüzeyde bırakılmış görünüyor.",
    },
    "waiting": {
        "icon": "⏳",
        "text": "Veri Bekleniyor",
        "message": "İlk tahmin için sensör verisi bekleniyor.",
    },
}


def format_seconds(seconds):
    total_seconds = int(seconds or 0)
    return str(timedelta(seconds=total_seconds))


def get_status():
    response = requests.get(
        f"{SERVER_URL}/api/status/{HELMET_ID}",
        timeout=3,
    )
    response.raise_for_status()
    return response.json()


def reset_status():
    response = requests.post(
        f"{SERVER_URL}/api/reset/{HELMET_ID}",
        timeout=3,
    )
    response.raise_for_status()


def create_history_chart(history):
    if not history:
        return None

    history_df = pd.DataFrame(history)
    history_df["durum_no"] = history_df["stable_label"].map(LABEL_NUMBER)

    fig, ax = plt.subplots(figsize=(11, 4.4))

    ax.plot(
        history_df["prediction_no"],
        history_df["durum_no"],
        marker="o",
        markersize=4,
        linewidth=2,
    )

    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(
        [
            "Baret Kafada",
            "Baret Kemerde / Belde",
            "Baret Elde",
            "Baret Yüzeyde",
        ]
    )

    ax.set_xlabel("Anlık Tahmin Sırası")
    ax.set_ylabel("Tahmin Edilen Durum")
    ax.set_title("Canlı Baret Durumu Geçmişi")
    ax.grid(True)
    fig.tight_layout()

    return fig


def create_duration_chart(state_seconds):
    duration_values = [
        float(state_seconds.get(label, 0))
        for label in LABEL_ORDER
    ]

    fig, ax = plt.subplots(figsize=(7, 4.4))

    ax.bar(
        [
            "Kafada",
            "Kemerde / Belde",
            "Elde",
            "Yüzeyde",
        ],
        duration_values,
    )

    ax.set_xlabel("Baret Durumu")
    ax.set_ylabel("Süre (saniye)")
    ax.set_title("Durumlara Göre Geçen Süre")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    return fig


# ======================================================
# STREAMLIT ARAYUZU
# ======================================================

st.set_page_config(
    page_title="Akıllı Baret Canlı Takip Paneli",
    page_icon="⛑️",
    layout="wide",
)

# Her 1 saniyede bir sayfayı yeniden çalıştırır.
st_autorefresh(interval=1000, key="live_refresh")

st.title("⛑️ Akıllı Baret Canlı Takip Paneli")
st.caption(
    "ESP32 + MPU6050 sensör verileri Wi-Fi üzerinden işlenerek "
    "baretin mevcut kullanım durumu anlık olarak tahmin edilir."
)

with st.sidebar:
    st.header("Sistem Bilgisi")
    st.write("**Baret Kimliği:** Baret-01")
    st.write("**Model:** V3 Final Random Forest")
    st.write("**Tahmin Yenileme:** Yaklaşık 1 saniye")
    st.write("**Algılanabilen Durumlar:**")
    st.write("- ✅ Baret Kafada")
    st.write("- ⚠️ Baret Kemerde / Belde")
    st.write("- ⚠️ Baret Elde")
    st.write("- ❌ Baret Yüzeyde")

    st.divider()

    if st.button("Canlı Durumu Sıfırla", width="stretch"):
        try:
            reset_status()
            st.success("Baret durumu sıfırlandı.")
        except Exception as error:
            st.error(f"Sıfırlama hatası: {error}")

try:
    status = get_status()
except Exception:
    st.error(
        "Canlı AI sunucusuna ulaşılamıyor. "
        "Önce `uvicorn live_server:app --host 0.0.0.0 --port 8000` "
        "komutuyla sunucuyu başlat."
    )
    st.stop()

connected = status["connected"]
ready = status["ready"]
status_level = status["status_level"]
current_label = status["current_label"]
current_label_tr = status["current_label_tr"]
current_confidence = status["current_confidence"]
state_seconds = status["state_seconds"]
history = status["history"]

connection_text = "🟢 Çevrimiçi" if connected else "🔴 Veri Akışı Yok"

st.divider()

top1, top2, top3, top4 = st.columns(4)

with top1:
    st.metric("Baret", status["helmet_id"])

with top2:
    st.metric("Bağlantı", connection_text)

with top3:
    st.metric("Alınan Sensör Satırı", status["samples_received"])

with top4:
    st.metric("Üretilen Tahmin", status["prediction_count"])

st.divider()

if not ready:
    st.info(
        f"İlk tahmin için veri bekleniyor: "
        f"{status['samples_in_buffer']} / "
        f"{status['required_samples_for_first_prediction']} sensör satırı alındı."
    )
    st.stop()

config = STATUS_CONFIG[status_level]

st.subheader("Anlık Baret Durumu")

state1, state2, state3, state4 = st.columns(4)

with state1:
    st.metric(
        "Mevcut Durum",
        f"{config['icon']} {current_label_tr}"
    )

with state2:
    st.metric(
        "Güven Oranı",
        f"%{current_confidence:.2f}"
    )

with state3:
    st.metric(
        "Kafada Geçen Süre",
        format_seconds(status["on_head_seconds"])
    )

with state4:
    st.metric(
        "Kafada Değil Süresi",
        format_seconds(status["not_on_head_seconds"])
    )

if status_level == "safe":
    st.success(f"{config['text']}: {config['message']}")
elif status_level == "warning":
    st.warning(f"{config['text']}: {config['message']}")
elif status_level == "danger":
    st.error(f"{config['text']}: {config['message']}")

st.divider()

chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("Canlı Durum Geçmişi")
    history_chart = create_history_chart(history)

    if history_chart is not None:
        st.pyplot(history_chart)
    else:
        st.info("Henüz grafik için tahmin verisi oluşmadı.")

with chart_right:
    st.subheader("Durum Süreleri")
    st.pyplot(create_duration_chart(state_seconds))

st.divider()

st.subheader("Son Tahminler")

if history:
    history_df = pd.DataFrame(history).tail(15).copy()

    history_df = history_df[
        [
            "prediction_no",
            "time",
            "stable_label_tr",
            "confidence",
        ]
    ]

    history_df.columns = [
        "Tahmin No",
        "Saat",
        "Durum",
        "Güven (%)",
    ]

    st.dataframe(
        history_df.iloc[::-1],
        width="stretch",
        hide_index=True,
    )
else:
    st.info("Henüz tahmin verisi bulunmuyor.")

st.caption(
    "Canlı panel, sensör verisini AI sunucusundan alır. "
    "Normal kullanımda SD karttan veri yüklenmesine gerek yoktur."
)