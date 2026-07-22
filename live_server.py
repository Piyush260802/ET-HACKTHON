from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
import time

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ======================================================
# AKILLI BARET - CANLI AI SUNUCUSU
# ======================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "helmet_rf_model_v3_final.joblib"

HELMET_ID_DEFAULT = "Baret-01"

SAMPLE_HZ = 20

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

STATUS_LEVEL = {
    "On Head": "safe",
    "On Belt": "warning",
    "In Hand": "warning",
    "On Surface": "danger",
}

# Yeni bir duruma geçişte 2 ardışık tahmin istenir.
# Böylece tek pencerelik sıçramalar durum değiştirmez.
STABLE_REQUIRED_COUNT = 2


# ======================================================
# MODEL YÜKLEME
# ======================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Final model bulunamadi: {MODEL_PATH}")

saved_model = joblib.load(MODEL_PATH)

model = saved_model["model"]
feature_columns = saved_model["feature_columns"]
window_size = int(saved_model["window_size"])   # 40 satır = 2 saniye
step_size = int(saved_model["step_size"])       # 20 satır = 1 saniye

STEP_SECONDS = step_size / SAMPLE_HZ

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


# ======================================================
# GELEN VERİ MODELLERİ
# ======================================================

class SensorSample(BaseModel):
    time_ms: int
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    temp_c: float


class SampleBatch(BaseModel):
    helmet_id: str = HELMET_ID_DEFAULT
    samples: list[SensorSample]


# ======================================================
# HER BARET İÇİN CANLI DURUM HAFIZASI
# ======================================================

@dataclass
class HelmetRuntime:
    helmet_id: str
    buffer: deque = field(default_factory=lambda: deque(maxlen=window_size))
    history: deque = field(default_factory=lambda: deque(maxlen=300))

    total_samples_received: int = 0
    prediction_count: int = 0

    stable_label: str | None = None
    stable_confidence: float | None = None

    pending_label: str | None = None
    pending_count: int = 0

    state_seconds: dict = field(
        default_factory=lambda: {label: 0.0 for label in LABEL_ORDER}
    )

    last_received_at: float | None = None


runtimes: dict[str, HelmetRuntime] = {}
runtime_lock = Lock()


def get_runtime(helmet_id: str) -> HelmetRuntime:
    if helmet_id not in runtimes:
        runtimes[helmet_id] = HelmetRuntime(helmet_id=helmet_id)

    return runtimes[helmet_id]


# ======================================================
# ÖZELLİK ÇIKARMA VE TAHMİN
# ======================================================

def rms(values):
    return np.sqrt(np.mean(np.square(values)))


def extract_window_features(window: pd.DataFrame) -> dict:
    window = window.copy()

    window["acc_mag"] = np.sqrt(
        window["acc_x"] ** 2 +
        window["acc_y"] ** 2 +
        window["acc_z"] ** 2
    )

    window["gyro_mag"] = np.sqrt(
        window["gyro_x"] ** 2 +
        window["gyro_y"] ** 2 +
        window["gyro_z"] ** 2
    )

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


def predict_current_window(runtime: HelmetRuntime) -> dict:
    window = pd.DataFrame(list(runtime.buffer))

    features = extract_window_features(window)
    X = pd.DataFrame([features])[feature_columns]

    raw_label = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    probability_map = {
        class_name: float(probabilities[index])
        for index, class_name in enumerate(model.classes_)
    }

    raw_confidence = probability_map[raw_label]

    # İlk tahminde doğrudan mevcut durum oluşturulur.
    if runtime.stable_label is None:
        runtime.stable_label = raw_label
        runtime.stable_confidence = raw_confidence
        runtime.pending_label = None
        runtime.pending_count = 0

    # Tahmin mevcut sabit durumla aynıysa durum korunur.
    elif raw_label == runtime.stable_label:
        runtime.stable_confidence = raw_confidence
        runtime.pending_label = None
        runtime.pending_count = 0

    # Farklı bir durum geldiyse iki ardışık pencere doğrulaması beklenir.
    else:
        if raw_label == runtime.pending_label:
            runtime.pending_count += 1
        else:
            runtime.pending_label = raw_label
            runtime.pending_count = 1

        if runtime.pending_count >= STABLE_REQUIRED_COUNT:
            runtime.stable_label = raw_label
            runtime.stable_confidence = raw_confidence
            runtime.pending_label = None
            runtime.pending_count = 0

    runtime.prediction_count += 1

    if runtime.stable_label is not None:
        runtime.state_seconds[runtime.stable_label] += STEP_SECONDS

    history_item = {
        "prediction_no": runtime.prediction_count,
        "time": datetime.now().strftime("%H:%M:%S"),
        "raw_label": raw_label,
        "raw_label_tr": LABEL_TR[raw_label],
        "stable_label": runtime.stable_label,
        "stable_label_tr": LABEL_TR[runtime.stable_label],
        "confidence": round(raw_confidence * 100, 2),
        "status_level": STATUS_LEVEL[runtime.stable_label],
    }

    runtime.history.append(history_item)

    return {
        "raw_label": raw_label,
        "raw_confidence": raw_confidence,
        "probabilities": probability_map,
    }


def build_public_status(runtime: HelmetRuntime) -> dict:
    connected = (
        runtime.last_received_at is not None
        and time.time() - runtime.last_received_at < 5
    )

    current_label = runtime.stable_label

    on_head_seconds = runtime.state_seconds["On Head"]
    not_on_head_seconds = sum(
        seconds
        for label, seconds in runtime.state_seconds.items()
        if label != "On Head"
    )

    return {
        "helmet_id": runtime.helmet_id,
        "connected": connected,
        "ready": current_label is not None,
        "samples_received": runtime.total_samples_received,
        "samples_in_buffer": len(runtime.buffer),
        "required_samples_for_first_prediction": window_size,
        "prediction_count": runtime.prediction_count,

        "current_label": current_label,
        "current_label_tr": LABEL_TR[current_label] if current_label else "Veri Bekleniyor",
        "current_confidence": (
            round(runtime.stable_confidence * 100, 2)
            if runtime.stable_confidence is not None
            else None
        ),
        "status_level": STATUS_LEVEL[current_label] if current_label else "waiting",

        "state_seconds": {
            label: round(seconds, 1)
            for label, seconds in runtime.state_seconds.items()
        },
        "on_head_seconds": round(on_head_seconds, 1),
        "not_on_head_seconds": round(not_on_head_seconds, 1),

        "history": list(runtime.history),
    }


# ======================================================
# FASTAPI
# ======================================================

app = FastAPI(
    title="Akilli Baret Canli AI Sunucusu",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "V3 Final Random Forest",
        "model_path": str(MODEL_PATH),
        "window_size": window_size,
        "step_size": step_size,
        "expected_batch_size": step_size,
        "message": "Canli AI sunucusu calisiyor.",
    }


@app.get("/api/helmets")
def list_helmets():
    with runtime_lock:
        return {
            "helmets": [
                build_public_status(runtime)
                for runtime in runtimes.values()
            ]
        }


@app.get("/api/status/{helmet_id}")
def get_status(helmet_id: str):
    with runtime_lock:
        runtime = get_runtime(helmet_id)
        return build_public_status(runtime)


@app.post("/api/reset/{helmet_id}")
def reset_helmet(helmet_id: str):
    with runtime_lock:
        runtimes[helmet_id] = HelmetRuntime(helmet_id=helmet_id)

        return {
            "message": f"{helmet_id} canli durumu sifirlandi.",
            "helmet_id": helmet_id,
        }


@app.post("/api/samples/batch")
def receive_sample_batch(batch: SampleBatch):
    if len(batch.samples) != step_size:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Her paket {step_size} sensor satiri icermeli. "
                f"Gelen satir sayisi: {len(batch.samples)}"
            ),
        )

    with runtime_lock:
        runtime = get_runtime(batch.helmet_id)

        for sample in batch.samples:
            if hasattr(sample, "model_dump"):
                sample_dict = sample.model_dump()
            else:
                sample_dict = sample.dict()

            runtime.buffer.append(sample_dict)

        runtime.total_samples_received += len(batch.samples)
        runtime.last_received_at = time.time()

        if len(runtime.buffer) >= window_size:
            predict_current_window(runtime)

        status = build_public_status(runtime)

        return {
            "helmet_id": status["helmet_id"],
            "connected": status["connected"],
            "ready": status["ready"],
            "samples_received": status["samples_received"],
            "prediction_count": status["prediction_count"],
            "current_label": status["current_label"],
            "current_label_tr": status["current_label_tr"],
            "current_confidence": status["current_confidence"],
            "status_level": status["status_level"],
            "state_seconds": status["state_seconds"],
        }