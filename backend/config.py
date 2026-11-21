"""Central configuration.

Every path is resolved relative to this file, so the server works no matter
which directory it is started from. Every setting can be overridden with an
environment variable (useful for Docker / Hugging Face Spaces).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# -----------------------------
# Model artefacts
# -----------------------------
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "dmis-lab/biobert-base-cased-v1.1")

MODEL_PATH = Path(os.getenv("MODEL_PATH", BACKEND_DIR / "models" / "hybrid_model.pth"))

# Where to download the weights from when MODEL_PATH does not exist.
# MODEL_HF_REPO (a Hugging Face model repo, e.g. "user/mimic-hybrid") is preferred
# when set because it is faster and more reliable than Google Drive.
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://drive.google.com/uc?id=1g9yuOIIFgxG-xR4_ri6w6LRpv9OBsvQw",
)
MODEL_HF_REPO = os.getenv("MODEL_HF_REPO", "")
MODEL_HF_FILENAME = os.getenv("MODEL_HF_FILENAME", "hybrid_model.pth")

# Optional SHA-256 of the weights file. When set, a downloaded file that does
# not match is deleted instead of being loaded.
MODEL_SHA256 = os.getenv("MODEL_SHA256", "").strip().lower()

# A real BioBERT state dict is ~430 MB. Anything far smaller is a failed
# download (for example a Google Drive HTML quota page).
MIN_MODEL_BYTES = _env_int("MIN_MODEL_BYTES", 50 * 1024 * 1024)

SCALER_PATH = Path(os.getenv("SCALER_PATH", BACKEND_DIR / "scaler" / "scaler.json"))

# -----------------------------
# Inference settings
# -----------------------------
# The SAME max length is used for prediction, attention and SHAP/LIME so the
# explanations describe exactly the input that was scored.
MAX_LENGTH = _env_int("MAX_LENGTH", 256)

# Decision threshold on the predicted mortality probability. A threshold below
# 0.5 trades precision for recall, which is usual for rare, high-cost outcomes
# such as in-hospital mortality.
THRESHOLD = _env_float("THRESHOLD", 0.3)

# The original pipeline lower-cased notes before tokenisation, so the published
# weights expect lower-cased text. Set to "false" only for a model trained on
# original-case notes.
LOWERCASE_TEXT = _env_bool("LOWERCASE_TEXT", True)

MAX_TEXT_CHARS = _env_int("MAX_TEXT_CHARS", 20000)

# Explainability budget (cheap now that BioBERT runs once per request).
SHAP_NSAMPLES = _env_int("SHAP_NSAMPLES", 1024)
LIME_NUM_SAMPLES = _env_int("LIME_NUM_SAMPLES", 2000)
RANDOM_SEED = _env_int("RANDOM_SEED", 42)

# -----------------------------
# Server settings
# -----------------------------
# Comma separated list. Empty means "same origin only" (the UI is served by
# this same server, so no CORS is needed by default).
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

# Load the model in a background thread at startup so the web server becomes
# reachable immediately (important for hosting health checks).
LOAD_MODEL_ON_STARTUP = _env_bool("LOAD_MODEL_ON_STARTUP", True)


# -----------------------------
# Structured features
# -----------------------------
@dataclass(frozen=True)
class Feature:
    name: str
    label: str
    unit: str
    min: float
    max: float
    # Plausible clinical range used to clip values before scaling (training and
    # inference must use the same clipping).
    clip_min: float
    clip_max: float


# Order matters: it must match the column order used to train the model.
FEATURES: tuple[Feature, ...] = (
    Feature("age", "Age", "years", 0, 120, 18, 100),
    Feature("gender", "Gender", "0 = male, 1 = female", 0, 1, 0, 1),
    Feature("heart_rate", "Heart rate", "bpm", 0, 300, 20, 250),
    Feature("sbp", "Systolic BP", "mmHg", 0, 300, 40, 260),
    Feature("dbp", "Diastolic BP", "mmHg", 0, 250, 15, 180),
    Feature("mbp", "Mean BP", "mmHg", 0, 300, 25, 200),
    Feature("spo2", "SpO2", "%", 0, 100, 50, 100),
    Feature("creatinine", "Creatinine", "mg/dL", 0, 30, 0.1, 20),
    Feature("glucose", "Glucose", "mg/dL", 0, 2000, 20, 1000),
    Feature("hemoglobin", "Hemoglobin", "g/dL", 0, 25, 3, 22),
)

FEATURE_NAMES: list[str] = [f.name for f in FEATURES]
FEATURE_LABELS: dict[str, str] = {f.name: f.label for f in FEATURES}
N_FEATURES = len(FEATURES)

DISCLAIMER = (
    "Research prototype for educational purposes only. Not a medical device and "
    "not validated for clinical decision-making."
)
