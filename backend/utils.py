# backend/utils.py

import numpy as np

# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = text.strip()

    return text


# -----------------------------
# FEATURE VALIDATION
# -----------------------------
EXPECTED_FEATURES = 10

def validate_features(features):
    if not isinstance(features, list):
        return False, "Features must be a list"

    if len(features) != EXPECTED_FEATURES:
        return False, f"Expected {EXPECTED_FEATURES} features"

    return True, None


# -----------------------------
# FEATURE FORMATTING
# -----------------------------
def format_features(features):
    try:
        return np.array(features, dtype=float)
    except:
        return None


# -----------------------------
# RESPONSE FORMATTER
# -----------------------------
def format_response(prob, label, words, features):
    return {
        "probability": round(prob, 4),
        "prediction": label,
        "explanation": {
            "important_words": words,
            "top_features": features
        }
    }