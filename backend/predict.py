# backend/predict.py

import torch
from load_model import model, tokenizer, scaler, device

from explain import (
    get_attention_words,
    get_top_features,
    generate_shap_plot,
    generate_lime_plot
)

from utils import clean_text, validate_features, format_features

THRESHOLD = 0.3


# -----------------------------
# 🔥 HIGHLIGHT TEXT
# -----------------------------
def highlight_text(text, important_words):
    highlighted = text

    for word in important_words:
        if len(word) > 3:
            highlighted = highlighted.replace(
                word,
                f"<mark>{word}</mark>"
            )

    return highlighted


# -----------------------------
# 🔥 DOCTOR STYLE EXPLANATION
# -----------------------------
def generate_reason(important_words, top_features):

    reasons = []

    if "sepsis" in important_words:
        reasons.append("Sepsis detected")

    if "ventilator" in important_words or "ventilation" in important_words:
        reasons.append("Patient on ventilator support")

    if "oxygen" in important_words or "saturation" in important_words:
        reasons.append("Oxygen level abnormal")

    if "creatinine" in top_features:
        reasons.append("Kidney function abnormal")

    if "glucose" in top_features:
        reasons.append("Glucose levels elevated")

    if not reasons:
        reasons.append("General clinical risk factors observed")

    return ", ".join(reasons)


# -----------------------------
# 🔥 MAIN PREDICT FUNCTION
# -----------------------------
def predict(text, features):

    # -----------------------------
    # VALIDATION
    # -----------------------------
    valid, error = validate_features(features)
    if not valid:
        return {"error": error}

    text = clean_text(text)
    features = format_features(features)

    if features is None:
        return {"error": "Invalid feature format"}

    # -----------------------------
    # SCALE FEATURES
    # -----------------------------
    scaled = scaler.transform([features])

    # -----------------------------
    # TOKENIZE TEXT
    # -----------------------------
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    structured = torch.tensor(scaled, dtype=torch.float).to(device)

    # -----------------------------
    # PREDICTION
    # -----------------------------
    with torch.no_grad():
        output = model(input_ids, attention_mask, structured).squeeze()
        prob = torch.sigmoid(output).item()

    label = "HIGH RISK" if prob > THRESHOLD else "LOW RISK"

    # -----------------------------
    # EXPLANATIONS
    # -----------------------------
    important_words = get_attention_words(text)
    top_features = get_top_features(features)

    # 🔥 SHAP + LIME (REAL)
    shap_plot = generate_shap_plot(features, text)
    lime_plot = generate_lime_plot(features, text)

    # 🔥 Highlight + Reason
    highlighted_text = highlight_text(text, important_words)
    reason = generate_reason(important_words, top_features)

    # 🔥 TEXT QUALITY MESSAGE
    if len(text.strip()) < 20:
        note_msg = "Prediction mainly driven by structured clinical features (vitals/labs)"
    else:
        note_msg = "Prediction influenced by both clinical notes and patient vitals"

    # -----------------------------
    # FINAL RESPONSE
    # -----------------------------
    return {
        "prediction": label,
        "probability": prob,

        "explanation": {
            "important_words": important_words,
            "top_features": top_features,
            "highlighted_text": highlighted_text
        },

        "reason": reason,
        "note_message": note_msg,

        "shap_plot": shap_plot,
        "lime_plot": lime_plot
    }