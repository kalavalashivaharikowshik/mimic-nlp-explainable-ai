# backend/explain.py

import matplotlib
matplotlib.use("Agg")

import torch
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

from load_model import model, scaler, device, tokenizer

import numpy as np
import io
import base64

# -----------------------------
# LOAD BACKGROUND DATA
# -----------------------------
background_data = np.load("background.npy")   # real data for LIME
background_shap = shap.kmeans(background_data, 30)  # optimized for SHAP

# -----------------------------
# FEATURE NAMES
# -----------------------------
feature_names = [
    "age", "gender",
    "heart_rate", "sbp", "dbp", "mbp", "spo2",
    "creatinine", "glucose", "hemoglobin"
]

# -----------------------------
# ATTENTION (TEXT)
# -----------------------------
def get_attention_words(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model.text_model(**inputs, output_attentions=True)

    attentions = outputs.attentions[-1]
    scores = attentions.mean(dim=1).squeeze(0)[0]

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    scores = scores.cpu().numpy()

    merged_words = []
    current_word = ""
    current_score = 0.0

    for token, score in zip(tokens, scores):

        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        if token.startswith("##"):
            current_word += token[2:]
            current_score += score
        else:
            if current_word != "":
                merged_words.append((current_word, current_score))
            current_word = token
            current_score = score

    if current_word != "":
        merged_words.append((current_word, current_score))

    merged_words = sorted(merged_words, key=lambda x: x[1], reverse=True)

    seen = set()
    final_words = []
    for word, score in merged_words:
        if (
            word not in seen
            and len(word) > 3
            and word.isalpha()
        ):
            seen.add(word)
            final_words.append(word)
        if len(final_words) == 8:
            break

    return final_words

# -----------------------------
# MODEL FUNCTION (REAL TEXT)
# -----------------------------
def model_fn(x, text):
    x_tensor = torch.tensor(x, dtype=torch.float).to(device)

    enc = tokenizer(
        [text] * len(x),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(
            enc["input_ids"],
            enc["attention_mask"],
            x_tensor
        ).squeeze(-1)

        probs = torch.sigmoid(outputs).cpu().numpy()

    return probs

# -----------------------------
# SHAP GRAPH (BASE64) ✅ FIXED
# -----------------------------
def generate_shap_plot(features, text):
    x = scaler.transform([features])

    explainer = shap.KernelExplainer(
        lambda data: model_fn(data, text),
        background_shap
    )

    shap_values = explainer.shap_values(x, nsamples=20)

    plt.figure()
    shap.summary_plot(
        shap_values,
        x,
        feature_names=feature_names,
        show=False
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close()

    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
# -----------------------------
# LIME GRAPH (BASE64)
# -----------------------------
def generate_lime_plot(features, text):

    explainer = LimeTabularExplainer(
        background_data,
        feature_names=feature_names,
        class_names=["Alive", "Dead"],
        mode="classification"
    )

    def predict_fn(x):
        probs = model_fn(x, text)
        return np.vstack([1 - probs, probs]).T

    features_scaled = scaler.transform([features])[0]

    exp = explainer.explain_instance(
        features_scaled,
        predict_fn,
        num_features=5
    )

    fig = exp.as_pyplot_figure()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close()

    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# -----------------------------
# SIMPLE FEATURE IMPORTANCE
# -----------------------------
def get_top_features(features):
    features = scaler.transform([features])[0]
    importance = np.abs(features)
    top_idx = np.argsort(importance)[-5:]
    return [feature_names[i] for i in top_idx]