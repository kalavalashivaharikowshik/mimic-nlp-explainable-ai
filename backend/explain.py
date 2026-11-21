"""SHAP and LIME explanations for the structured (vitals / labs) features.

Performance fix
---------------
The clinical note is identical for every perturbed sample SHAP and LIME
create, so its BioBERT embedding is computed ONCE per request and only the
small fusion head is re-evaluated. The old code re-ran BioBERT for every
sample (600 SHAP + 5,000 LIME forward passes, LIME in a single batch that
needed several GB of memory), which took minutes on CPU or crashed with
out-of-memory errors.

Explanations work in raw clinical units (mmHg, mg/dL, ...) so the charts are
readable by clinicians; scaling happens inside the prediction function.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import shap
import torch

from .config import FEATURE_LABELS, FEATURE_NAMES, FEATURES, SHAP_NSAMPLES
from .plots import contribution_chart
from .utils import clip_features

PredictFn = Callable[[np.ndarray], np.ndarray]


def make_predict_fn(model, scaler, text_emb: torch.Tensor, device) -> PredictFn:
    """Return f(raw_features[n, 10]) -> mortality probability[n] for one note."""

    def predict_proba(raw: np.ndarray) -> np.ndarray:
        raw = np.atleast_2d(np.asarray(raw, dtype=np.float64))
        scaled = scaler.transform(clip_features(raw))
        structured = torch.tensor(scaled, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model.head(text_emb, structured).squeeze(-1)
        return torch.sigmoid(logits).cpu().numpy().astype(np.float64)

    return predict_proba


def format_value(name: str, value: float) -> str:
    if name == "gender":
        return "female" if round(value) == 1 else "male"
    if name in ("creatinine", "hemoglobin"):
        return f"{value:.1f}"
    return f"{value:.0f}"


def feature_display_labels(raw: np.ndarray) -> list[str]:
    return [f"{FEATURE_LABELS[n]} = {format_value(n, v)}" for n, v in zip(FEATURE_NAMES, raw)]


# -----------------------------
# SHAP
# -----------------------------
class ShapExplainer:
    """Kernel SHAP over the 10 structured features (text held fixed)."""

    def __init__(self, background: np.ndarray, n_clusters: int = 20):
        self.background = shap.kmeans(background, n_clusters)

    def explain(self, predict_fn: PredictFn, raw: np.ndarray) -> dict:
        explainer = shap.KernelExplainer(predict_fn, self.background)
        # 2^10 = 1024 coalitions: with nsamples >= 1022 Kernel SHAP enumerates
        # them all, so the result is exact and deterministic (the old
        # nsamples=20 produced noisy values that changed on every request).
        values = np.asarray(
            explainer.shap_values(raw.reshape(1, -1), nsamples=SHAP_NSAMPLES, silent=True)
        ).reshape(-1)
        base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

        plot = contribution_chart(
            feature_display_labels(raw),
            values.tolist(),
            title="SHAP: what moved this patient's risk",
            xlabel="Change in predicted mortality probability",
        )
        return {
            "base_value": base_value,
            "values": {name: float(v) for name, v in zip(FEATURE_NAMES, values)},
            "plot": plot,
        }


def top_features(raw: np.ndarray, shap_values: dict[str, float], k: int = 5) -> list[dict]:
    """The k features with the largest absolute SHAP contribution, largest first.

    The old version ranked features by how far the input was from the training
    mean (|z-score|) without asking the model at all, which put gender in the
    top 5 for almost every patient, and listed them in ascending order.
    """
    ranked = sorted(FEATURES, key=lambda f: abs(shap_values[f.name]), reverse=True)[:k]
    index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    return [
        {
            "name": f.name,
            "label": f.label,
            "value": float(raw[index[f.name]]),
            "contribution": float(shap_values[f.name]),
            "direction": "increases risk" if shap_values[f.name] > 0 else "decreases risk",
        }
        for f in ranked
    ]
