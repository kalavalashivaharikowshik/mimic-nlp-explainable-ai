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

from collections.abc import Callable

import numpy as np
import shap
import torch
from lime.lime_tabular import LimeTabularExplainer

from .config import (
    FEATURE_LABELS,
    FEATURE_NAMES,
    FEATURES,
    LIME_NUM_SAMPLES,
    RANDOM_SEED,
    SHAP_NSAMPLES,
)
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
    return [
        f"{FEATURE_LABELS[n]} = {format_value(n, v)}"
        for n, v in zip(FEATURE_NAMES, raw, strict=False)
    ]


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
            "values": {name: float(v) for name, v in zip(FEATURE_NAMES, values, strict=False)},
            "plot": plot,
        }


# -----------------------------
# LIME
# -----------------------------
GENDER_INDEX = FEATURE_NAMES.index("gender")


class LimeExplainer:
    """LIME on raw clinical units, deterministic for a given input."""

    def __init__(self, background: np.ndarray):
        self.background = background

    def explain(self, predict_fn: PredictFn, raw: np.ndarray) -> dict:
        # A fresh explainer with a fixed seed per request makes the result
        # reproducible (a shared RandomState would drift between requests).
        explainer = LimeTabularExplainer(
            self.background,
            feature_names=[FEATURE_LABELS[n] for n in FEATURE_NAMES],
            class_names=["Survives", "Dies"],
            categorical_features=[GENDER_INDEX],
            categorical_names={GENDER_INDEX: ["male", "female"]},
            discretize_continuous=True,
            mode="classification",
            random_state=RANDOM_SEED,
        )

        def classifier_fn(x: np.ndarray) -> np.ndarray:
            p = predict_fn(x)
            return np.column_stack([1 - p, p])

        exp = explainer.explain_instance(
            raw,
            classifier_fn,
            labels=(1,),
            num_features=len(FEATURE_NAMES),
            num_samples=LIME_NUM_SAMPLES,
        )
        pairs = exp.as_list(label=1)
        weights = {rule: float(w) for rule, w in pairs}
        top = pairs[:6]
        plot = contribution_chart(
            [rule for rule, _ in top],
            [w for _, w in top],
            title="LIME: local rules around this patient",
            xlabel="Weight towards mortality",
        )
        return {"weights": weights, "plot": plot}


def _direction(value: float, eps: float = 1e-6) -> str:
    if value > eps:
        return "increases risk"
    if value < -eps:
        return "decreases risk"
    return "no effect"


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
            "direction": _direction(shap_values[f.name]),
        }
        for f in ranked
    ]
