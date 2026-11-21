"""SHAP and LIME for the structured (vitals / labs) features, in pure numpy.

No ``shap`` / ``lime`` / ``scikit-learn`` dependency is needed at runtime, which
keeps the server small enough for free hosting (512 MB RAM).

* SHAP: with only 10 features, the exact (interventional) Shapley values can
  be computed by evaluating all 2^10 = 1,024 feature coalitions against a
  reference cohort. Values add up exactly to ``prediction - base_value``.
* LIME: the standard tabular recipe - quartile discretisation, perturbations
  drawn from the reference cohort, exponential kernel, weighted ridge
  regression - with a fixed seed so results are reproducible.

The clinical note is held fixed: its BioBERT embedding is computed once per
request and only the tiny fusion head is re-evaluated for every sample.
Explanations work in raw clinical units (mmHg, mg/dL, ...).
"""

from __future__ import annotations

from collections.abc import Callable
from math import factorial

import numpy as np

from .config import FEATURE_LABELS, FEATURE_NAMES, FEATURES, LIME_NUM_SAMPLES, RANDOM_SEED
from .fusion import FusionHead
from .scaler import JsonStandardScaler
from .utils import clip_features

PredictFn = Callable[[np.ndarray], np.ndarray]
N = len(FEATURE_NAMES)
GENDER_INDEX = FEATURE_NAMES.index("gender")


def make_predict_fn(
    head: FusionHead, scaler: JsonStandardScaler, text_emb: np.ndarray
) -> PredictFn:
    """Return f(raw_features[n, 10]) -> mortality probability[n] for one note."""
    text_part = head.text_part(text_emb)

    def predict_proba(raw: np.ndarray) -> np.ndarray:
        raw = np.atleast_2d(np.asarray(raw, dtype=np.float64))
        return head.proba(text_part, scaler.transform(clip_features(raw)))

    return predict_proba


def format_value(name: str, value: float) -> str:
    if name == "gender":
        return "female" if round(value) == 1 else "male"
    if name in ("creatinine", "hemoglobin"):
        return f"{value:.1f}"
    return f"{value:.0f}"


# -----------------------------
# SHAP (exact)
# -----------------------------
class ShapExplainer:
    def __init__(self, background: np.ndarray, n_reference: int = 50):
        self.reference = np.asarray(background, dtype=np.float64)[:n_reference]
        self.codes = np.arange(2**N)
        self.masks = ((self.codes[:, None] >> np.arange(N)) & 1).astype(bool)  # (1024, 10)
        self.sizes = self.masks.sum(axis=1)
        # Shapley weight |S|! (N - |S| - 1)! / N! for coalitions S not containing i
        self.weights = np.array(
            [
                factorial(s) * factorial(N - s - 1) / factorial(N) if s < N else 0.0
                for s in range(N + 1)
            ]
        )

    def explain(self, predict_fn: PredictFn, raw: np.ndarray) -> dict:
        raw = np.asarray(raw, dtype=np.float64)
        ref = self.reference
        # rows[c, r] = patient values where coalition c is "on", reference row r elsewhere
        rows = np.where(self.masks[:, None, :], raw[None, None, :], ref[None, :, :])
        value = predict_fn(rows.reshape(-1, N)).reshape(len(self.masks), len(ref)).mean(axis=1)

        phi = np.zeros(N)
        for i in range(N):
            s = self.codes[~self.masks[:, i]]
            phi[i] = np.sum(self.weights[self.sizes[s]] * (value[s | (1 << i)] - value[s]))

        return {
            "base_value": float(value[0]),  # expected prediction over the reference cohort
            "values": {name: float(v) for name, v in zip(FEATURE_NAMES, phi, strict=True)},
        }


# -----------------------------
# LIME
# -----------------------------
class LimeExplainer:
    def __init__(
        self, background: np.ndarray, num_samples: int = LIME_NUM_SAMPLES, seed: int = RANDOM_SEED
    ):
        self.background = np.asarray(background, dtype=np.float64)
        self.num_samples = num_samples
        self.seed = seed
        self.edges = [np.percentile(self.background[:, j], [25, 50, 75]) for j in range(N)]

    def _bins(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        out = np.empty(x.shape, dtype=int)
        for j in range(N):
            if j == GENDER_INDEX:
                out[:, j] = np.round(x[:, j]).astype(int)
            else:
                out[:, j] = np.searchsorted(self.edges[j], x[:, j], side="left")
        return out

    def _rule(self, j: int, b: int) -> str:
        label = FEATURE_LABELS[FEATURE_NAMES[j]]
        if j == GENDER_INDEX:
            return f"{label} = {'female' if b == 1 else 'male'}"
        q = self.edges[j]
        if b == 0:
            return f"{label} <= {q[0]:.2f}"
        if b == len(q):
            return f"{label} > {q[-1]:.2f}"
        return f"{q[b - 1]:.2f} < {label} <= {q[b]:.2f}"

    def explain(self, predict_fn: PredictFn, raw: np.ndarray) -> dict:
        raw = np.asarray(raw, dtype=np.float64)
        rng = np.random.default_rng(self.seed)  # fresh generator => reproducible per request

        n = self.num_samples
        picks = rng.integers(0, len(self.background), size=(n, N))
        samples = self.background[picks, np.arange(N)]
        samples[0] = raw

        instance_bins = self._bins(raw)[0]
        z = (self._bins(samples) == instance_bins).astype(np.float64)
        y = predict_fn(samples)

        distance = np.sqrt(((1.0 - z) ** 2).sum(axis=1))
        width = 0.75 * np.sqrt(N)
        w = np.sqrt(np.exp(-(distance**2) / width**2))

        # Weighted ridge regression (alpha = 1) with intercept
        wsum = w.sum()
        z_mean = (w[:, None] * z).sum(axis=0) / wsum
        y_mean = (w * y).sum() / wsum
        zc, yc = z - z_mean, y - y_mean
        coef = np.linalg.solve((zc * w[:, None]).T @ zc + np.eye(N), (zc * w[:, None]).T @ yc)

        ss_res = np.sum(w * (yc - zc @ coef) ** 2)
        ss_tot = np.sum(w * yc**2) or 1.0

        order = np.argsort(-np.abs(coef))
        weights = {self._rule(j, int(instance_bins[j])): float(coef[j]) for j in order}
        return {"weights": weights, "fit_r2": float(1 - ss_res / ss_tot)}


def _direction(value: float, eps: float = 1e-6) -> str:
    if value > eps:
        return "increases risk"
    if value < -eps:
        return "decreases risk"
    return "no effect"


def top_features(raw: np.ndarray, shap_values: dict[str, float], k: int = 5) -> list[dict]:
    """The k features with the largest absolute SHAP contribution, largest first."""
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
