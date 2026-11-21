"""Portable standard scaler.

The original project stored a pickled scikit-learn ``StandardScaler``. Pickles
are tied to the exact numpy / scikit-learn versions that wrote them (the old
file needed numpy 2 while requirements pinned numpy 1.24, so it could not be
loaded) and unpickling runs arbitrary code. The parameters are just two
vectors, so they are stored as plain JSON instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import FEATURE_NAMES


class JsonStandardScaler:
    """Minimal drop-in replacement for ``sklearn.preprocessing.StandardScaler``."""

    def __init__(self, mean, scale, feature_names=None):
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)
        self.feature_names = list(feature_names) if feature_names else list(FEATURE_NAMES)

        if self.mean_.shape != self.scale_.shape or self.mean_.ndim != 1:
            raise ValueError("mean and scale must be 1-D vectors of the same length")
        if np.any(self.scale_ <= 0):
            raise ValueError("scale values must be strictly positive")

    @property
    def n_features_in_(self) -> int:
        return int(self.mean_.shape[0])

    def transform(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {x.shape[1]}")
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return x * self.scale_ + self.mean_

    # ---- persistence ----
    @classmethod
    def load(cls, path: str | Path) -> "JsonStandardScaler":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        names = data.get("feature_names")
        if names and list(names) != list(FEATURE_NAMES):
            raise ValueError(
                f"Scaler feature order {names} does not match configured features {FEATURE_NAMES}"
            )
        return cls(data["mean"], data["scale"], names)

    def save(self, path: str | Path, source: str = "") -> None:
        payload = {
            "type": "StandardScaler",
            "source": source,
            "feature_names": self.feature_names,
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def fit(cls, x) -> "JsonStandardScaler":
        x = np.asarray(x, dtype=np.float64)
        scale = x.std(axis=0)
        scale[scale == 0] = 1.0
        return cls(x.mean(axis=0), scale)
