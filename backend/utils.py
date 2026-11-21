"""Input cleaning and validation helpers."""

from __future__ import annotations

import math
import re

import numpy as np

from .config import FEATURES, LOWERCASE_TEXT, N_FEATURES

_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str | None, lowercase: bool = LOWERCASE_TEXT) -> str:
    """Normalise whitespace (and optionally case) of a clinical note."""
    if not text:
        return ""
    text = _WHITESPACE.sub(" ", text).strip()
    return text.lower() if lowercase else text


def validate_features(features) -> list[str]:
    """Return a list of human-readable problems (empty list = valid).

    Checks the count, that every value is a finite number, that gender is 0/1
    and that each value lies in a physiologically possible range.
    """
    if not isinstance(features, (list, tuple)):
        return ["features must be a list of numbers"]
    if len(features) != N_FEATURES:
        return [f"expected {N_FEATURES} features, got {len(features)}"]

    errors: list[str] = []
    for feat, value in zip(FEATURES, features):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{feat.label} must be a number")
            continue
        if not math.isfinite(value):
            errors.append(f"{feat.label} must be a finite number")
            continue
        if feat.name == "gender" and value not in (0, 1):
            errors.append("Gender must be 0 (male) or 1 (female)")
            continue
        if not feat.min <= value <= feat.max:
            errors.append(
                f"{feat.label} must be between {feat.min:g} and {feat.max:g} {feat.unit}"
            )

    if not errors:
        sbp, dbp = features[3], features[4]
        if dbp > sbp:
            errors.append("Diastolic BP cannot be higher than systolic BP")
    return errors


def clip_features(features) -> np.ndarray:
    """Clip values into the plausible clinical range used during training."""
    x = np.asarray(features, dtype=np.float64)
    lo = np.array([f.clip_min for f in FEATURES])
    hi = np.array([f.clip_max for f in FEATURES])
    return np.clip(x, lo, hi)
