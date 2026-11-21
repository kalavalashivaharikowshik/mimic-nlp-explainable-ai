"""Synthetic reference cohort for SHAP and LIME.

SHAP and LIME need "background" patients to compare against. The old project
shipped ``background.npy`` with 300 rows of *real* MIMIC patients. Together
with the scaler that was committed alongside it, those rows could be converted
straight back to the original patient values, and redistributing MIMIC data is
not allowed under the PhysioNet credentialed data use agreement.

This module generates a deterministic, fully synthetic ICU-like cohort from
published population-level ranges instead. No patient record is involved.
"""

from __future__ import annotations

import numpy as np

from .config import FEATURES, RANDOM_SEED

N_BACKGROUND = 200


def make_background(n: int = N_BACKGROUND, seed: int = RANDOM_SEED) -> np.ndarray:
    """Return an ``(n, 10)`` array of plausible adult ICU values in raw units."""
    rng = np.random.default_rng(seed)

    age = rng.normal(64, 16, n)
    gender = rng.binomial(1, 0.44, n).astype(float)
    heart_rate = rng.normal(87, 17, n)
    sbp = rng.normal(119, 21, n)
    dbp = rng.normal(63, 13, n)
    mbp = (sbp + 2 * dbp) / 3 + rng.normal(0, 3, n)
    spo2 = 100 - rng.gamma(2.0, 1.5, n)
    creatinine = rng.lognormal(np.log(1.0), 0.55, n)
    glucose = rng.lognormal(np.log(130), 0.28, n)
    hemoglobin = rng.normal(10.8, 2.0, n)

    data = np.column_stack(
        [age, gender, heart_rate, sbp, dbp, mbp, spo2, creatinine, glucose, hemoglobin]
    )
    lo = np.array([f.clip_min for f in FEATURES])
    hi = np.array([f.clip_max for f in FEATURES])
    return np.clip(data, lo, hi)
