import json

import numpy as np
import pandas as pd
import pytest

from backend.background import make_background
from backend.config import FEATURE_NAMES
from backend.scaler import JsonStandardScaler
from tests.fakes import make_artifacts
from training.preprocess import clean_structured, prepare
from training.train import best_f1_threshold, run


def synthetic_cohort(n=240, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(make_background(n, seed), columns=FEATURE_NAMES)
    risk = 0.05 * (df["age"] - 60) + 1.2 * (df["creatinine"] - 1) - 0.3 * (df["spo2"] - 96)
    df["label"] = (risk + rng.normal(0, 1, n) > 2).astype(int)
    df["text"] = np.where(
        df["label"] == 1, "septic shock on norepinephrine intubated", "patient stable"
    )
    return df


def test_clean_structured_drops_impossible_values():
    df = pd.DataFrame([[60, 0, 999999, 120, 70, 86, 102.7, 1.0, 100, 13]], columns=FEATURE_NAMES)
    out = clean_structured(df)
    assert np.isnan(out.loc[0, "heart_rate"])  # error code -> missing, not clipped
    assert np.isnan(out.loc[0, "spo2"])  # SpO2 > 100 is impossible


def test_prepare_fits_scaler_on_train_only_and_stratifies():
    df = synthetic_cohort()
    df.loc[0:10, "spo2"] = 999999
    s = prepare(df)
    assert len(s.train) + len(s.val) + len(s.test) == len(df)
    assert not s.train[FEATURE_NAMES].isna().any().any()
    np.testing.assert_allclose(s.scaler.mean_, s.train[FEATURE_NAMES].mean().to_numpy(), rtol=1e-9)
    assert s.scaler.scale_[FEATURE_NAMES.index("spo2")] < 5  # outliers did not explode the scale
    assert abs(s.train.label.mean() - s.test.label.mean()) < 0.06


def test_prepare_requires_columns():
    with pytest.raises(ValueError, match="missing columns"):
        prepare(pd.DataFrame({"text": ["a"], "label": [1]}))


def test_best_f1_threshold():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8])
    t = best_f1_threshold(y, p)
    assert 0.3 <= t < 0.7


def test_training_run_end_to_end(tmp_path):
    arts = make_artifacts(tmp_path)
    report = run(
        synthetic_cohort(), arts.model, arts.tokenizer, tmp_path / "out", epochs=2, batch_size=32
    )
    out = tmp_path / "out"
    assert (out / "hybrid_model.pth").exists()
    JsonStandardScaler.load(out / "scaler.json")
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["test"]["auroc"] is not None and 0 <= saved["test"]["auroc"] <= 1
    assert report["validation"]["threshold"] == saved["test"]["threshold"]
