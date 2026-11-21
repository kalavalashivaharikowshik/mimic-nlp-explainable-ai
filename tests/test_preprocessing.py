import json
import math

import numpy as np
import pytest

from backend.background import make_background
from backend.config import FEATURE_NAMES, FEATURES, SCALER_PATH
from backend.scaler import JsonStandardScaler
from backend.utils import clean_text, clip_features, validate_features
from tests.conftest import CRITICAL, STABLE


# ---------- scaler ----------
def test_scaler_json_matches_feature_order():
    data = json.loads(SCALER_PATH.read_text())
    assert data["feature_names"] == FEATURE_NAMES
    assert len(data["mean"]) == len(data["scale"]) == 10


def test_scaler_transform_and_inverse_roundtrip():
    s = JsonStandardScaler.load(SCALER_PATH)
    x = np.array([STABLE, CRITICAL], dtype=float)
    np.testing.assert_allclose(s.inverse_transform(s.transform(x)), x, rtol=1e-12)
    # Values extracted from the original sklearn scaler.pkl
    assert s.mean_[0] == pytest.approx(63.0396049)
    assert s.scale_[0] == pytest.approx(16.7165164)


def test_scaler_rejects_wrong_feature_count():
    s = JsonStandardScaler.load(SCALER_PATH)
    with pytest.raises(ValueError):
        s.transform([1, 2, 3])


def test_scaler_rejects_mismatched_feature_order(tmp_path):
    data = json.loads(SCALER_PATH.read_text())
    data["feature_names"] = list(reversed(data["feature_names"]))
    p = tmp_path / "s.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="does not match"):
        JsonStandardScaler.load(p)


def test_scaler_fit_and_save(tmp_path):
    x = np.random.default_rng(0).normal(size=(50, 10))
    x[:, 3] = 5.0  # constant column must not produce division by zero
    s = JsonStandardScaler.fit(x)
    p = tmp_path / "s.json"
    s.save(p)
    s2 = JsonStandardScaler.load(p)
    assert np.all(np.isfinite(s2.transform(x)))


# ---------- validation ----------
@pytest.mark.parametrize("features", [STABLE, CRITICAL])
def test_valid_features(features):
    assert validate_features(features) == []


@pytest.mark.parametrize(
    "features, expected",
    [
        ([1, 2, 3], "expected 10 features"),
        ("abc", "must be a list"),
        ([45, 2, 76, 122, 78, 93, 98, 0.9, 104, 14], "Gender must be 0"),
        ([45, 0, 76, 122, 78, 93, 180, 0.9, 104, 14], "SpO2 must be between 0 and 100"),
        ([-5, 0, 76, 122, 78, 93, 98, 0.9, 104, 14], "Age must be between"),
        ([45, 0, 76, 122, 78, 93, 98, math.nan, 104, 14], "finite"),
        ([45, 0, 76, 122, 78, 93, 98, math.inf, 104, 14], "finite"),
        ([45, 0, 76, 90, 120, 93, 98, 0.9, 104, 14], "Diastolic BP cannot be higher"),
        ([45, 0, "x", 122, 78, 93, 98, 0.9, 104, 14], "must be a number"),
        ([45, True, 76, 122, 78, 93, 98, 0.9, 104, 14], "must be a number"),
    ],
)
def test_invalid_features(features, expected):
    errors = validate_features(features)
    assert errors and any(expected in e for e in errors), errors


def test_clean_text_normalises_whitespace_and_case():
    assert clean_text("  Patient \n\n STABLE\t ok ") == "patient stable ok"
    assert clean_text("Patient STABLE", lowercase=False) == "Patient STABLE"
    assert clean_text(None) == ""


def test_clip_features_into_training_range():
    clipped = clip_features([10, 0, 400, 122, 78, 93, 30, 0.0, 5, 30])
    assert clipped[0] == 18 and clipped[2] == 250 and clipped[6] == 50
    assert clipped[7] == pytest.approx(0.1) and clipped[8] == 20 and clipped[9] == 22


# ---------- synthetic background ----------
def test_background_is_deterministic_and_plausible():
    a, b = make_background(), make_background()
    np.testing.assert_array_equal(a, b)
    assert a.shape == (200, 10)
    for i, f in enumerate(FEATURES):
        assert a[:, i].min() >= f.clip_min and a[:, i].max() <= f.clip_max
    assert set(np.unique(a[:, 1])) <= {0.0, 1.0}
    assert a[:, 6].max() <= 100  # SpO2 can never exceed 100 %
