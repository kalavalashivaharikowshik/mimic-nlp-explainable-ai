import base64

import pytest
import torch
from fastapi.testclient import TestClient

from backend import config
from backend.load_model import ensure_weights, is_valid_weights_file, load_state_dict_compat
from backend.main import app
from backend.model import HybridModel
from backend.predict import InvalidInputError
from tests.conftest import CRITICAL, STABLE


@pytest.fixture()
def client(predictor):
    app.state.model.set_predictor(predictor)
    with TestClient(app) as c:
        yield c


# ---------- engine ----------
def test_predictor_response_shape(predictor):
    r = predictor.predict("Septic shock on norepinephrine, intubated.", CRITICAL)
    assert r["prediction"] in {"HIGH RISK", "LOW RISK"}
    assert 0.0 <= r["probability"] <= 1.0
    assert r["threshold"] == config.THRESHOLD
    assert (r["prediction"] == "HIGH RISK") == (r["probability"] >= config.THRESHOLD)
    assert len(r["explanation"]["top_features"]) == 5
    assert r["shap"]["base_value"] + sum(r["shap"]["values"].values()) == pytest.approx(
        r["probability"], abs=1e-5
    )
    assert any(f["source"] == "clinical note" for f in r["clinical_flags"])
    assert r["disclaimer"]


def test_predictor_is_deterministic(predictor):
    a = predictor.predict("patient stable", STABLE)
    b = predictor.predict("patient stable", STABLE)
    assert a["probability"] == b["probability"]
    assert a["lime"]["weights"] == b["lime"]["weights"]


def test_predictor_rejects_invalid(predictor):
    with pytest.raises(InvalidInputError) as exc:
        predictor.predict("x", [45, 3, 76, 122, 78, 93, 98, 0.9, 104, 14])
    assert "Gender" in exc.value.errors[0]


def test_short_note_message(predictor):
    assert "very short" in predictor.predict("", STABLE)["note_message"]


# ---------- HTTP API ----------
def test_health_and_metadata(client):
    assert client.get("/health").json()["model"] == "ready"
    meta = client.get("/api/metadata").json()
    assert [f["name"] for f in meta["features"]] == config.FEATURE_NAMES
    assert meta["threshold"] == config.THRESHOLD


def test_predict_success(client):
    res = client.post("/predict", json={"text": "septic shock, intubated", "features": CRITICAL})
    assert res.status_code == 200
    body = res.json()
    assert body["prediction"] in {"HIGH RISK", "LOW RISK"}
    assert base64.b64decode(body["shap"]["plot"])[:4] == b"\x89PNG"
    assert base64.b64decode(body["lime"]["plot"])[:4] == b"\x89PNG"


@pytest.mark.parametrize(
    "payload, fragment",
    [
        ({"text": "x", "features": [1, 2]}, "at least 10"),
        ({"text": "x", "features": ["a"] * 10}, "valid number"),
        ({"features": STABLE}, "text"),
        ({"text": "x", "features": [45, 0, 76, 122, 78, 93, 150, 0.9, 104, 14]}, "SpO2"),
        ({"text": "x" * (config.MAX_TEXT_CHARS + 1), "features": STABLE}, "at most"),
    ],
)
def test_predict_validation_errors_return_422(client, payload, fragment):
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    detail = res.json()["detail"]
    assert isinstance(detail, list) and any(fragment in m for m in detail), detail


def test_predict_returns_503_when_model_not_ready(predictor):
    state = app.state.model
    state.status, state.predictor = "loading", None
    try:
        with TestClient(app) as c:
            res = c.post("/predict", json={"text": "x", "features": STABLE})
            assert res.status_code == 503
            assert "loading" in res.json()["detail"][0]
    finally:
        state.set_predictor(predictor)


def test_frontend_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "ICU Mortality Prediction" in page.text
    assert client.get("/frontend/script.js").status_code == 200
    assert client.get("/frontend/style.css").status_code == 200
    assert page.headers["x-content-type-options"] == "nosniff"


def test_no_wildcard_cors_by_default(client):
    res = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in res.headers


# ---------- model loading ----------
def test_small_or_missing_weights_are_invalid(tmp_path):
    assert not is_valid_weights_file(tmp_path / "missing.pth")
    tiny = tmp_path / "html.pth"
    tiny.write_text("<html>Google Drive quota exceeded</html>")
    assert not is_valid_weights_file(tiny)


def test_failed_download_leaves_no_broken_file(tmp_path, monkeypatch):
    import backend.load_model as lm

    def fake_download(path):
        path.write_text("<html>quota</html>")

    monkeypatch.setattr(lm, "_download_to", fake_download)
    target = tmp_path / "models" / "hybrid_model.pth"
    with pytest.raises(RuntimeError, match="invalid"):
        ensure_weights(target)
    assert not target.exists()
    assert list(target.parent.iterdir()) == []


def test_valid_download_is_moved_into_place(tmp_path, monkeypatch):
    import backend.load_model as lm

    monkeypatch.setattr(lm.config, "MIN_MODEL_BYTES", 10)
    monkeypatch.setattr(lm, "_download_to", lambda p: p.write_bytes(b"x" * 100))
    target = tmp_path / "hybrid_model.pth"
    assert ensure_weights(target) == target and target.stat().st_size == 100


def test_state_dict_compat(artifacts):
    cfg = artifacts.model.text_model.config
    sd = {f"module.{k}": v for k, v in artifacts.model.state_dict().items()}
    sd["module.text_model.embeddings.position_ids"] = torch.arange(512)[None]
    fresh = HybridModel(text_config=cfg)
    load_state_dict_compat(fresh, sd)
    for (k, v), v2 in zip(
        artifacts.model.state_dict().items(), fresh.state_dict().values(), strict=False
    ):
        assert torch.equal(v, v2), k

    broken = dict(artifacts.model.state_dict())
    broken.pop("classifier.0.weight")
    with pytest.raises(RuntimeError, match="do not match"):
        load_state_dict_compat(HybridModel(text_config=cfg), broken)
