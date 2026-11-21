import base64

import numpy as np
import pytest
import torch

from backend.background import make_background
from backend.clinical_flags import clinical_flags, note_flags, vital_flags
from backend.explain import LimeExplainer, ShapExplainer, make_predict_fn, top_features
from backend.nlp import key_terms
from tests.conftest import CRITICAL, STABLE

PNG_MAGIC = b"\x89PNG"


def toy_predict(x):
    """Known function: risk rises with age and creatinine, falls with SpO2."""
    x = np.atleast_2d(x)
    z = 0.04 * (x[:, 0] - 60) + 0.8 * (x[:, 7] - 1) - 0.1 * (x[:, 6] - 95) - 1.5
    return 1 / (1 + np.exp(-z))


@pytest.fixture(scope="module")
def background():
    return make_background()


# ---------- SHAP ----------
def test_shap_is_additive_and_deterministic(background):
    raw = np.array(CRITICAL, dtype=float)
    shap_exp = ShapExplainer(background)
    r1, r2 = shap_exp.explain(toy_predict, raw), shap_exp.explain(toy_predict, raw)
    assert r1["values"] == r2["values"]
    total = r1["base_value"] + sum(r1["values"].values())
    assert total == pytest.approx(float(toy_predict(raw)[0]), abs=1e-6)
    # Features the toy model ignores get exactly zero credit
    assert r1["values"]["heart_rate"] == pytest.approx(0, abs=1e-9)
    assert base64.b64decode(r1["plot"])[:4] == PNG_MAGIC


def test_top_features_ranked_by_absolute_shap(background):
    raw = np.array(CRITICAL, dtype=float)
    values = ShapExplainer(background).explain(toy_predict, raw)["values"]
    top = top_features(raw, values, k=3)
    assert [t["name"] for t in top] == sorted(values, key=lambda n: abs(values[n]), reverse=True)[:3]
    assert {t["name"] for t in top} == {"age", "creatinine", "spo2"}
    assert all(t["direction"] == "increases risk" for t in top)


def test_top_features_no_effect_label():
    raw = np.array(STABLE, dtype=float)
    values = dict.fromkeys(["age", "gender", "heart_rate", "sbp", "dbp", "mbp", "spo2", "creatinine", "glucose", "hemoglobin"], 0.0)
    assert {t["direction"] for t in top_features(raw, values)} == {"no effect"}


# ---------- LIME ----------
def test_lime_uses_raw_units_and_is_deterministic(background):
    raw = np.array(CRITICAL, dtype=float)
    lime = LimeExplainer(background)
    r1, r2 = lime.explain(toy_predict, raw), lime.explain(toy_predict, raw)
    assert r1["weights"] == r2["weights"]
    rules = " ".join(r1["weights"])
    assert "Creatinine" in rules and "SpO2" in rules
    # Raw clinical thresholds, not z-scores: some rule threshold must be > 10
    numbers = [float(t) for t in rules.replace("<", " ").replace(">", " ").replace("=", " ").split() if t.replace(".", "", 1).isdigit()]
    assert any(n > 10 for n in numbers)
    assert base64.b64decode(r1["plot"])[:4] == PNG_MAGIC


# ---------- prediction function ----------
def test_predict_fn_matches_full_forward(artifacts):
    tok, model, scaler = artifacts.tokenizer, artifacts.model, artifacts.scaler
    enc = tok("patient in septic shock", return_tensors="pt")
    with torch.no_grad():
        emb, _ = model.encode_text(enc["input_ids"], enc["attention_mask"])
        expected = torch.sigmoid(
            model(enc["input_ids"], enc["attention_mask"], torch.tensor(scaler.transform([STABLE]), dtype=torch.float32))
        ).item()
    fn = make_predict_fn(model, scaler, emb, torch.device("cpu"))
    assert fn(np.array(STABLE))[0] == pytest.approx(expected, rel=1e-5)
    assert fn(np.array([STABLE, CRITICAL])).shape == (2,)


# ---------- attention key terms ----------
def test_key_terms_filters_stop_words_and_groups_subwords(artifacts):
    tok = artifacts.tokenizer
    text = "patient within normal range, intubated for sepsis. bp low"
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
    n = enc["input_ids"].shape[1]
    att = torch.rand(1, 2, n, n)
    terms = key_terms(text, (att,), enc.word_ids(0), offsets)
    words = [t["word"] for t in terms]
    assert "intubated" in words  # rebuilt from 'intubat' + '##ed' style pieces
    assert "bp" in words  # clinical abbreviation kept
    assert not {"patient", "within", "normal", "range", "for"} & set(words)
    assert terms[0]["score"] == 1.0
    assert key_terms("", None, [], []) == []


# ---------- clinical flags ----------
def test_no_false_flags_for_normal_patient():
    assert vital_flags(np.array(STABLE, dtype=float)) == []
    assert note_flags("patient stable, no sepsis, denies chest pain") == []


def test_vital_flags_use_real_thresholds():
    flags = " ".join(vital_flags(np.array(CRITICAL, dtype=float)))
    for expected in ("Tachycardia", "Hypotension", "Low oxygen", "creatinine", "Hyperglycaemia"):
        assert expected in flags


@pytest.mark.parametrize(
    "text, sepsis, ventilation",
    [
        ("septic shock on norepinephrine", True, False),
        ("no evidence of sepsis", False, False),
        ("sepsis ruled out", False, False),
        ("denies sepsis. intubated overnight", False, True),
        ("not intubated, sepsis likely", True, False),
    ],
)
def test_note_flag_negation(text, sepsis, ventilation):
    msgs = note_flags(text)
    assert ("Sepsis mentioned in note" in msgs) is sepsis
    assert ("Mechanical ventilation mentioned" in msgs) is ventilation


def test_clinical_flags_have_sources():
    flags = clinical_flags(np.array(CRITICAL, dtype=float), "septic shock")
    assert {f["source"] for f in flags} == {"vitals/labs", "clinical note"}
