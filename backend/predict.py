"""Prediction engine: one request -> prediction + all explanations."""

from __future__ import annotations

import threading

import numpy as np
import torch

from . import config
from .background import make_background
from .clinical_flags import clinical_flags
from .explain import LimeExplainer, ShapExplainer, make_predict_fn, top_features
from .load_model import Artifacts
from .nlp import key_terms
from .utils import clean_text, clip_features, validate_features


class InvalidInputError(ValueError):
    """Raised for user input problems (mapped to HTTP 422)."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


class MortalityPredictor:
    def __init__(self, artifacts: Artifacts, background: np.ndarray | None = None):
        self.model = artifacts.model
        self.tokenizer = artifacts.tokenizer
        self.scaler = artifacts.scaler
        self.device = artifacts.device

        background = make_background() if background is None else background
        self.shap = ShapExplainer(background)
        self.lime = LimeExplainer(background)

        # One request at a time keeps memory bounded on small (free-tier) hosts.
        self._lock = threading.Lock()

    def _encode(self, text: str):
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=config.MAX_LENGTH,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        word_ids = enc.word_ids(0)
        offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            text_emb, attentions = self.model.encode_text(
                input_ids, attention_mask, output_attentions=True
            )
        return text_emb, attentions, word_ids, offsets

    def predict(self, text: str, features: list[float]) -> dict:
        errors = validate_features(features)
        if errors:
            raise InvalidInputError(errors)

        text = clean_text(text)
        raw = clip_features(features)

        with self._lock:
            text_emb, attentions, word_ids, offsets = self._encode(text)
            predict_fn = make_predict_fn(self.model, self.scaler, text_emb, self.device)

            prob = float(predict_fn(raw)[0])
            shap_result = self.shap.explain(predict_fn, raw)
            lime_result = self.lime.explain(predict_fn, raw)

        terms = key_terms(text, attentions, word_ids, offsets)

        if len(text.split()) < 5:
            note_message = (
                "The clinical note is very short, so the text part of the model has little "
                "information to work with."
            )
        else:
            note_message = "Prediction combines the clinical note and the vitals/labs."

        return {
            "prediction": "HIGH RISK" if prob >= config.THRESHOLD else "LOW RISK",
            "probability": prob,
            "threshold": config.THRESHOLD,
            "explanation": {
                "important_words": [t["word"] for t in terms],
                "key_terms": terms,
                "top_features": top_features(raw, shap_result["values"]),
            },
            "clinical_flags": clinical_flags(raw, text),
            "note_message": note_message,
            "shap": shap_result,
            "lime": lime_result,
            "disclaimer": config.DISCLAIMER,
        }
