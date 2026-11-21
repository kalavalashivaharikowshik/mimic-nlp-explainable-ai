"""Prediction engine: one request -> prediction + all explanations.

Runs entirely on ONNX Runtime + numpy: one BioBERT (ONNX) pass per request,
then the numpy fusion head for the prediction, exact SHAP and LIME.
"""

from __future__ import annotations

import threading

import numpy as np

from . import config
from .artifacts import OnnxArtifacts
from .background import make_background
from .clinical_flags import clinical_flags
from .explain import LimeExplainer, ShapExplainer, make_predict_fn, top_features
from .nlp import key_terms
from .utils import clean_text, clip_features, validate_features


class InvalidInputError(ValueError):
    """Raised for user input problems (mapped to HTTP 422)."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


class MortalityPredictor:
    def __init__(self, artifacts: OnnxArtifacts, background: np.ndarray | None = None):
        self.artifacts = artifacts
        self.session = artifacts.session
        self.tokenizer = artifacts.tokenizer
        self.head = artifacts.head
        self.scaler = artifacts.scaler

        background = make_background() if background is None else background
        self.shap = ShapExplainer(background)
        self.lime = LimeExplainer(background)

        # One request at a time keeps memory bounded on small (free-tier) hosts.
        self._lock = threading.Lock()

    def encode(self, text: str):
        """Run the ONNX encoder once: [CLS] embedding, [CLS] attention, token alignment."""
        enc = self.tokenizer.encode(text)
        feed = {
            "input_ids": np.asarray([enc.ids], dtype=np.int64),
            "attention_mask": np.asarray([enc.attention_mask], dtype=np.int64),
        }
        cls_embedding, cls_attention = self.session.run(None, feed)
        return cls_embedding[0], cls_attention[0], enc.word_ids, enc.offsets

    def predict(self, text: str, features: list[float]) -> dict:
        errors = validate_features(features)
        if errors:
            raise InvalidInputError(errors)

        text = clean_text(text, lowercase=self.artifacts.lowercase)
        raw = clip_features(features)

        with self._lock:
            text_emb, cls_attention, word_ids, offsets = self.encode(text)
            predict_fn = make_predict_fn(self.head, self.scaler, text_emb)

            prob = float(predict_fn(raw)[0])
            shap_result = self.shap.explain(predict_fn, raw)
            lime_result = self.lime.explain(predict_fn, raw)

        terms = key_terms(text, cls_attention, word_ids, offsets)

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
            "borderline": abs(prob - config.THRESHOLD) < config.BORDERLINE_MARGIN,
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
