"""The fusion head (vitals MLP + classifier) in pure numpy.

The head is tiny (~110k parameters), so running it in numpy is exact and needs
no deep-learning runtime. The classifier's first layer is split into a text
part and a vitals part: the text part is computed ONCE per note, so each of the
tens of thousands of SHAP/LIME evaluations only costs a few thousand
multiply-adds.

Weights come from ``head.npz`` written by ``tools/export_onnx.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HEAD_KEYS = (
    "struct_net.0.weight",
    "struct_net.0.bias",
    "struct_net.2.weight",
    "struct_net.2.bias",
    "classifier.0.weight",
    "classifier.0.bias",
    "classifier.3.weight",
    "classifier.3.bias",
)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class FusionHead:
    def __init__(self, weights: dict[str, np.ndarray]):
        missing = [k for k in HEAD_KEYS if k not in weights]
        if missing:
            raise ValueError(f"head weights are missing {missing}")
        w = {k: np.asarray(weights[k], dtype=np.float32) for k in HEAD_KEYS}

        self.w1, self.b1 = w["struct_net.0.weight"], w["struct_net.0.bias"]
        self.w2, self.b2 = w["struct_net.2.weight"], w["struct_net.2.bias"]
        c1 = w["classifier.0.weight"]
        self.struct_out = self.w2.shape[0]
        self.hidden = c1.shape[1] - self.struct_out
        self.c1_text = c1[:, : self.hidden]
        self.c1_struct = c1[:, self.hidden :]
        self.c1_b = w["classifier.0.bias"]
        self.c2, self.c2_b = w["classifier.3.weight"], w["classifier.3.bias"]

    @classmethod
    def load(cls, path: str | Path) -> FusionHead:
        with np.load(path) as data:
            return cls({k: data[k] for k in data.files})

    def text_part(self, text_emb: np.ndarray) -> np.ndarray:
        """Pre-compute the classifier contribution of one [CLS] embedding."""
        emb = np.asarray(text_emb, dtype=np.float32).reshape(-1)
        if emb.shape[0] != self.hidden:
            raise ValueError(f"expected a {self.hidden}-d text embedding, got {emb.shape[0]}")
        return self.c1_text @ emb + self.c1_b

    def logits(self, text_part: np.ndarray, scaled: np.ndarray) -> np.ndarray:
        """Logits for a batch of already-scaled structured feature rows."""
        x = np.atleast_2d(np.asarray(scaled, dtype=np.float32))
        s = _relu(x @ self.w1.T + self.b1) @ self.w2.T + self.b2
        h = _relu(s @ self.c1_struct.T + text_part)
        return (h @ self.c2.T + self.c2_b).reshape(-1)

    def proba(self, text_part: np.ndarray, scaled: np.ndarray, chunk: int = 8192) -> np.ndarray:
        """Probabilities, evaluated in chunks to keep peak memory low."""
        x = np.atleast_2d(scaled)
        out = np.empty(len(x), dtype=np.float64)
        for start in range(0, len(x), chunk):
            stop = start + chunk
            out[start:stop] = _sigmoid(self.logits(text_part, x[start:stop]).astype(np.float64))
        return out
