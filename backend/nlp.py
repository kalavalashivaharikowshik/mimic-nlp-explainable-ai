"""Attention-based key terms from the clinical note.

Attention weights show where the model "looked"; they are a useful hint but
not a faithful causal explanation (Jain & Wallace, 2019). The UI labels them
accordingly.

Improvements over the old version:
- the attention maps come from the same forward pass and the same max length
  as the prediction (no second BioBERT call, no 128 vs 256 mismatch)
- sub-word pieces are grouped with the tokenizer's word ids and the original
  text span is shown, instead of re-gluing '##' pieces by hand
- pieces are combined with max() instead of sum(), so long words are not
  favoured just for having more pieces
- common English / charting stop words ('patient', 'within', 'range', ...)
  are removed, while short clinical abbreviations (BP, HR, ICU, ...) are kept
"""

from __future__ import annotations

import numpy as np

STOP_WORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers him his how i if in into
    is it its itself just me more most my no nor not now of off on once only or other
    our out over own same she should so some such than that the their them then there
    these they this those through to too under until up very was we were what when
    where which while who whom why will with would you your yours
    patient patients pt within range normal noted note today yesterday per day days
    hour hours since given continue continued remains remain currently current also
    well likely appears seen still overall status signs sign denies without
    """.split()
)

CLINICAL_ABBREVIATIONS = frozenset(
    """
    bp hr rr o2 iv gi mi ct icu ards aki ckd copd chf cad dvt pe uti dka gcs ett ngt
    abg cvp map vt vf afib dnr cpr ekg ecg hgb wbc inr bun
    """.split()
)


def _keep(word: str) -> bool:
    w = word.lower()
    if w in CLINICAL_ABBREVIATIONS:
        return True
    return len(w) >= 3 and w.isalpha() and w not in STOP_WORDS


def key_terms(
    text: str,
    cls_attention: np.ndarray | None,
    word_ids: list[int | None],
    offsets: list[tuple[int, int]],
    top_k: int = 8,
) -> list[dict]:
    """Return up to ``top_k`` words ranked by [CLS] attention in the last layer.

    Args:
        text: the exact text that was tokenised.
        cls_attention: last-layer attention from [CLS] to every token,
            averaged over heads (an output of the ONNX encoder).
        word_ids: word index of every token (None for special tokens).
        offsets: character offsets of every token in ``text``.
    """
    if cls_attention is None or not text:
        return []
    cls_attention = np.asarray(cls_attention, dtype=np.float64).reshape(-1)

    words: dict[int, dict] = {}
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        start, end = offsets[idx]
        score = float(cls_attention[idx])
        entry = words.setdefault(wid, {"start": start, "end": end, "score": score})
        entry["start"] = min(entry["start"], start)
        entry["end"] = max(entry["end"], end)
        entry["score"] = max(entry["score"], score)

    best: dict[str, float] = {}
    for entry in words.values():
        word = text[entry["start"] : entry["end"]]
        if not _keep(word):
            continue
        key = word.lower()
        best[key] = max(best.get(key, 0.0), entry["score"])

    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    if not ranked:
        return []
    top = ranked[0][1] or 1.0
    return [{"word": w, "score": round(s / top, 4)} for w, s in ranked]
