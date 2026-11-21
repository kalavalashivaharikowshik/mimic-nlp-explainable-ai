"""Rule-based clinical flags shown next to the model output.

These are transparent rules, NOT model output. The old ``generate_reason``
said "Kidney function abnormal" whenever creatinine appeared in a top-5 list
(about half of all patients, including those with normal creatinine) and said
"Sepsis detected" for notes such as "no sepsis". Flags are now based on
standard reference thresholds and the note is checked for negation.
"""

from __future__ import annotations

import re

import numpy as np

from .config import FEATURE_NAMES


def _v(raw: np.ndarray, name: str) -> float:
    return float(raw[FEATURE_NAMES.index(name)])


def vital_flags(raw: np.ndarray) -> list[str]:
    flags: list[str] = []
    hr, sbp, mbp = _v(raw, "heart_rate"), _v(raw, "sbp"), _v(raw, "mbp")
    spo2, creat = _v(raw, "spo2"), _v(raw, "creatinine")
    glucose, hgb, age = _v(raw, "glucose"), _v(raw, "hemoglobin"), _v(raw, "age")

    if hr > 100:
        flags.append(f"Tachycardia (heart rate {hr:.0f} bpm > 100)")
    elif hr < 50:
        flags.append(f"Bradycardia (heart rate {hr:.0f} bpm < 50)")
    if mbp < 65:
        flags.append(f"Hypotension (mean BP {mbp:.0f} mmHg < 65)")
    elif sbp < 90:
        flags.append(f"Low systolic BP ({sbp:.0f} mmHg < 90)")
    if sbp >= 180:
        flags.append(f"Severe hypertension (systolic BP {sbp:.0f} mmHg)")
    if spo2 < 92:
        flags.append(f"Low oxygen saturation (SpO2 {spo2:.0f}% < 92%)")
    if creat > 1.5:
        flags.append(f"Elevated creatinine ({creat:.1f} mg/dL > 1.5) - possible kidney dysfunction")
    if glucose > 180:
        flags.append(f"Hyperglycaemia (glucose {glucose:.0f} mg/dL > 180)")
    elif glucose < 70:
        flags.append(f"Hypoglycaemia (glucose {glucose:.0f} mg/dL < 70)")
    if hgb < 8:
        flags.append(f"Severe anaemia (haemoglobin {hgb:.1f} g/dL < 8)")
    if age >= 80:
        flags.append(f"Advanced age ({age:.0f} years)")
    return flags


# (regex, message). Patterns are matched on lower-cased text.
NOTE_PATTERNS: list[tuple[str, str]] = [
    (r"\bseptic shock\b|\bsepsis\b|\bseptic\b", "Sepsis mentioned in note"),
    (
        r"\bintubat\w*|\bmechanical(ly)? ventilat\w*|\bventilator\b|\bvent\b",
        "Mechanical ventilation mentioned",
    ),
    (
        r"\bvasopressors?\b|\bpressors?\b|\bnorepinephrine\b|\blevophed\b|\bvasopressin\b",
        "Vasopressor support mentioned",
    ),
    (r"\bcardiac arrest\b|\bcode blue\b|\bcpr\b", "Cardiac arrest / CPR mentioned"),
    (
        r"\bdnr\b|\bdni\b|\bcomfort care\b|\bcmo\b|\bhospice\b",
        "Limitation of care / comfort care mentioned",
    ),
    (r"\bhypotens\w*", "Hypotension mentioned in note"),
    (r"\bhypoxi\w*|\brespiratory failure\b|\bards\b", "Hypoxia / respiratory failure mentioned"),
    (
        r"\bacute kidney injury\b|\baki\b|\brenal failure\b|\bdialysis\b",
        "Kidney injury / dialysis mentioned",
    ),
    (r"\bhemorrhag\w*|\bhaemorrhag\w*|\bbleed\w*", "Bleeding mentioned in note"),
    (
        r"\bunresponsive\b|\baltered mental status\b|\bobtunded\b|\bcoma\b",
        "Altered consciousness mentioned",
    ),
]

NEGATION_CUES = re.compile(
    r"\b(no|not|denies|denied|without|negative for|ruled out|rules out|free of|absence of|resolved)\b"
)
# Cues that negate a term mentioned BEFORE them ("sepsis ruled out").
POST_NEGATION_CUES = re.compile(r"\b(ruled out|unlikely|resolved|negative|excluded)\b")
_SENTENCE_SPLIT = re.compile(r"[.;\n!?]")
# Negation does not carry across a clause boundary ("not intubated, sepsis likely").
_CLAUSE_BOUNDARY = re.compile(r",|:|\bbut\b|\bhowever\b|\balthough\b")


def _is_negated(sentence: str, match_start: int, match_end: int, window_words: int = 5) -> bool:
    """True if a negation cue appears within a few words before or after the match."""
    before_text = _CLAUSE_BOUNDARY.split(sentence[:match_start])[-1]
    after_text = _CLAUSE_BOUNDARY.split(sentence[match_end:])[0]
    before = before_text.split()[-window_words:]
    after = after_text.split()[:3]
    return bool(
        NEGATION_CUES.search(" ".join(before)) or POST_NEGATION_CUES.search(" ".join(after))
    )


def note_flags(text: str) -> list[str]:
    flags: list[str] = []
    sentences = _SENTENCE_SPLIT.split(text.lower())
    for pattern, message in NOTE_PATTERNS:
        regex = re.compile(pattern)
        for sentence in sentences:
            m = regex.search(sentence)
            if m and not _is_negated(sentence, m.start(), m.end()):
                flags.append(message)
                break
    return flags


def clinical_flags(raw: np.ndarray, text: str) -> list[dict]:
    return [{"source": "vitals/labs", "message": m} for m in vital_flags(raw)] + [
        {"source": "clinical note", "message": m} for m in note_flags(text)
    ]
