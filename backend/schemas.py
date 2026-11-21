"""Request / response models for the API (also drive the /docs page)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .config import MAX_TEXT_CHARS, N_FEATURES


class PatientInput(BaseModel):
    text: str = Field(
        ...,
        max_length=MAX_TEXT_CHARS,
        description="Free-text clinical note.",
        examples=["Patient has sepsis and low blood pressure, started on vasopressors."],
    )
    features: list[float] = Field(
        ...,
        min_length=N_FEATURES,
        max_length=N_FEATURES,
        description=(
            "[age, gender (0=male, 1=female), heart_rate, sbp, dbp, mbp, spo2, "
            "creatinine, glucose, hemoglobin]"
        ),
        examples=[[65, 1, 110, 85, 50, 62, 89, 2.4, 210, 9.1]],
    )


class FeatureContribution(BaseModel):
    name: str
    label: str
    value: float
    contribution: float = Field(description="SHAP value (change in mortality probability).")
    direction: str = Field(description="'increases risk', 'decreases risk' or 'no effect'")


class KeyTerm(BaseModel):
    word: str
    score: float


class ClinicalFlag(BaseModel):
    source: str = Field(description="'vitals/labs' or 'clinical note'")
    message: str


class Explanation(BaseModel):
    important_words: list[str]
    key_terms: list[KeyTerm]
    top_features: list[FeatureContribution]


class ShapResult(BaseModel):
    base_value: float = Field(description="Average prediction over the reference cohort.")
    values: dict[str, float] = Field(
        description="Exact SHAP value per feature; base_value + sum(values) = probability."
    )


class LimeResult(BaseModel):
    weights: dict[str, float] = Field(description="Local rule -> weight towards mortality.")
    fit_r2: float = Field(description="How well the local linear model fits (0-1).")


class PredictionResponse(BaseModel):
    prediction: str
    probability: float = Field(description="Predicted probability of in-hospital mortality (0-1).")
    threshold: float
    borderline: bool = Field(
        description="True when the probability is within BORDERLINE_MARGIN of the threshold."
    )
    explanation: Explanation
    clinical_flags: list[ClinicalFlag]
    note_message: str
    shap: ShapResult
    lime: LimeResult
    disclaimer: str
