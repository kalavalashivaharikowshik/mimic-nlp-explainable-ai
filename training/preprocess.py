"""Data preparation shared by training (and consistent with inference).

The original scaler was fitted on raw MIMIC values that still contained
charting errors (SpO2 mean 102.7 %, MBP std 1,722 mmHg, heart-rate std
339 bpm). After scaling, normal and critical vitals became almost identical
numbers, so the model effectively ignored heart rate, blood pressure and
SpO2. This module removes impossible values, clips to plausible clinical
ranges and fits the scaler on the TRAINING split only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.config import FEATURE_NAMES, FEATURES
from backend.scaler import JsonStandardScaler

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


def clean_structured(df: pd.DataFrame) -> pd.DataFrame:
    """Turn impossible values into NaN and clip plausible ones into range.

    Values outside the hard limits (e.g. SpO2 > 100, HR 999999 error codes)
    are physiologically impossible, so they are treated as missing rather than
    clipped (clipping would turn an error code into an extreme real value).
    """
    out = df.copy()
    for f in FEATURES:
        col = pd.to_numeric(out[f.name], errors="coerce")
        col = col.where((col >= f.min) & (col <= f.max))
        out[f.name] = col.clip(f.clip_min, f.clip_max)
    return out


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    medians: dict[str, float]
    scaler: JsonStandardScaler


def prepare(df: pd.DataFrame, seed: int = 42, val_size=0.15, test_size=0.15) -> Splits:
    """Clean, split (stratified), impute with train medians, fit scaler on train."""
    missing = {TEXT_COLUMN, LABEL_COLUMN, *FEATURE_NAMES} - set(df.columns)
    if missing:
        raise ValueError(f"Input data is missing columns: {sorted(missing)}")

    df = df.dropna(subset=[LABEL_COLUMN]).copy()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    df = clean_structured(df)

    train, rest = train_test_split(
        df, test_size=val_size + test_size, stratify=df[LABEL_COLUMN], random_state=seed
    )
    val, test = train_test_split(
        rest,
        test_size=test_size / (val_size + test_size),
        stratify=rest[LABEL_COLUMN],
        random_state=seed,
    )

    medians = train[FEATURE_NAMES].median().to_dict()
    train, val, test = (s.fillna(medians) for s in (train, val, test))

    scaler = JsonStandardScaler.fit(train[FEATURE_NAMES].to_numpy(dtype=np.float64))
    return Splits(
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
        medians,
        scaler,
    )
