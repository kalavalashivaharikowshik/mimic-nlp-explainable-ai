"""Train the hybrid BioBERT + vitals model and export it for the web app.

Input: a CSV/Parquet file you build from MIMIC (PhysioNet credentialed access
required; never commit it) with one row per ICU stay and columns:

    text, age, gender, heart_rate, sbp, dbp, mbp, spo2, creatinine, glucose,
    hemoglobin, label        (label: 1 = died in hospital, 0 = survived)

gender must be 0 = male, 1 = female (same as the web form).

Usage:
    python -m training.train --data training/data/cohort.csv --epochs 3

Outputs (training/outputs/ by default):
    hybrid_model.pth   -> copy to backend/models/ (or upload to Hugging Face)
    scaler.json        -> copy to backend/scaler/scaler.json
    metrics.json       -> validation/test AUROC, AUPRC, F1, chosen threshold
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from backend.config import FEATURE_NAMES, LOWERCASE_TEXT, MAX_LENGTH, TEXT_MODEL_NAME
from backend.model import HybridModel
from backend.utils import clean_text
from training.preprocess import LABEL_COLUMN, TEXT_COLUMN, prepare

logger = logging.getLogger("train")


class StayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, scaler, tokenizer, max_length: int, lowercase: bool):
        self.texts = [clean_text(t, lowercase=lowercase) for t in df[TEXT_COLUMN]]
        self.x = scaler.transform(df[FEATURE_NAMES].to_numpy(dtype=np.float64)).astype(np.float32)
        self.y = df[LABEL_COLUMN].to_numpy(dtype=np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.texts[i], self.x[i], self.y[i]

    def collate(self, batch):
        texts, x, y = zip(*batch, strict=True)
        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"], torch.tensor(np.stack(x)), torch.tensor(y)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_proba(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for ids, mask, x, y in loader:
        logits = model(ids.to(device), mask.to(device), x.to(device)).squeeze(-1)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold that maximises F1 on the validation set."""
    candidates = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, y_prob >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = y_prob >= threshold
    both_classes = len(np.unique(y_true)) > 1
    return {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, y_prob)) if both_classes else None,
        "auprc": float(average_precision_score(y_true, y_prob)) if both_classes else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
    }


def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total = 0.0
    for ids, mask, x, y in loader:
        optimizer.zero_grad()
        logits = model(ids.to(device), mask.to(device), x.to(device)).squeeze(-1)
        loss = loss_fn(logits, y.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


def run(
    df: pd.DataFrame,
    model: HybridModel,
    tokenizer,
    out_dir: Path,
    epochs: int = 3,
    batch_size: int = 16,
    lr_text: float = 2e-5,
    lr_head: float = 1e-3,
    max_length: int = MAX_LENGTH,
    lowercase: bool = LOWERCASE_TEXT,
    seed: int = 42,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = prepare(df, seed=seed)
    logger.info("train=%d val=%d test=%d", len(splits.train), len(splits.val), len(splits.test))

    def loader(frame, shuffle):
        ds = StayDataset(frame, splits.scaler, tokenizer, max_length, lowercase)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=ds.collate)

    train_dl, val_dl, test_dl = (
        loader(splits.train, True),
        loader(splits.val, False),
        loader(splits.test, False),
    )

    model.to(device)
    # Mortality is rare (~10-15 %): weight positives so the model does not just predict "survives".
    pos = float(splits.train[LABEL_COLUMN].mean())
    pos_weight = torch.tensor([(1 - pos) / max(pos, 1e-6)], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.text_model.parameters(), "lr": lr_text},
            {
                "params": list(model.struct_net.parameters()) + list(model.classifier.parameters()),
                "lr": lr_head,
            },
        ],
        weight_decay=0.01,
    )

    best_auprc, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_dl, optimizer, loss_fn, device)
        val_prob, val_y = predict_proba(model, val_dl, device)
        auprc = average_precision_score(val_y, val_prob) if len(np.unique(val_y)) > 1 else 0.0
        history.append({"epoch": epoch, "train_loss": loss, "val_auprc": float(auprc)})
        logger.info("epoch %d  loss %.4f  val AUPRC %.4f", epoch, loss, auprc)
        if auprc > best_auprc:
            best_auprc = auprc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_prob, val_y = predict_proba(model, val_dl, device)
    threshold = best_f1_threshold(val_y, val_prob)
    test_prob, test_y = predict_proba(model, test_dl, device)

    report = {
        "text_model": TEXT_MODEL_NAME,
        "max_length": max_length,
        "lowercase_text": lowercase,
        "history": history,
        "validation": metrics(val_y, val_prob, threshold),
        "test": metrics(test_y, test_prob, threshold),
        "imputation_medians": splits.medians,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "hybrid_model.pth")
    splits.scaler.save(
        out_dir / "scaler.json", source="training/train.py (cleaned, train split only)"
    )
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Test metrics: %s", json.dumps(report["test"], indent=2))
    logger.info("Saved model, scaler and metrics to %s", out_dir)
    logger.info("Set THRESHOLD=%.2f when serving this model.", threshold)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, type=Path, help="CSV or Parquet cohort file")
    parser.add_argument("--out", default=Path("training/outputs"), type=Path)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--lr-text", default=2e-5, type=float)
    parser.add_argument("--lr-head", default=1e-3, type=float)
    parser.add_argument("--max-length", default=MAX_LENGTH, type=int)
    parser.add_argument("--keep-case", action="store_true", help="do not lower-case notes")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = pd.read_parquet(args.data) if args.data.suffix == ".parquet" else pd.read_csv(args.data)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    model = HybridModel(pretrained=True)
    run(
        df,
        model,
        tokenizer,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_text=args.lr_text,
        lr_head=args.lr_head,
        max_length=args.max_length,
        lowercase=not args.keep_case,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
