"""Tiny stand-ins for BioBERT so the full pipeline can be tested offline in seconds.

``make_artifacts`` builds the PyTorch model (used by training / export tests);
``make_onnx_artifacts`` exports it with the real tools/export_onnx.py and loads
it exactly like the production server does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import BertConfig, BertTokenizerFast

from backend.artifacts import OnnxArtifacts, load_artifacts
from backend.config import SCALER_PATH
from backend.model import HybridModel
from backend.scaler import JsonStandardScaler


@dataclass
class TorchArtifacts:
    model: HybridModel
    tokenizer: BertTokenizerFast
    scaler: JsonStandardScaler
    device: torch.device


VOCAB_WORDS = """
patient stable sepsis septic shock norepinephrine intubated ventilator hypotension
vitals normal range signs infection no within the and of on is has low blood pressure
creatinine kidney injury overnight chest pain denies oxygen saturation icu bp hr
""".split()


def make_tokenizer(tmp_dir: Path) -> BertTokenizerFast:
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ",", ".", "##s", "##ed"] + VOCAB_WORDS
    path = Path(tmp_dir) / "vocab.txt"
    path.write_text("\n".join(vocab), encoding="utf-8")
    return BertTokenizerFast(vocab_file=str(path), do_lower_case=True)


def make_artifacts(tmp_dir: Path, seed: int = 0) -> TorchArtifacts:
    torch.manual_seed(seed)
    tokenizer = make_tokenizer(tmp_dir)
    cfg = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
    )
    model = HybridModel(text_config=cfg).eval()
    scaler = JsonStandardScaler.load(SCALER_PATH)
    return TorchArtifacts(
        model=model, tokenizer=tokenizer, scaler=scaler, device=torch.device("cpu")
    )


def make_onnx_artifacts(
    tmp_dir: Path, quantize: bool = False
) -> tuple[OnnxArtifacts, TorchArtifacts]:
    """Export the tiny model to ONNX and load it like the server (no downloads)."""
    from tools.export_onnx import export_all

    torch_arts = make_artifacts(tmp_dir)
    out = Path(tmp_dir) / "onnx"
    export_all(torch_arts.model, torch_arts.tokenizer, out, quantize=quantize)
    return load_artifacts(out, base_url=""), torch_arts
