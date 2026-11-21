"""Convert the trained PyTorch model into lightweight ONNX Runtime artefacts.

Output folder (default ``dist/onnx``):

    encoder.onnx        BioBERT -> [CLS] embedding + [CLS] attention (float32)
    encoder.int8.onnx   same, int8 dynamic quantization (~4x smaller, used by default)
    head.npz            fusion head weights for backend/fusion.py
    tokenizer.json      fast tokenizer for the `tokenizers` library
    manifest.json       versions, settings and SHA-256 of every file

The web server only needs these files plus onnxruntime / tokenizers / numpy.
PyTorch and transformers are needed here, at export time, only.

Usage:
    python -m tools.export_onnx --weights backend/models/hybrid_model.pth
    python -m tools.export_onnx --download            # fetch weights from MODEL_URL first
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from backend import config
from backend.downloads import download_file, sha256_of
from backend.fusion import HEAD_KEYS, FusionHead
from backend.model import HybridModel, load_state_dict_compat

logger = logging.getLogger("export_onnx")

FORMAT_VERSION = 1
OPSET = 17
PARITY_TEXTS = [
    "patient is stable, vitals within normal range, no signs of infection",
    "78 y/o with septic shock on norepinephrine, intubated overnight for hypoxic respiratory failure",
    "",
]


class EncoderForExport(nn.Module):
    """Outputs exactly what the server needs: CLS embedding and CLS attention."""

    def __init__(self, model: HybridModel):
        super().__init__()
        self.text_model = model.text_model

    def forward(self, input_ids, attention_mask):
        out = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask, output_attentions=True
        )
        cls_embedding = out.last_hidden_state[:, 0, :]
        # Last layer, mean over heads, attention from [CLS] to every token.
        cls_attention = out.attentions[-1].mean(dim=1)[:, 0, :]
        return cls_embedding, cls_attention


def export_encoder(model: HybridModel, path: Path, tokenizer) -> None:
    # The wrapper must be in eval mode too: torch.onnx.export restores the
    # wrapper's original training flag afterwards, which would otherwise switch
    # BioBERT's dropout back on for everything that runs after the export.
    wrapper = EncoderForExport(model).eval()
    sample = tokenizer("patient in septic shock", return_tensors="pt")
    torch.onnx.export(
        wrapper,
        (sample["input_ids"], sample["attention_mask"]),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["cls_embedding", "cls_attention"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "cls_embedding": {0: "batch"},
            "cls_attention": {0: "batch", 1: "sequence"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
    )
    model.eval()


def quantize_encoder(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)


def export_head(model: HybridModel, path: Path) -> None:
    state = model.state_dict()
    np.savez(path, **{k: state[k].detach().cpu().numpy().astype(np.float32) for k in HEAD_KEYS})


def export_tokenizer(tokenizer, path: Path) -> None:
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required (AutoTokenizer(..., use_fast=True))")
    tokenizer.backend_tokenizer.save(str(path))


def parity_check(
    model: HybridModel, tokenizer, out_dir: Path, encoder_file: str, max_length: int
) -> float:
    """Max |p_torch - p_onnx| over sample notes and random vitals."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    session = ort.InferenceSession(str(out_dir / encoder_file), providers=["CPUExecutionProvider"])
    fast = Tokenizer.from_file(str(out_dir / "tokenizer.json"))
    fast.enable_truncation(max_length)
    head = FusionHead.load(out_dir / "head.npz")

    rng = np.random.default_rng(0)
    scaled = rng.normal(size=(8, model.struct_net[0].in_features)).astype(np.float32)
    worst = 0.0
    for text in PARITY_TEXTS:
        enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            ref = torch.sigmoid(
                model(enc["input_ids"], enc["attention_mask"], torch.from_numpy(scaled))
            ).squeeze(-1)
        e = fast.encode(text)
        ids = np.array([e.ids], dtype=np.int64)
        mask = np.array([e.attention_mask], dtype=np.int64)
        emb, _ = session.run(None, {"input_ids": ids, "attention_mask": mask})
        got = head.proba(head.text_part(emb[0]), scaled)
        worst = max(worst, float(np.abs(ref.numpy() - got).max()))
    return worst


def export_all(
    model: HybridModel,
    tokenizer,
    out_dir: Path,
    quantize: bool = True,
    max_length: int = config.MAX_LENGTH,
    lowercase: bool = config.LOWERCASE_TEXT,
    text_model_name: str = config.TEXT_MODEL_NAME,
    source_sha256: str | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    export_tokenizer(tokenizer, out_dir / "tokenizer.json")
    export_head(model, out_dir / "head.npz")
    export_encoder(model, out_dir / "encoder.onnx", tokenizer)
    files = ["encoder.onnx", "head.npz", "tokenizer.json"]
    default_encoder = "encoder.onnx"
    if quantize:
        quantize_encoder(out_dir / "encoder.onnx", out_dir / "encoder.int8.onnx")
        files.append("encoder.int8.onnx")
        default_encoder = "encoder.int8.onnx"

    parity = {
        name: parity_check(model, tokenizer, out_dir, name, max_length)
        for name in files
        if name.startswith("encoder")
    }
    for name, diff in parity.items():
        logger.info("Parity %s: max |p_torch - p_onnx| = %.6f", name, diff)
    if parity["encoder.onnx"] > 1e-3:
        raise RuntimeError(
            f"float32 ONNX export does not match PyTorch (diff {parity['encoder.onnx']})"
        )

    manifest = {
        "format_version": FORMAT_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "text_model": text_model_name,
        "hidden_size": int(model.text_model.config.hidden_size),
        "n_structured": int(model.struct_net[0].in_features),
        "max_length": int(max_length),
        "lowercase_text": bool(lowercase),
        "default_encoder": default_encoder,
        "source_weights_sha256": source_sha256,
        "parity_max_abs_diff": parity,
        "files": {
            name: {"sha256": sha256_of(out_dir / name), "bytes": (out_dir / name).stat().st_size}
            for name in files
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--weights", type=Path, default=config.BACKEND_DIR / "models" / "hybrid_model.pth"
    )
    parser.add_argument(
        "--download", action="store_true", help="download weights from MODEL_URL if missing"
    )
    parser.add_argument("--out", type=Path, default=Path("dist/onnx"))
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.download:
        download_file(config.MODEL_URL, args.weights, min_bytes=50 * 1024 * 1024)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME, use_fast=True)
    model = HybridModel()
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    load_state_dict_compat(model, state)

    manifest = export_all(
        model,
        tokenizer,
        args.out,
        quantize=not args.no_quantize,
        source_sha256=sha256_of(args.weights),
    )
    for name, info in manifest["files"].items():
        logger.info("%-20s %8.1f MB  %s", name, info["bytes"] / 1e6, info["sha256"][:12])
    logger.info("Done. Upload the contents of %s as a GitHub release (see README).", args.out)


if __name__ == "__main__":
    main()
