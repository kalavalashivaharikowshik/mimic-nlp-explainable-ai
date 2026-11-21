"""Convert the trained PyTorch model into lightweight ONNX Runtime artefacts.

Output folder (default ``dist/onnx``):

    encoder.onnx        BioBERT -> [CLS] embedding + [CLS] attention (float32)
    encoder.int8.onnx   mixed precision: int8 dynamic quantization, except the 12
                        feed-forward output projections which stay float32
                        (~2.2x smaller, used by default)
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
import re
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

# Synthetic clinical phrases (no patient data) used to measure how much
# quantization changes the predicted probabilities.
EVAL_PHRASES = [
    "patient is stable",
    "vitals within normal range",
    "no signs of infection",
    "tolerating diet",
    "ambulating independently",
    "plan transfer to floor",
    "afebrile overnight",
    "pain well controlled",
    "septic shock",
    "started on norepinephrine",
    "vasopressin added",
    "intubated for hypoxic respiratory failure",
    "acute kidney injury",
    "rising creatinine",
    "oliguric",
    "lactate elevated",
    "dnr/dni per family",
    "comfort measures discussed",
    "gi bleed",
    "transfused 2 units prbc",
    "altered mental status",
    "pneumonia on ct",
    "chf exacerbation",
    "copd exacerbation",
    "atrial fibrillation with rvr",
    "s/p cabg",
    "post-op day 1",
    "extubated this morning",
    "weaning sedation",
    "on bipap",
    "blood cultures pending",
    "broad spectrum antibiotics",
    "metastatic cancer",
    "liver cirrhosis",
    "hepatic encephalopathy",
    "dka on insulin drip",
    "stroke with left sided weakness",
    "family meeting held",
]

# BERT's feed-forward output projections (3072 -> 768) see large activation
# outliers; quantizing their inputs to int8 caused most of the error (max
# probability change 0.18 -> 0.055 when they are kept in float32).
FLOAT32_NODE_PATTERN = re.compile(r"/layer\.\d+/output/dense/MatMul$")


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


def quantize_encoder(src: Path, dst: Path) -> list[str]:
    """Mixed-precision int8: everything except the sensitive FFN output MatMuls."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    graph = onnx.load(str(src), load_external_data=False).graph
    keep_float = [
        n.name for n in graph.node if n.op_type == "MatMul" and FLOAT32_NODE_PATTERN.search(n.name)
    ]
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, nodes_to_exclude=keep_float)
    return keep_float


def evaluation_notes(n: int = 120, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    return [
        ". ".join(rng.choice(EVAL_PHRASES, size=int(rng.integers(1, 12)), replace=False)) + "."
        for _ in range(n)
    ]


def quantization_report(
    out_dir: Path, encoder_file: str, n_struct: int, threshold: float = config.THRESHOLD
) -> dict:
    """Compare an encoder with encoder.onnx on synthetic notes + synthetic vitals."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    from backend.background import make_background
    from backend.scaler import JsonStandardScaler
    from backend.utils import clean_text, clip_features

    fast = Tokenizer.from_file(str(out_dir / "tokenizer.json"))
    fast.enable_truncation(config.MAX_LENGTH)
    head = FusionHead.load(out_dir / "head.npz")
    notes = evaluation_notes()
    if n_struct == len(config.FEATURE_NAMES):
        x = JsonStandardScaler.load(config.SCALER_PATH).transform(
            clip_features(make_background(len(notes), seed=7))
        )
    else:
        x = np.random.default_rng(7).normal(size=(len(notes), n_struct))

    def probs(name: str) -> np.ndarray:
        session = ort.InferenceSession(str(out_dir / name), providers=["CPUExecutionProvider"])
        out = []
        for note, row in zip(notes, x, strict=True):
            e = fast.encode(clean_text(note))
            emb, _ = session.run(
                None,
                {"input_ids": np.array([e.ids]), "attention_mask": np.array([e.attention_mask])},
            )
            out.append(head.proba(head.text_part(emb[0]), row[None])[0])
        return np.array(out)

    ref, got = probs("encoder.onnx"), probs(encoder_file)
    diff = np.abs(got - ref)
    flips = (got >= threshold) != (ref >= threshold)
    return {
        "n_notes": len(notes),
        "max_abs_diff": round(float(diff.max()), 5),
        "mean_abs_diff": round(float(diff.mean()), 5),
        "label_flips": int(flips.sum()),
        "max_flip_distance_from_threshold": round(float(np.abs(ref[flips] - threshold).max()), 5)
        if flips.any()
        else 0.0,
    }


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
    float32_nodes: list[str] = []
    if quantize:
        float32_nodes = quantize_encoder(out_dir / "encoder.onnx", out_dir / "encoder.int8.onnx")
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

    quantization = None
    if quantize:
        quantization = quantization_report(
            out_dir, "encoder.int8.onnx", int(model.struct_net[0].in_features)
        )
        quantization["float32_nodes"] = len(float32_nodes)
        logger.info(
            "Quantization vs float32 on %d synthetic notes: %s",
            quantization["n_notes"],
            quantization,
        )
        # Labels may only flip for patients that sit right at the threshold.
        if (
            quantization["mean_abs_diff"] > 0.02
            or quantization["max_flip_distance_from_threshold"] > 0.05
        ):
            raise RuntimeError(f"int8 encoder is not accurate enough: {quantization}")

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
        "quantization": quantization,
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
