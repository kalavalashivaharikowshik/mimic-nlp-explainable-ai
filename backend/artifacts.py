"""Download (if needed) and load the ONNX Runtime artefacts.

The server needs only onnxruntime, tokenizers and numpy - no PyTorch. The
artefacts are produced by ``tools/export_onnx.py`` and published as a GitHub
release; ``manifest.json`` lists the SHA-256 of every file, so a truncated or
tampered download is rejected.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from . import config
from .downloads import download_file, is_valid_file
from .fusion import FusionHead
from .scaler import JsonStandardScaler

logger = logging.getLogger(__name__)


@dataclass
class OnnxArtifacts:
    session: object  # onnxruntime.InferenceSession
    tokenizer: object  # tokenizers.Tokenizer
    head: FusionHead
    scaler: JsonStandardScaler
    manifest: dict
    model_dir: Path

    @property
    def max_length(self) -> int:
        return int(self.manifest.get("max_length", config.MAX_LENGTH))

    @property
    def lowercase(self) -> bool:
        return bool(self.manifest.get("lowercase_text", config.LOWERCASE_TEXT))


def encoder_file(manifest: dict) -> str:
    return config.ONNX_ENCODER or manifest.get("default_encoder", "encoder.int8.onnx")


def ensure_artifacts(
    model_dir: Path = config.ONNX_DIR, base_url: str = config.MODEL_BASE_URL
) -> dict:
    """Make sure every file the server needs exists and matches the manifest."""
    model_dir = Path(model_dir)
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        if not base_url:
            raise RuntimeError(f"No model found in {model_dir} and MODEL_BASE_URL is empty")
        download_file(f"{base_url.rstrip('/')}/manifest.json", manifest_path, min_bytes=50)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    needed = [encoder_file(manifest), "head.npz", "tokenizer.json"]
    for name in needed:
        info = manifest["files"].get(name)
        if info is None:
            raise RuntimeError(f"{name} is not listed in {manifest_path}")
        if not base_url:
            if not is_valid_file(model_dir / name, sha256=info["sha256"]):
                raise RuntimeError(
                    f"{model_dir / name} is missing or corrupt and MODEL_BASE_URL is empty"
                )
            continue
        download_file(
            f"{base_url.rstrip('/')}/{name}",
            model_dir / name,
            sha256=info["sha256"],
        )
    return manifest


def load_artifacts(
    model_dir: Path = config.ONNX_DIR, base_url: str = config.MODEL_BASE_URL
) -> OnnxArtifacts:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    model_dir = Path(model_dir)
    manifest = ensure_artifacts(model_dir, base_url)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if config.ORT_LOW_MEMORY:
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.add_session_config_entry("session.disable_prepacking", "1")
    if config.ORT_THREADS > 0:
        options.intra_op_num_threads = config.ORT_THREADS
        options.inter_op_num_threads = 1
    encoder = encoder_file(manifest)
    session = ort.InferenceSession(
        str(model_dir / encoder), options, providers=["CPUExecutionProvider"]
    )

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.enable_truncation(int(manifest.get("max_length", config.MAX_LENGTH)))
    tokenizer.no_padding()

    head = FusionHead.load(model_dir / "head.npz")
    scaler = JsonStandardScaler.load(config.SCALER_PATH)
    if head.hidden != manifest.get("hidden_size", head.hidden):
        raise RuntimeError("head.npz does not match the encoder hidden size in manifest.json")

    logger.info(
        "Loaded %s (%s) with ONNX Runtime %s", encoder, manifest.get("text_model"), ort.__version__
    )
    return OnnxArtifacts(
        session=session,
        tokenizer=tokenizer,
        head=head,
        scaler=scaler,
        manifest=manifest,
        model_dir=model_dir,
    )
