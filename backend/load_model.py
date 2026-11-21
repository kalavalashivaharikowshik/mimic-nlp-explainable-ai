"""Download (if needed) and load the model, tokenizer and scaler.

Nothing is loaded at import time any more: importing this module is cheap, and
the heavy work happens only when :func:`load_artifacts` is called.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch

from . import config
from .model import HybridModel
from .scaler import JsonStandardScaler

logger = logging.getLogger(__name__)


@dataclass
class Artifacts:
    model: HybridModel
    tokenizer: object
    scaler: JsonStandardScaler
    device: torch.device


# -----------------------------
# Download helpers
# -----------------------------
def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def is_valid_weights_file(path: Path) -> bool:
    """Cheap sanity checks that catch truncated / HTML 'quota exceeded' downloads."""
    if not path.is_file():
        return False
    size = path.stat().st_size
    if size < config.MIN_MODEL_BYTES:
        logger.warning("Weights file %s is only %d bytes - treating it as corrupt", path, size)
        return False
    if config.MODEL_SHA256 and sha256_of(path) != config.MODEL_SHA256:
        logger.warning("Weights file %s failed the SHA-256 check", path)
        return False
    return True


def _download_to(tmp_path: Path) -> None:
    if config.MODEL_HF_REPO:
        from huggingface_hub import hf_hub_download

        logger.info("Downloading weights from Hugging Face repo %s", config.MODEL_HF_REPO)
        cached = hf_hub_download(repo_id=config.MODEL_HF_REPO, filename=config.MODEL_HF_FILENAME)
        shutil.copyfile(cached, tmp_path)
    elif "drive.google.com" in config.MODEL_URL:
        import gdown

        logger.info("Downloading weights from Google Drive")
        result = gdown.download(config.MODEL_URL, str(tmp_path), quiet=False)
        if result is None:
            raise RuntimeError("Google Drive download failed (file private or quota exceeded?)")
    else:
        logger.info("Downloading weights from %s", config.MODEL_URL)
        with (
            urllib.request.urlopen(config.MODEL_URL, timeout=60) as resp,
            open(tmp_path, "wb") as fh,
        ):
            shutil.copyfileobj(resp, fh)


def ensure_weights(path: Path = config.MODEL_PATH) -> Path:
    """Make sure a valid weights file exists at ``path``, downloading it if needed.

    The download goes to a temporary file that is only moved into place after it
    passes validation, so an interrupted download can never leave a broken file
    that blocks every future start.
    """
    path = Path(path)
    if is_valid_weights_file(path):
        return path
    if path.exists():
        path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".part")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        _download_to(tmp_path)
        if not is_valid_weights_file(tmp_path):
            raise RuntimeError(
                "Downloaded model file is invalid. Download it manually and place it at "
                f"{path} (see README)."
            )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return path


# -----------------------------
# Loading
# -----------------------------
def load_state_dict_compat(model: torch.nn.Module, state_dict: dict) -> None:
    """Load weights saved by older/newer transformers versions.

    ``embeddings.position_ids`` was a persistent buffer in transformers < 4.31
    and is not saved by newer versions; it is safe to ignore either way.
    ``module.`` prefixes from DataParallel training are stripped.
    """
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    result = model.load_state_dict(state_dict, strict=False)

    def harmless(key: str) -> bool:
        return key.endswith("position_ids")

    missing = [k for k in result.missing_keys if not harmless(k)]
    unexpected = [k for k in result.unexpected_keys if not harmless(k)]
    if missing or unexpected:
        raise RuntimeError(
            f"Weights do not match the model architecture. Missing: {missing[:5]} "
            f"Unexpected: {unexpected[:5]}"
        )


def load_artifacts() -> Artifacts:
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    weights = ensure_weights(config.MODEL_PATH)

    model = HybridModel()
    # weights_only=True refuses to unpickle arbitrary Python objects.
    state_dict = torch.load(weights, map_location="cpu", weights_only=True)
    load_state_dict_compat(model, state_dict)
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    scaler = JsonStandardScaler.load(config.SCALER_PATH)

    logger.info("Model, tokenizer and scaler loaded")
    return Artifacts(model=model, tokenizer=tokenizer, scaler=scaler, device=device)
