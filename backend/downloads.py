"""Safe file downloads (no PyTorch dependency).

Files are downloaded to a temporary file next to the destination and only
moved into place after they pass validation (minimum size and optional
SHA-256), so an interrupted download or an HTML error page can never be left
behind as a "valid" file.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

USER_AGENT = "mimic-nlp-explainable-ai/2 (+https://github.com/kalavalashivaharikowshik/mimic-nlp-explainable-ai)"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def is_valid_file(path: Path, min_bytes: int = 1, sha256: str | None = None) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    size = path.stat().st_size
    if size < min_bytes:
        logger.warning("%s is only %d bytes - treating it as corrupt", path, size)
        return False
    if sha256 and sha256_of(path) != sha256.lower():
        logger.warning("%s failed the SHA-256 check", path)
        return False
    return True


def _fetch(url: str, tmp_path: Path) -> None:
    if "drive.google.com" in url:
        import gdown  # only needed for Google Drive links

        if gdown.download(url, str(tmp_path), quiet=False) is None:
            raise RuntimeError("Google Drive download failed (file private or quota exceeded?)")
        return
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as resp, open(tmp_path, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)


def download_file(url: str, dest: Path, min_bytes: int = 1, sha256: str | None = None) -> Path:
    """Download ``url`` to ``dest`` unless a valid copy already exists."""
    dest = Path(dest)
    if is_valid_file(dest, min_bytes, sha256):
        return dest
    if dest.exists():
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        logger.info("Downloading %s", url)
        _fetch(url, tmp_path)
        if not is_valid_file(tmp_path, min_bytes, sha256):
            raise RuntimeError(
                f"Downloaded file from {url} is invalid (too small or wrong checksum)"
            )
        os.replace(tmp_path, dest)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return dest
