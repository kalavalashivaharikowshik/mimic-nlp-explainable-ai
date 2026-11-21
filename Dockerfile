# Production image: FastAPI backend + static frontend on one port.
# Works on Hugging Face Spaces (Docker SDK), Render, Railway, Fly.io, or locally:
#   docker build -t mimic-nlp .
#   docker run -p 7860:7860 mimic-nlp      ->  http://localhost:7860

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860 \
    HF_HOME=/app/.cache/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# Install dependencies first so this layer is cached between code changes.
# CPU-only PyTorch keeps the image ~2 GB smaller than the CUDA build.
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY --chown=appuser:appuser backend ./backend
COPY --chown=appuser:appuser frontend ./frontend
RUN mkdir -p /app/.cache /app/backend/models && chown -R appuser:appuser /app

USER appuser

# Optionally bake the model weights and BioBERT tokenizer into the image for
# fast cold starts. If the download fails at build time (e.g. Google Drive
# quota), the app simply downloads them on first start instead.
ARG PRELOAD_MODEL=1
RUN if [ "$PRELOAD_MODEL" = "1" ]; then \
      python -c "from transformers import AutoConfig, AutoTokenizer; from backend.config import TEXT_MODEL_NAME as m; AutoConfig.from_pretrained(m); AutoTokenizer.from_pretrained(m)" \
      && python -c "from backend.load_model import ensure_weights; ensure_weights()" \
      || echo 'Model preload skipped - it will be downloaded at startup.'; \
    fi

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"7860\")}/health', timeout=4)" || exit 1

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
