# Lean production image: FastAPI + ONNX Runtime (no PyTorch), UI and API on one port.
# Runs within 512 MB RAM (Render free tier), or anywhere Docker runs:
#   docker build -t mimic-nlp .
#   docker run -p 8000:8000 mimic-nlp      ->  http://localhost:8000

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    ORT_THREADS=1

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# Dependencies first so this layer is cached between code changes (~150 MB).
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY --chown=appuser:appuser backend ./backend
COPY --chown=appuser:appuser frontend ./frontend
RUN mkdir -p /app/backend/models && chown -R appuser:appuser /app

USER appuser

# Bake the ONNX model (~110 MB, from the GitHub release) into the image, so
# free hosts with an ephemeral disk don't re-download it on every restart.
# If the release doesn't exist yet, the app downloads it at startup instead.
ARG PRELOAD_MODEL=1
RUN if [ "$PRELOAD_MODEL" = "1" ]; then \
      python -c "from backend.artifacts import ensure_artifacts; ensure_artifacts()" \
      || echo 'Model preload skipped - it will be downloaded at startup.'; \
    fi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=4)" || exit 1

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
