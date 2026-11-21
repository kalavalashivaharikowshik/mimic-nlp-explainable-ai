"""FastAPI application: serves the web UI and the prediction API.

Run from the project root:
    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, config
from .predict import InvalidInputError, MortalityPredictor
from .schemas import PatientInput, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("mimic-nlp")


class ModelState:
    """Tracks background model loading so the server can start instantly."""

    def __init__(self):
        self.status = "not_loaded"  # not_loaded | loading | ready | error
        self.error: str | None = None
        self.predictor: MortalityPredictor | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self.status in ("loading", "ready"):
                return
            self.status = "loading"
        try:
            from .artifacts import load_artifacts

            predictor = MortalityPredictor(load_artifacts())
            with self._lock:
                self.predictor = predictor
                self.status = "ready"
            logger.info("Model ready")
        except Exception as exc:  # surfaced through /health
            logger.exception("Model failed to load")
            with self._lock:
                if self.status != "ready":
                    self.error = str(exc).splitlines()[0][:300]
                    self.status = "error"

    def set_predictor(self, predictor: MortalityPredictor) -> None:
        """Inject a ready predictor (used by tests)."""
        with self._lock:
            self.predictor = predictor
            self.status = "ready"
            self.error = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    state: ModelState = app.state.model
    if config.LOAD_MODEL_ON_STARTUP and state.status == "not_loaded":
        threading.Thread(target=state.load, name="model-loader", daemon=True).start()
    yield


app = FastAPI(
    title="ICU Mortality Prediction API",
    description=(
        "Multimodal (clinical note + vitals/labs) mortality risk model with attention, "
        "SHAP and LIME explanations. " + config.DISCLAIMER
    ),
    version=__version__,
    lifespan=lifespan,
)
app.state.model = ModelState()

# CORS is only needed when the UI is hosted on a different origin.
if config.ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.ALLOWED_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    return response


# -----------------------------
# Error handling: consistent {"detail": [messages]} bodies
# -----------------------------
@app.exception_handler(RequestValidationError)
async def request_validation_handler(request: Request, exc: RequestValidationError):
    messages = []
    for err in exc.errors():
        location = ".".join(str(p) for p in err.get("loc", []) if p != "body")
        messages.append(f"{location}: {err.get('msg')}" if location else err.get("msg"))
    return JSONResponse(status_code=422, content={"detail": messages})


@app.exception_handler(InvalidInputError)
async def invalid_input_handler(request: Request, exc: InvalidInputError):
    return JSONResponse(status_code=422, content={"detail": exc.errors})


# -----------------------------
# Web UI
# -----------------------------
app.mount("/frontend", StaticFiles(directory=config.FRONTEND_DIR), name="frontend")


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(config.FRONTEND_DIR / "index.html")


# -----------------------------
# API
# -----------------------------
@app.get("/health", tags=["system"])
def health():
    state: ModelState = app.state.model
    body = {"status": "ok", "model": state.status, "version": __version__}
    if state.error:
        body["error"] = state.error
    return body


@app.get("/api/metadata", tags=["system"])
def metadata():
    return {
        "features": [
            {"name": f.name, "label": f.label, "unit": f.unit, "min": f.min, "max": f.max}
            for f in config.FEATURES
        ],
        "threshold": config.THRESHOLD,
        "max_text_chars": config.MAX_TEXT_CHARS,
        "disclaimer": config.DISCLAIMER,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict_api(data: PatientInput):
    state: ModelState = app.state.model
    if state.status != "ready" or state.predictor is None:
        if state.status == "error":
            return JSONResponse(
                status_code=503,
                content={"detail": [f"Model failed to load: {state.error}"]},
            )
        if state.status == "not_loaded":
            return JSONResponse(
                status_code=503,
                content={"detail": ["The model is not loaded (LOAD_MODEL_ON_STARTUP is off)."]},
            )
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "15"},
            content={"detail": ["The model is still loading. Please try again in a moment."]},
        )
    return state.predictor.predict(data.text, data.features)
