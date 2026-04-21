# backend/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from predict import predict
import os

app = FastAPI()

# -----------------------------
# ✅ PATH CONFIGURATION
# -----------------------------
# This finds the 'frontend' folder which is one level up from the 'backend' folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# 1. Mount the frontend folder so the browser can load style.css and script.js
# We use "/frontend" as the web path to avoid conflicts with your other static folder
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

# 2. Keep your existing static mount if you use it for backend images/plots
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------------
# ✅ CORS FIX
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# INPUT FORMAT
# -----------------------------
class PatientInput(BaseModel):
    text: str
    features: list

# -----------------------------
# ✅ ROOT - NOW SERVES YOUR UI
# -----------------------------
@app.get("/")
def serve_frontend():
    # This sends your index.html directly to the browser
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# -----------------------------
# PREDICT
# -----------------------------
@app.post("/predict")
def predict_api(data: PatientInput):
    # This calls your logic from predict.py
    return predict(data.text, data.features)