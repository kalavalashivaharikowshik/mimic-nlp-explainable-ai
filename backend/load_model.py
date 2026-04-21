# backend/load_model.py

import os
import torch
import joblib
import gdown
from transformers import AutoTokenizer
from model import HybridModel

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "models/hybrid_model.pth"

# 🔥 PUT YOUR GOOGLE DRIVE FILE ID HERE
MODEL_URL = "https://drive.google.com/uc?id=1g9yuOIIFgxG-xR4_ri6w6LRpv9OBsvQw"

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# DOWNLOAD MODEL IF NOT EXISTS
# -----------------------------
if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from Google Drive...")
    os.makedirs("models", exist_ok=True)
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = HybridModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -----------------------------
# TOKENIZER
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# -----------------------------
# SCALER
# -----------------------------
scaler = joblib.load("scaler/scaler.pkl")