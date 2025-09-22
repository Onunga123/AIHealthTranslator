# app/main.py
from fastapi import FastAPI, HTTPException
from pathlib import Path
import logging
import torch

from app.database import init_db
from app.scripts.seed_dictionary import seed_dictionary_terms

# Routers
from app.routers import dictionary, auth

# Transformers imports
from transformers import MarianMTModel, MarianTokenizer

# Add CORS middleware for frontend-backend integration
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# Initialize FastAPI app
# ----------------------------
app = FastAPI(
    title="AI Health Translator",
    description="""
AI Health Translator is an intelligent system for translating 
medical and health-related terms between English, Luo, and Kiswahili.  
It integrates a custom-trained Luo model, Hugging Face's Kiswahili model, 
and a health dictionary for fallback lookups.
""",
    version="1.0.0",
    contact={
        "name": "AI Health Translator Team",
        "email": "support@aihealthtranslator.org",
    },
    license_info={
        "name": "MIT License",
    },
)
# Allow frontend (localhost:3000) to access backend APIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Initialize Database
# ----------------------------
logging.info("Starting DB initialization...")
init_db()
seed_dictionary_terms()  # ✅ ensures dictionary has seed terms
logging.info("DB initialization complete.")

# ----------------------------
# Luo model paths (local)
# ----------------------------
LUO_MODEL_DIR = Path(__file__).resolve().parent.parent / "training" / "luo_model"
SOURCE_SPM = LUO_MODEL_DIR / "source.spm"
TARGET_SPM = LUO_MODEL_DIR / "target.spm"
VOCAB_FILE = LUO_MODEL_DIR / "vocab.json"
MODEL_FILE = LUO_MODEL_DIR / "model.safetensors"

# ----------------------------
# Load Luo model and tokenizer
# ----------------------------
try:
    if not (SOURCE_SPM.exists() and TARGET_SPM.exists() and VOCAB_FILE.exists() and MODEL_FILE.exists()):
        raise FileNotFoundError("One or more Luo model files are missing in the training/luo_model directory.")

    luo_tokenizer = MarianTokenizer(
        source_spm=str(SOURCE_SPM),
        target_spm=str(TARGET_SPM),
        vocab=str(VOCAB_FILE),
    )
    
    luo_model = MarianMTModel.from_pretrained(
        str(LUO_MODEL_DIR),
        local_files_only=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    logging.info("✅ Luo model and tokenizer loaded successfully.")

except Exception as e:
    raise RuntimeError(f"Failed to load Luo model/tokenizer: {e}")

# ----------------------------
# Kiswahili model (Hugging Face)
# ----------------------------
KIS_MODEL_NAME = "Helsinki-NLP/opus-mt-en-sw"  # English ↔ Kiswahili
try:
    kis_tokenizer = MarianTokenizer.from_pretrained(KIS_MODEL_NAME)
    kis_model = MarianMTModel.from_pretrained(KIS_MODEL_NAME)
    logging.info("✅ Kiswahili model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load Kiswahili model/tokenizer: {e}")

# ----------------------------
# API Endpoints
# ----------------------------

@app.get("/", tags=["Root"])
async def root():
    return {"message": "AI Health Translator is running!"}

# Luo translation
@app.get("/translate/luo/{text}", tags=["Translation"])
async def translate_luo(text: str):
    try:
        inputs = luo_tokenizer(text, return_tensors="pt", padding=True)
        translated = luo_model.generate(**inputs)
        output = [luo_tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return {"input": text, "translation": output[0], "language": "Luo"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Luo translation failed: {e}")

# Kiswahili translation
@app.get("/translate/sw/{text}", tags=["Translation"])
async def translate_kiswahili(text: str):
    try:
        inputs = kis_tokenizer(text, return_tensors="pt", padding=True)
        translated = kis_model.generate(**inputs)
        output = [kis_tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return {"input": text, "translation": output[0], "language": "Swahili"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kiswahili translation failed: {e}")

# ----------------------------
# Include Routers
# ----------------------------
app.include_router(auth.router)          # Authentication
app.include_router(dictionary.router)    # Health Dictionary
