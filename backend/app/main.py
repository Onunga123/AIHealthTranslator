# app/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging
import sys
import torch

from app.database import init_db
from app.scripts.seed_dictionary import seed_dictionary_terms

# Routers
from app.routers import dictionary, auth

# Transformers imports
from transformers import MarianMTModel, MarianTokenizer

# ----------------------------
# Logging configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),           # console
        logging.FileHandler("backend.log", mode="a") # file logging
    ]
)
logger = logging.getLogger("backend")

# ----------------------------
# Initialize FastAPI app
# ----------------------------
app = FastAPI(
    title="AI Health Translator",
    description="AI Health Translator translates medical terms between English, Luo, and Kiswahili.",
    version="1.0.0",
    contact={"name": "AI Health Translator Team", "email": "support@aihealthtranslator.org"},
    license_info={"name": "MIT License"},
)

# ----------------------------
# CORS configuration
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "https://frontend-opal-five-82.vercel.app",  # Vercel frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Middleware: log all requests
# ----------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"[REQUEST] Incoming: {request.method} {request.url}")
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"[REQUEST] Unhandled exception: {e}", exc_info=True)
        raise
    logger.info(f"[REQUEST] Completed with status {response.status_code}")
    return response

# ----------------------------
# Initialize Database
# ----------------------------
try:
    logger.info("Starting database initialization...")
    init_db()
    seed_dictionary_terms()
    logger.info("Database initialization complete.")
except Exception as e:
    logger.error(f"Database initialization failed: {e}", exc_info=True)
    raise

# ----------------------------
# Luo model paths
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
        raise FileNotFoundError("One or more Luo model files are missing in training/luo_model directory.")

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
    logger.info("Luo model and tokenizer loaded successfully.")

except Exception as e:
    logger.error(f"Failed to load Luo model/tokenizer: {e}", exc_info=True)
    raise RuntimeError(f"Failed to load Luo model/tokenizer: {e}")

# ----------------------------
# Kiswahili model (Hugging Face)
# ----------------------------
KIS_MODEL_NAME = "Helsinki-NLP/opus-mt-en-sw"
try:
    kis_tokenizer = MarianTokenizer.from_pretrained(KIS_MODEL_NAME)
    kis_model = MarianMTModel.from_pretrained(KIS_MODEL_NAME)
    logger.info("Kiswahili model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Kiswahili model/tokenizer: {e}", exc_info=True)
    raise RuntimeError(f"Failed to load Kiswahili model/tokenizer: {e}")

# ----------------------------
# API Endpoints
# ----------------------------
@app.get("/", tags=["Root"])
async def root():
    logger.info("[ROOT] Root endpoint accessed")
    return {"message": "AI Health Translator is running!"}

@app.get("/translate/luo/{text}", tags=["Translation"])
async def translate_luo(text: str):
    try:
        inputs = luo_tokenizer(text, return_tensors="pt", padding=True)
        translated = luo_model.generate(**inputs)
        output = [luo_tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        logger.info(f"[TRANSLATION] Luo: '{text}' -> '{output[0]}'")
        return {"input": text, "translation": output[0], "language": "Luo"}
    except Exception as e:
        logger.error(f"[TRANSLATION] Luo translation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Luo translation failed: {e}")

@app.get("/translate/sw/{text}", tags=["Translation"])
async def translate_kiswahili(text: str):
    try:
        inputs = kis_tokenizer(text, return_tensors="pt", padding=True)
        translated = kis_model.generate(**inputs)
        output = [kis_tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        logger.info(f"[TRANSLATION] Kiswahili: '{text}' -> '{output[0]}'")
        return {"input": text, "translation": output[0], "language": "Swahili"}
    except Exception as e:
        logger.error(f"[TRANSLATION] Kiswahili translation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Kiswahili translation failed: {e}")

# ----------------------------
# Include Routers
# ----------------------------
auth.logger.propagate = True
dictionary_logger = logging.getLogger("dictionary")
dictionary_logger.propagate = True

app.include_router(auth.router)
app.include_router(dictionary.router)
