# backend/app/main.py
from fastapi import FastAPI, HTTPException
from app.database import init_db, SessionLocal
from app.scripts.seed_dictionary import seed_dictionary_terms
from app.routers import auth, user, translation, speech, languages, health, dictionary, words
from app.models import Dictionary  # Ensure this exists

# -----------------------------
# Transformers imports
# -----------------------------
import torch
from transformers import MarianMTModel, MarianTokenizer

# -----------------------------
# App setup
# -----------------------------
from fastapi.middleware.cors import CORSMiddleware  # ✅ added import

app = FastAPI(title="AI Health Translator API", version="1.0.0")

# ✅ added CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Initialize DB and seed admin/dictionary
# -----------------------------
init_db()
seed_dictionary_terms()
print("✅ Database initialized and dictionary seeded.")

# -----------------------------
# Load Translation Models
# -----------------------------
device = torch.device("cpu")

# Luo (custom fine-tuned)
try:
    LUO_MODEL_PATH = "./backend/training/luo_model"
    luo_tokenizer = MarianTokenizer.from_pretrained(LUO_MODEL_PATH)
    luo_model = MarianMTModel.from_pretrained(LUO_MODEL_PATH).to(device)
    print("✅ Luo model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load Luo model: {e}")
    luo_tokenizer, luo_model = None, None

# Kiswahili (pretrained)
try:
    SWA_MODEL_NAME = "Helsinki-NLP/opus-mt-en-swa"
    swa_tokenizer = MarianTokenizer.from_pretrained(SWA_MODEL_NAME)
    swa_model = MarianMTModel.from_pretrained(SWA_MODEL_NAME).to(device)
    print("✅ Kiswahili model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load Kiswahili model: {e}")
    swa_tokenizer, swa_model = None, None


def translate_text(model, tokenizer, text: str) -> str:
    """Helper to translate English -> Target language."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def lookup_term(term: str, target_lang: str):
    """Fallback: Check dictionary if AI fails."""
    db = SessionLocal()
    try:
        entry = db.query(Dictionary).filter(Dictionary.term == term).first()
        if entry:
            if target_lang.lower() == "luo" and entry.luo:
                return entry.luo
            elif target_lang.lower() == "swahili" and entry.swahili:
                return entry.swahili
        return None
    finally:
        db.close()


# -----------------------------
# AI Translation Endpoints
# -----------------------------
@app.post("/translate/luo")
def translate_to_luo(payload: dict):
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    # Try AI first
    if luo_model and luo_tokenizer:
        try:
            ai_translation = translate_text(luo_model, luo_tokenizer, text)
            if ai_translation and ai_translation.strip():
                return {"translation": ai_translation, "source": "AI"}
        except Exception as e:
            print(f"⚠️ Luo AI translation failed: {e}")

    # Fallback to dictionary
    dictionary_translation = lookup_term(text, target_lang="luo")
    if dictionary_translation:
        return {"translation": dictionary_translation, "source": "dictionary"}

    raise HTTPException(status_code=404, detail="Translation not found in AI or dictionary")


@app.post("/translate/swahili")
def translate_to_swahili(payload: dict):
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    # Try AI first
    if swa_model and swa_tokenizer:
        try:
            ai_translation = translate_text(swa_model, swa_tokenizer, text)
            if ai_translation and ai_translation.strip():
                return {"translation": ai_translation, "source": "AI"}
        except Exception as e:
            print(f"⚠️ Kiswahili AI translation failed: {e}")

    # Fallback to dictionary
    dictionary_translation = lookup_term(text, target_lang="swahili")
    if dictionary_translation:
        return {"translation": dictionary_translation, "source": "dictionary"}

    raise HTTPException(status_code=404, detail="Translation not found in AI or dictionary")


# -----------------------------
# Existing Routers
# -----------------------------
app.include_router(auth.router)
app.include_router(user.router)
app.include_router(translation.router)
app.include_router(speech.router)
app.include_router(languages.router)
app.include_router(health.router)
app.include_router(dictionary.router)
app.include_router(words.router)


@app.get("/")
def root():
    return {"message": "AI Health Translator API is running!"}
