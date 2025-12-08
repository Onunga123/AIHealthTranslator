from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from transformers import pipeline
from ..database import get_db
from ..schemas.translation import TranslationCreate, TranslationOut
from ..models.translation_log import TranslationLog
from ..models.dictionary_term import DictionaryTerm
from .auth import get_current_user
from ..models.user import User

router = APIRouter(prefix="/translation", tags=["Translation"])

# Load Hugging Face translator (English, Swahili, Luo, etc.)
# MarianMT works for many languages, but Luo support is limited
# For Luo, we’ll fallback more often to the dictionary
try:
    hf_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-swa")
except Exception as e:
    hf_translator = None
    print(f"⚠️ Hugging Face translator not loaded: {e}")

# Built-in fallback dictionary
SIMPLE_DICTIONARY = {
    ("en", "sw"): {
        "malaria": "malaria",
        "fever": "homa",
        "headache": "maumivu ya kichwa",
        "hospital": "hospitali",
        "doctor": "daktari",
        "medicine": "dawa",
        "pain": "maumivu",
        "water": "maji",
    },
    ("en", "luo"): {
        "malaria": "malaria",
        "fever": "chweya",
        "headache": "tich weche",
        "hospital": "osiep daktari",
        "doctor": "daktari",
        "medicine": "rem",
        "pain": "rembo",
        "water": "pi",
    },
    ("sw", "en"): {
        "homa": "fever",
        "dawa": "medicine",
        "maji": "water",
        "daktari": "doctor",
        "hospitali": "hospital",
        "maumivu": "pain",
    },
    ("luo", "en"): {
        "chweya": "fever",
        "rem": "medicine",
        "pi": "water",
    },
}


def ai_translate(text: str, src: str, tgt: str) -> str | None:
    """Try Hugging Face AI translation first"""
    if not hf_translator:
        return None

    try:
        # MarianMT models use specific language codes (example: en↔sw)
        if (src, tgt) == ("en", "sw"):
            result = hf_translator(text, src_lang="en", tgt_lang="sw")
            return result[0]["translation_text"]
        elif (src, tgt) == ("sw", "en"):
            result = hf_translator(text, src_lang="sw", tgt_lang="en")
            return result[0]["translation_text"]
        else:
            # Luo not well-supported → rely on dictionary fallback
            return None
    except Exception as e:
        print(f"❌ AI translation failed: {e}")
        return None


def dictionary_fallback(text: str, src: str, tgt: str, db: Session) -> str:
    """Fallback to DB dictionary and static dictionary"""
    words = text.lower().split()
    translated_words: list[str] = []
    for w in words:
        # Check DB dictionary
        term = db.query(DictionaryTerm).filter(DictionaryTerm.term_en == w).first()
        if term:
            if tgt == "sw" and term.term_sw:
                translated_words.append(term.term_sw)
                continue
            if tgt == "luo" and term.term_luo:
                translated_words.append(term.term_luo)
                continue
        # Check static dictionary
        mapping = SIMPLE_DICTIONARY.get((src, tgt), {})
        translated_words.append(mapping.get(w, w))
    return " ".join(translated_words)


@router.post("/text", response_model=TranslationOut)
def translate_text(
    payload: TranslationCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    src = payload.source_lang.lower()
    tgt = payload.target_lang.lower()

    # Step 1: Try Hugging Face AI first
    translated = ai_translate(payload.text, src, tgt)

    # Step 2: Fallback to dictionary if AI fails
    if not translated or translated.strip() == payload.text.strip():
        translated = dictionary_fallback(payload.text, src, tgt, db)

    # Log translation
    log = TranslationLog(
        user_id=user.id if user else None,
        source_text=payload.text,
        translated_text=translated,
        source_lang=src,
        target_lang=tgt,
    )
    db.add(log)
    db.commit()

    return TranslationOut(
        source_text=payload.text,
        translated_text=translated,
        source_lang=src,
        target_lang=tgt,
    )
