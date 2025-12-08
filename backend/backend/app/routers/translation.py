import logging
import ctranslate2
import sentencepiece as spm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from fastapi import APIRouter, HTTPException

# Import dictionary router / functions
from . import dictionary

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/translate", tags=["Translation"])

# Globals for models
luo_translator, luo_sp = None, None
luo_hf_model, luo_hf_tokenizer = None, None  # Hugging Face Luo model
sw_model, sw_tokenizer = None, None


def load_translation_models():
    """Load Luo (CTranslate2 and Hugging Face) and Kiswahili (Hugging Face) models."""
    global luo_translator, luo_sp, luo_hf_model, luo_hf_tokenizer, sw_model, sw_tokenizer

    # -----------------------------
    # Luo model (CTranslate2)
    # -----------------------------
    try:
        luo_model_dir = "app/training/models/luo_translator"
        luo_sp_model = f"{luo_model_dir}/sp.model"

        luo_translator = ctranslate2.Translator(luo_model_dir)
        luo_sp = spm.SentencePieceProcessor()
        luo_sp.load(luo_sp_model)

        logger.info("✅ Luo model loaded successfully (CTranslate2).")
    except Exception as e:
        logger.error(f"❌ Failed to load Luo CTranslate2 model: {e}")
        luo_translator, luo_sp = None, None

    # -----------------------------
    # Luo model (Hugging Face Transformers)
    # -----------------------------
    try:
        luo_hf_model_dir = "app/training/models/luo_translator"
        luo_hf_tokenizer = AutoTokenizer.from_pretrained(luo_hf_model_dir)
        luo_hf_model = AutoModelForSeq2SeqLM.from_pretrained(luo_hf_model_dir)

        logger.info("✅ Luo model loaded successfully (Hugging Face).")
    except Exception as e:
        logger.error(f"❌ Failed to load Luo Hugging Face model: {e}")
        luo_hf_model, luo_hf_tokenizer = None, None

    # -----------------------------
    # Kiswahili model
    # -----------------------------
    try:
        sw_model_name = "Helsinki-NLP/opus-mt-en-sw"
        sw_tokenizer = AutoTokenizer.from_pretrained(sw_model_name)
        sw_model = AutoModelForSeq2SeqLM.from_pretrained(sw_model_name)

        logger.info("✅ Kiswahili model loaded successfully (Hugging Face).")
    except Exception as e:
        logger.error(f"❌ Failed to load Kiswahili model: {e}")
        sw_tokenizer, sw_model = None, None


def translate_luo(text: str) -> str:
    """Translate English → Luo using Hugging Face model (primary) or CTranslate2 (fallback)."""
    # Try Hugging Face model first
    if luo_hf_model is not None and luo_hf_tokenizer is not None:
        inputs = luo_hf_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = luo_hf_model.generate(**inputs)
        return luo_hf_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Fallback: CTranslate2
    if luo_translator is None or luo_sp is None:
        raise RuntimeError("Luo model is not available.")
    tokens = luo_sp.encode(text, out_type=str)
    results = luo_translator.translate_batch([tokens])
    translated_tokens = results[0].hypotheses[0]
    return luo_sp.decode(translated_tokens)


def translate_kiswahili(text: str) -> str:
    """Translate English → Kiswahili using Hugging Face model."""
    if sw_tokenizer is None or sw_model is None:
        raise RuntimeError("Kiswahili model is not available.")

    inputs = sw_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = sw_model.generate(**inputs)
    return sw_tokenizer.decode(outputs[0], skip_special_tokens=True)


def translate_with_fallback(text: str, target_lang: str) -> str:
    """Translate with AI, fallback to dictionary if AI fails or word not found."""
    try:
        if target_lang == "luo":
            return translate_luo(text)
        elif target_lang == "sw":
            return translate_kiswahili(text)
        else:
            raise ValueError(f"Unsupported language code: {target_lang}")
    except Exception as e:
        logger.warning(f"⚠️ AI translation failed for '{text}' → {target_lang}: {e}")
        # fallback: use dictionary router logic word by word
        translated_words = []
        for word in text.lower().split():
            meaning = dictionary.get_word_translation(word, target_lang)
            translated_words.append(meaning if meaning else word)
        return " ".join(translated_words)


# -------------------------------
# API Endpoint
# -------------------------------
@router.post("/")
def api_translate(payload: dict):
    """
    Translate text into Luo or Kiswahili.
    Example request:
    {
        "text": "hello doctor",
        "lang": "luo"
    }
    """
    text = payload.get("text")
    lang = payload.get("lang")

    if not text or not lang:
        raise HTTPException(status_code=400, detail="Both 'text' and 'lang' are required.")

    try:
        result = translate_with_fallback(text, lang)
        return {"input": text, "lang": lang, "translation": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
