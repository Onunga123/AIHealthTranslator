from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import logging

from ..database import get_db
from ..models.dictionary_term import DictionaryTerm
from .auth import get_current_user
from ..models.user import User

from .translation import translate_with_fallback

router = APIRouter(prefix="/dictionary", tags=["Health Dictionary"])
logger = logging.getLogger("dictionary")


@router.get("/{term_en}")
async def lookup_term(term_en: str, db: Session = Depends(get_db)):
    # 1. Try AI translation first
    try:
        logger.info(f"Trying AI translation for: {term_en}")
        ai_result = {
            "term_sw": translate_with_fallback(term_en, "sw"),
            "term_luo": translate_with_fallback(term_en, "luo")
        }
        if ai_result and (ai_result.get("term_sw") or ai_result.get("term_luo")):
            logger.info("AI translation successful")
            return {
                "term_en": term_en.lower(),
                "term_sw": ai_result.get("term_sw"),
                "term_luo": ai_result.get("term_luo"),
                "source": "ai",
            }
    except Exception as e:
        logger.warning(f"AI translation failed: {e}")

    # 2. If AI fails, fallback to dictionary DB
    logger.info("Falling back to dictionary DB")
    term = db.query(DictionaryTerm).filter(DictionaryTerm.term_en == term_en.lower()).first()
    if term:
        return {
            "term_en": term.term_en,
            "term_sw": term.term_sw,
            "term_luo": term.term_luo,
            "source": "dictionary",
        }

    # 3. If neither AI nor DB has it, return 404
    raise HTTPException(status_code=404, detail="Term not found")
