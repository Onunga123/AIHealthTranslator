from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class AddWordRequest(BaseModel):
    term_en: str
    term_sw: str
    term_luo: str

@router.post("/add-word/")
def add_word(payload: AddWordRequest = Body(...)):
    # Here you would add the word to your database
    # For demonstration, just echo back the payload
    return {
        "message": "Word added successfully!",
        "term_en": payload.term_en,
        "term_sw": payload.term_sw,
        "term_luo": payload.term_luo
    }
