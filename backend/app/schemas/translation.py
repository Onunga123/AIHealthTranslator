from pydantic import BaseModel

class TranslationCreate(BaseModel):
    text: str
    source_lang: str
    target_lang: str


class TranslationOut(BaseModel):
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
