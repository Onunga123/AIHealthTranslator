from pydantic import BaseModel, Field

# --- Auth ---
class RegisterIn(BaseModel):
    username: str
    email: str
    password: str

class RegisterOut(BaseModel):
    id: int
    username: str
    email: str

class LoginIn(BaseModel):
    username: str
    password: str

class LoginOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

# --- Translation ---
class TranslationIn(BaseModel):
    text: str = Field(..., example="The patient is suffering from malaria")
    target_language: str = Field(..., example="sw")

class TranslationOut(BaseModel):
    translated_text: str
    target_language: str

# --- Speech ---
class SpeechToTextIn(BaseModel):
    audio_file: str

class SpeechToTextOut(BaseModel):
    text: str

class TextToSpeechIn(BaseModel):
    text: str

class TextToSpeechOut(BaseModel):
    audio_file: str

# --- Languages ---
class LanguageOut(BaseModel):
    code: str
    name: str
