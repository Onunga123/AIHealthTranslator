from pydantic import BaseModel

class SpeechToTextOut(BaseModel):
    transcript: str
    language: str


class TextToSpeechOut(BaseModel):
    audio_b64: str
    sample_rate: int
    channels: int
    sample_width: int
