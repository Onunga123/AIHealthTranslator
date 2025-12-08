import base64
import io
import math
import struct
import wave

from fastapi import APIRouter, UploadFile, File, Form
from ..schemas.speech import SpeechToTextOut, TextToSpeechOut

router = APIRouter(prefix="/speech", tags=["Speech"])

@router.post("/to-text", response_model=SpeechToTextOut)
async def speech_to_text(file: UploadFile = File(...), language: str = Form("en")):
    # Mock STT: returns a placeholder transcript
    # Reads the bytes just to "use" the uploaded file
    await file.read()
    return SpeechToTextOut(transcript="(mock) audio transcribed successfully", language=language)


def generate_beep_wav_base64(text: str, duration_sec: float = 0.5, freq_hz: int = 440, sample_rate: int = 16000) -> tuple[str, int, int, int]:
    """Generate a simple sine beep and return as base64 WAV. (No external deps)"""
    n_samples = int(duration_sec * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        channels = 1
        sampwidth = 2  # 16-bit
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)

        max_amp = 32767
        for i in range(n_samples):
            # simple sine wave
            val = int(max_amp * 0.3 * math.sin(2 * math.pi * freq_hz * (i / sample_rate)))
            wf.writeframes(struct.pack("<h", val))

    audio_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return audio_b64, sample_rate, channels, 2


@router.post("/to-audio", response_model=TextToSpeechOut)
async def text_to_speech(text: str = Form(...), language: str = Form("en")):
    # Mock TTS: returns a small beep WAV as base64 (works offline)
    audio_b64, sample_rate, channels, sample_width = generate_beep_wav_base64(text)
    return TextToSpeechOut(
        audio_b64=audio_b64,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
    )
