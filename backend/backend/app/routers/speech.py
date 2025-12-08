# app/routers/speech.py
import base64
import io
import tempfile
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from ..schemas.speech import SpeechToTextOut, TextToSpeechOut

# ML imports
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    from TTS.api import TTS
except Exception:
    TTS = None

router = APIRouter(prefix="/speech", tags=["Speech"])
logger = logging.getLogger("speech")

# Globals loaded at startup
whisper_model = None
tts_model = None

def load_speech_models(device: str = "cpu", whisper_size: str = "small"):
    global whisper_model, tts_model
    # Load Whisper via faster-whisper (supports CPU)
    if WhisperModel is not None:
        try:
            whisper_model = WhisperModel(model_size_or_path=whisper_size, device=device, compute_type="int8" if device!="cpu" else "float32")
            logger.info("Loaded Whisper model (size=%s) on device=%s", whisper_size, device)
        except Exception as e:
            whisper_model = None
            logger.exception("Failed to load Whisper model: %s", e)
    else:
        logger.warning("faster-whisper not available; STT won't work.")

    # Load Coqui TTS (example using default speaker)
    if TTS is not None:
        try:
            # replace "tts_models/en/ljspeech/tacotron2-DDC" with a local model you prefer
            tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
            logger.info("Loaded TTS model.")
        except Exception as e:
            tts_model = None
            logger.exception("Failed to load TTS model: %s", e)
    else:
        logger.warning("Coqui TTS not available; TTS won't work.")


@router.post("/to-text", response_model=SpeechToTextOut)
async def speech_to_text(file: UploadFile = File(...), language: str = Form("en")):
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Speech-to-text model not loaded")

    # Save temporary file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="." + (file.filename.split(".")[-1] if "." in file.filename else "wav"))
    contents = await file.read()
    tmp.write(contents)
    tmp.flush()
    tmp.close()

    # transcribe
    try:
        segments, info = whisper_model.transcribe(tmp.name, beam_size=5, language=language)
        # join segments
        transcript = " ".join([s.text for s in segments])
    except Exception as e:
        logger.exception("STT error: %s", e)
        raise HTTPException(status_code=500, detail="Speech-to-text failed")

    return SpeechToTextOut(transcript=transcript, language=language)


@router.post("/to-audio", response_model=TextToSpeechOut)
async def text_to_speech(text: str = Form(...), language: str = Form("en")):
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Text-to-speech model not loaded")

    # Synthesize TTS to a temporary WAV (Coqui TTS returns numpy array or writes file)
    try:
        # Coqui TTS can return binary audio directly:
        wav = tts_model.tts(text)
        # wav is array (float32). Convert to bytes using soundfile
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, wav, samplerate=22050, format="WAV")
        audio_bytes = buf.getvalue()
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        return TextToSpeechOut(audio_b64=audio_b64, sample_rate=22050, channels=1, sample_width=2)
    except Exception as e:
        logger.exception("TTS error: %s", e)
        raise HTTPException(status_code=500, detail="Text-to-speech failed")
