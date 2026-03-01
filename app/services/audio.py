import re
import time
import uuid

import soundfile as sf
import numpy as np
import mlflow

from app import config

from app.core import stt, tts
from app.services import chat
from app.utils.logger import get_logger

log = get_logger(__name__)


def _save_upload(audio_tuple: tuple) -> str:
    """Save Gradio audio input (sample_rate, numpy_array) to a temp WAV file."""
    sample_rate, data = audio_tuple

    if data.dtype != np.int16:
        if np.issubdtype(data.dtype, np.floating):
            data = (data * 32767).astype(np.int16)
        else:
            data = data.astype(np.int16)

    path = str(config.TEMP_DIR / f"input_{uuid.uuid4().hex[:8]}.wav")
    sf.write(path, data, sample_rate)
    return path


@mlflow.trace(name="audio_process_audio")
def process_audio(audio_input: tuple) -> dict:
    """Full audio pipeline: STT -> chat -> TTS.

    Args:
        audio_input: Gradio audio tuple (sample_rate, numpy_array)

    Returns dict with keys: text_response, audio_path, stt_text, stt_latency_ms, tts_latency_ms.
    """
    total_start = time.perf_counter()

    audio_path = _save_upload(audio_input)
    log.info("Saved user audio to %s", audio_path)

    stt_result = stt.transcribe(audio_path)
    stt_text = stt_result["text"]

    if not stt_text.strip():
        log.warning("STT returned empty transcription")
        return {
            "text_response": "I couldn't understand the audio. Could you please try again?",
            "audio_path": None,
            "stt_text": "",
            "stt_latency_ms": stt_result["latency_ms"],
            "tts_latency_ms": 0,
        }

    text_response = chat.process_message(stt_text)

    tts_text = re.sub(r"https?://\S+", "", text_response).strip()
    tts_result = tts.synthesize(tts_text)

    total_latency = (time.perf_counter() - total_start) * 1000

    log.info("Audio pipeline complete (%.0f ms): STT='%s' -> Response='%s'",
             total_latency, stt_text[:60], text_response[:60])

    return {
        "text_response": text_response,
        "audio_path": tts_result["audio_path"],
        "stt_text": stt_text,
        "stt_latency_ms": stt_result["latency_ms"],
        "tts_latency_ms": tts_result["latency_ms"],
    }
