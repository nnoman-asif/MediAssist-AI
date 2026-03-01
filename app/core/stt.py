import os
import time
from typing import Optional

import librosa
import torch
import torch.nn.functional as F
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from app import config
from app.utils.logger import get_logger

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

log = get_logger(__name__)

_model = None
_processor = None
_pipe = None
_device = None
_torch_dtype = None

MODEL_ID = "openai/whisper-large-v3"
SUPPORTED_LANGUAGES = {"ur", "en"}


def _get_device():
    if torch.cuda.is_available():
        log.info("CUDA available: %s (VRAM: %.1f GB)",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)
        return "cuda:0", torch.float16
    log.info("CUDA not available, using CPU")
    return "cpu", torch.float32


def load_model() -> None:
    """Download (if needed) and load Whisper into GPU memory."""
    global _model, _processor, _pipe, _device, _torch_dtype
    if _pipe is not None:
        return

    _device, _torch_dtype = _get_device()
    model_dir = str(config.MODELS_DIR / "whisper-large-v3")

    log.info("Step 1/3: Loading Whisper model weights ...")
    start = time.perf_counter()

    try:
        _model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=_torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=model_dir,
            device_map=_device,
        )
        log.info("Step 1/3: Model loaded on %s", _device)
    except Exception:
        log.error("Failed to load Whisper model", exc_info=True)
        raise

    try:
        log.info("Step 2/3: Loading processor ...")
        _processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=model_dir)
    except Exception:
        log.error("Failed to load Whisper processor", exc_info=True)
        raise

    try:
        log.info("Step 3/3: Building pipeline ...")
        _pipe = pipeline(
            "automatic-speech-recognition",
            model=_model,
            tokenizer=_processor.tokenizer,
            feature_extractor=_processor.feature_extractor,
            torch_dtype=_torch_dtype,
        )
    except Exception:
        log.error("Failed to create ASR pipeline", exc_info=True)
        raise

    elapsed = time.perf_counter() - start
    log.info("Whisper loaded successfully in %.1f s on %s", elapsed, _device)


def detect_language(audio_path: str) -> str:
    """Detect the spoken language from an audio file.

    Compares English vs Urdu probabilities from the decoder's language
    prediction logits and returns whichever is higher.
    """
    if _model is None or _processor is None:
        load_model()

    audio_array, _ = librosa.load(audio_path, sr=16000)
    input_features = _processor(
        audio_array, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(_device, dtype=_torch_dtype)

    decoder_input = torch.tensor(
        [[_model.config.decoder_start_token_id]], device=_device
    )

    with torch.no_grad():
        logits = _model(input_features, decoder_input_ids=decoder_input).logits

    probs = F.softmax(logits[0, -1], dim=-1)

    en_token = _processor.tokenizer.convert_tokens_to_ids("<|en|>")
    ur_token = _processor.tokenizer.convert_tokens_to_ids("<|ur|>")

    en_prob = probs[en_token].item()
    ur_prob = probs[ur_token].item()

    detected = "ur" if ur_prob > en_prob else "en"
    log.info("Language probs: en=%.4f, ur=%.4f -> %s", en_prob, ur_prob, detected)
    return detected


def transcribe(audio_path: str, language: Optional[str] = None) -> dict:
    """Transcribe an audio file with automatic language detection.

    Returns dict with keys: text, language, latency_ms.
    """
    if _pipe is None:
        load_model()

    if not language:
        language = detect_language(audio_path)

    log.info("Transcribing %s (language=%s) ...", audio_path, language)
    start = time.perf_counter()
    result = _pipe(
        audio_path,
        generate_kwargs={"language": language},
        return_timestamps=False,
    )
    latency = (time.perf_counter() - start) * 1000

    text = result.get("text", "").strip()

    log.info("STT [%s] %.0f ms: %s", language, latency, text[:80])
    return {"text": text, "language": language, "latency_ms": round(latency, 1)}
