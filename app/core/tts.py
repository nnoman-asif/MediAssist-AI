import asyncio
import re
import time
import uuid

import edge_tts

from app import config
from app.utils.logger import get_logger

log = get_logger(__name__)

VOICES = {
    "en": "en-US-GuyNeural",
    "ur": "ur-PK-AsadNeural",
}

_URDU_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")


def detect_language(text: str) -> str:
    urdu_chars = len(_URDU_RE.findall(text))
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return "en"
    return "ur" if urdu_chars / total_alpha > 0.3 else "en"


async def _synthesize_async(text: str, voice: str, output_path: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def synthesize(text: str, language: str | None = None) -> dict:
    """Convert text to speech.

    Returns dict with keys: audio_path, language, latency_ms.
    """
    if not language:
        language = detect_language(text)

    voice = VOICES.get(language, VOICES["en"])
    output_path = str(config.TEMP_DIR / f"tts_{uuid.uuid4().hex[:8]}.mp3")

    log.info("TTS [%s / %s] generating ...", language, voice)
    start = time.perf_counter()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pool.submit(lambda: asyncio.run(_synthesize_async(text, voice, output_path))).result()
    else:
        asyncio.run(_synthesize_async(text, voice, output_path))

    latency = (time.perf_counter() - start) * 1000
    log.info("TTS done in %.0f ms -> %s", latency, output_path)

    return {"audio_path": output_path, "language": language, "latency_ms": round(latency, 1)}
