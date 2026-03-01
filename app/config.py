import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(f"Missing required env var: {key}")
    return val


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default)


# ── OpenAI ───────────────────────────────────────────────────────────
OPENAI_API_KEY = _require("OPENAI_API_KEY")
OPENAI_MODEL = _require("OPENAI_MODEL")

# ── Google Maps ──────────────────────────────────────────────────────
GOOGLE_MAPS_API_KEY = _require("GOOGLE_MAPS_API_KEY")

# ── Paths ────────────────────────────────────────────────────────────
MODELS_DIR = BASE_DIR / _optional("MODELS_DIR", "models")
DATA_DIR = BASE_DIR / _optional("DATA_DIR", "data")
PDFS_DIR = DATA_DIR / "pdfs"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
TEMP_DIR = BASE_DIR / "temp"
LOGS_DIR = BASE_DIR / "logs"
MLRUNS_DIR = BASE_DIR / "mlruns"

# ── App ──────────────────────────────────────────────────────────────
APP_PORT = int(_optional("APP_PORT", "7860"))
LOG_LEVEL = _optional("LOG_LEVEL", "INFO").upper()

# ── Ensure directories exist ─────────────────────────────────────────
for d in (MODELS_DIR, PDFS_DIR, VECTORSTORE_DIR, TEMP_DIR, LOGS_DIR, MLRUNS_DIR):
    d.mkdir(parents=True, exist_ok=True)
