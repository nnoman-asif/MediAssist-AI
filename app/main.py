"""MediAssist AI -- entry point.

Usage:
    python -m app.main
"""
import mlflow

from app import config
from app.core import rag, stt
from app.ui.gradio_app import create_app
from app.utils.logger import get_logger

log = get_logger(__name__)


def _init_mlflow() -> None:
    db_path = config.MLRUNS_DIR / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("MediAssist")
    mlflow.langchain.autolog()
    log.info("MLflow tracking at %s (langchain autolog enabled)", tracking_uri)


def _preload_models() -> None:
    log.info("Pre-loading embedding model for RAG ...")
    rag.load_vectorstore()

    log.info("Pre-loading Whisper STT model (this may take a minute on first run) ...")
    stt.load_model()


def main() -> None:
    log.info("=" * 60)
    log.info("MediAssist AI starting up")
    log.info("=" * 60)

    try:
        _init_mlflow()
    except Exception as exc:
        log.warning("MLflow init failed (non-fatal): %s", exc)

    try:
        _preload_models()
    except Exception as exc:
        log.error("Model preloading failed: %s", exc, exc_info=True)
        log.info("Continuing without preloaded models -- they will load on first use")

    app = create_app()

    log.info("Launching Gradio on port %d", config.APP_PORT)
    app.launch(
        server_port=config.APP_PORT,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
