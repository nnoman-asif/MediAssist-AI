from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app import config
from app.utils.logger import get_logger

log = get_logger(__name__)

_model = None


def get_chat_model():
    """Return a LangChain ChatOpenAI model."""
    global _model
    if _model is not None:
        return _model

    _model = ChatOpenAI(
        api_key=config.OPENAI_API_KEY,
        model=config.OPENAI_MODEL,
        temperature=0.7,
        max_tokens=1024,
    )
    log.info("OpenAI model initialised (model=%s)", config.OPENAI_MODEL)
    return _model
