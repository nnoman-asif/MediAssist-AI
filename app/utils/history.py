import json
from datetime import datetime, timezone
from typing import Optional

from app.config import DATA_DIR

_HISTORY_FILE = DATA_DIR / "chat_history.json"
_MAX_MESSAGES = 12  # 6 user + 6 assistant turns


def _load() -> list[dict]:
    if _HISTORY_FILE.exists():
        try:
            return json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save(history: list[dict]) -> None:
    _HISTORY_FILE.write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def add_message(role: str, content: str) -> list[dict]:
    history = _load()
    history.append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    if len(history) > _MAX_MESSAGES:
        history = history[-_MAX_MESSAGES:]
    _save(history)
    return history


def get_langchain_messages():
    """Return history as LangChain message objects."""
    from langchain_core.messages import AIMessage, HumanMessage
    msgs = []
    for e in _load():
        if e["role"] == "user":
            msgs.append(HumanMessage(content=e["content"]))
        elif e["role"] == "assistant":
            msgs.append(AIMessage(content=e["content"]))
    return msgs


def clear() -> None:
    _save([])


def get_user_location() -> Optional[str]:
    """Read persisted user location, if any."""
    loc_file = DATA_DIR / "user_location.txt"
    if loc_file.exists():
        return loc_file.read_text(encoding="utf-8").strip() or None
    return None


def set_user_location(location: str) -> None:
    loc_file = DATA_DIR / "user_location.txt"
    loc_file.write_text(location, encoding="utf-8")
