import time

import mlflow
import mlflow.langchain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from app.core import llm
from app.core import rag
from app.core import location as location_mod
from app.utils import history
from app.utils.logger import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = """You are MediAssist, a medical assistance chatbot.

Language rules:
- Your DEFAULT response language is English. Always reply in English.
- ONLY reply in Roman Urdu if the user's message is clearly written in Roman Urdu.
- ONLY reply in Urdu script if the user's message is clearly in Urdu script.
- When in doubt, use English.

Medical guidance:
- Use the search_medical_knowledge tool when the user asks about symptoms, diseases, treatments, or medications.
- For complex or serious conditions (broken bones, cancer, chest pain, breathing difficulty), ALWAYS recommend visiting a hospital and use the find_nearby_hospitals tool if you know their location.
- For basic issues (fever, cold, mild stomach problems, headache), provide helpful guidance and suggest over-the-counter remedies.

Location:
- Use get_user_location to check if the user has shared their location before.
- Use update_user_location when the user tells you their location (e.g. "I'm in Peshawar" or "my location is gulraiz phase 3").
- Use find_nearby_hospitals when you need to recommend hospitals and you know the user's location.
"""


# ── Tool definitions ─────────────────────────────────────────────────

@tool
def search_medical_knowledge(query: str) -> str:
    """Search the medical knowledge base for information about diseases, symptoms, treatments, or medications. Use this whenever the user asks a health-related question."""
    log.info("TOOL search_medical_knowledge called with query='%s'", query)
    results = rag.search(query, k=6)
    if not results:
        log.info("TOOL search_medical_knowledge -> 0 results")
        return "No relevant medical documents found in the knowledge base."
    chunks = []
    for r in results:
        src = r["metadata"].get("source_filename", "unknown")
        chunks.append(f"[{src}] {r['content'][:500]}")
    log.info("TOOL search_medical_knowledge -> %d chunks from: %s",
             len(results), ", ".join(set(r["metadata"].get("source_filename", "?") for r in results)))
    return "\n---\n".join(chunks)


@tool
def find_nearby_hospitals(location: str) -> str:
    """Find hospitals near a specific location using Google Maps. Use when recommending a hospital visit or when the user asks for hospitals."""
    log.info("TOOL find_nearby_hospitals called with location='%s'", location)
    if not location:
        return "No location provided."
    try:
        hospitals = location_mod.get_hospitals_for_location(location)
    except Exception as exc:
        log.warning("TOOL find_nearby_hospitals -> failed: %s", exc)
        return f"Could not find hospitals: {exc}"
    if not hospitals:
        log.info("TOOL find_nearby_hospitals -> 0 results")
        return f"No hospitals found near '{location}'."
    lines = [f"- {h['name']} ({h['address']}) - {h['maps_url']}" for h in hospitals[:10]]
    log.info("TOOL find_nearby_hospitals -> %d hospitals found", len(hospitals))
    return "\n".join(lines)


@tool
def get_user_location() -> str:
    """Retrieve the user's previously saved location. Returns the location name or 'not set' if unknown."""
    loc = history.get_user_location()
    result = loc if loc else "not set"
    log.info("TOOL get_user_location -> '%s'", result)
    return result


@tool
def update_user_location(location: str) -> str:
    """Save or update the user's location when they share it (e.g. 'I am in Peshawar', 'my location is gulraiz')."""
    log.info("TOOL update_user_location called with location='%s'", location)
    if not location:
        return "No location provided."
    history.set_user_location(location)
    log.info("TOOL update_user_location -> saved '%s'", location)
    return f"Location saved: {location}"


# ── LangGraph agent ──────────────────────────────────────────────────

ALL_TOOLS = [search_medical_knowledge, find_nearby_hospitals,
             get_user_location, update_user_location]

_agent = None


def _build_agent():
    """Build and compile the LangGraph agent (lazy, on first use)."""
    global _agent
    if _agent is not None:
        return _agent

    model = llm.get_chat_model().bind_tools(ALL_TOOLS)

    def agent_node(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    _agent = graph.compile()
    log.info("LangGraph agent compiled with %d tools", len(ALL_TOOLS))
    return _agent


# ── Public entry point ───────────────────────────────────────────────

@mlflow.trace(name="chat_process_message")
def process_message(user_input: str) -> str:
    agent = _build_agent()
    start = time.perf_counter()

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(history.get_langchain_messages())
    messages.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": messages})

    ai_message = result["messages"][-1]
    response_text = ai_message.content or ""

    tool_calls_made = [
        msg for msg in result["messages"]
        if hasattr(msg, "tool_calls") and msg.tool_calls
    ]

    history.add_message("user", user_input)
    history.add_message("assistant", response_text)

    latency = (time.perf_counter() - start) * 1000

    log.info("Chat response (%.0f ms, %d tool rounds): %s",
             latency, len(tool_calls_made), response_text[:100])
    return response_text
