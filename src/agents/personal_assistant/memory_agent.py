"""Memory sub-agent — handles explicit knowledge base queries and updates."""

import logging
from datetime import datetime

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from agents.personal_assistant.knowledge_store import retrieve_facts, store_facts
from agents.personal_assistant.prompts import MEMORY_SYSTEM_PROMPT
from agents.personal_assistant.state import AgentState
from core import get_model, settings

logger = logging.getLogger(__name__)


@tool
async def search_knowledge(query: str) -> str:
    """Search the personal knowledge base for information about a person, project, or topic."""
    docs = await retrieve_facts(query, k=10)
    if not docs:
        return "No relevant facts found."
    return "\n".join(
        f"[{d.metadata.get('entity_type', 'general')}] "
        f"(stored: {d.metadata.get('insertion_time', 'unknown')}) {d.page_content}"
        for d in docs
    )


@tool
async def save_fact(content: str, entity_type: str, entity_name: str) -> str:
    """Store a new fact in the personal knowledge base.

    Args:
        content: Full self-contained sentence describing the fact.
        entity_type: One of: person, project, process, general.
        entity_name: Primary entity label, e.g. 'Paulo' or 'Project Alpha'.
    """
    await store_facts([{"content": content, "entity_type": entity_type, "entity_name": entity_name}])
    return f"Stored: {content}"


MEMORY_TOOLS = [search_knowledge, save_fact]


async def respond(state: AgentState, config: RunnableConfig) -> AgentState:
    model_name = (
        config["configurable"].get("memory_model")
        or config["configurable"].get("model")
        or settings.DEFAULT_MODEL
    )
    m = get_model(model_name).bind_tools(MEMORY_TOOLS)

    context = state.get("retrieved_context", "")
    context_section = f"\n## Already retrieved context:\n{context}" if context else ""
    system = MEMORY_SYSTEM_PROMPT.format(
        date=datetime.now().strftime("%B %d, %Y"),
        context_section=context_section,
    )

    messages = [SystemMessage(content=system)] + state["messages"]
    response = await m.ainvoke(messages, config)
    return {"messages": [response]}


async def execute_tools(state: AgentState, config: RunnableConfig) -> AgentState:
    last_msg = state["messages"][-1]
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return {}

    tool_map = {t.name: t for t in MEMORY_TOOLS}
    results = []
    for call in last_msg.tool_calls:
        t = tool_map.get(call["name"])
        if t is None:
            continue
        try:
            result = await t.ainvoke(call["args"], config)
            results.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        except Exception as e:
            results.append(ToolMessage(content=f"Tool failed: {e}", tool_call_id=call["id"]))

    return {"messages": results}


def route_after_respond(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "execute_tools"
    return END


graph = StateGraph(AgentState)
graph.add_node("respond", respond)
graph.add_node("execute_tools", execute_tools)
graph.set_entry_point("respond")
graph.add_conditional_edges("respond", route_after_respond, ["execute_tools", END])
graph.add_edge("execute_tools", "respond")

memory_agent = graph.compile()
