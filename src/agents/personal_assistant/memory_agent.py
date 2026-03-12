"""Memory sub-agent — handles explicit knowledge base queries and updates."""

import logging
from datetime import datetime

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from agents.personal_assistant.graph_store import get_entity_neighborhood, search_entities
from agents.personal_assistant.knowledge_store import retrieve_facts, store_facts
from agents.personal_assistant.prompts import MEMORY_SYSTEM_PROMPT
from agents.personal_assistant.state import AgentState
from core import get_model, settings

logger = logging.getLogger(__name__)


@tool
async def search_knowledge(query: str, include_history: bool = False) -> str:
    """Search the personal knowledge base for information about a person, project, or topic.

    Args:
        query: The search query.
        include_history: If True, also return invalidated (historical) facts, annotated
            with their invalidation timestamp. Use when the user asks about past state,
            previous roles, or how something has changed over time.
    """
    docs = await retrieve_facts(query, k=10, include_history=include_history)
    if not docs:
        return "No relevant facts found."
    lines = []
    for d in docs:
        is_valid = d.metadata.get("is_valid", 1)
        timestamp = d.metadata.get("insertion_time", "unknown")
        prefix = f"[{d.metadata.get('entity_type', 'general')}] (stored: {timestamp})"
        if not is_valid:
            invalidated_at = d.metadata.get("invalidated_at", "unknown")
            prefix += f" [HISTORICAL — invalidated: {invalidated_at}]"
        lines.append(f"{prefix} {d.page_content}")
    return "\n".join(lines)


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


@tool
async def search_graph(entity_name: str) -> str:
    """Search the knowledge graph for entities matching a name.

    Use this to check whether a person, project, or other entity exists in the graph.
    Pass a name or partial name (e.g. 'Salave', 'Paulo', 'Alpha') — NOT a full sentence.

    Args:
        entity_name: Name or partial name to search for (case-insensitive substring match).
    """
    return await search_entities(entity_name, limit=10)


@tool
async def get_graph_neighborhood(entity_name: str, include_history: bool = False) -> str:
    """Get all relationships for a named entity from the knowledge graph.

    Args:
        entity_name: Exact entity name (e.g. 'Paulo', 'Project Alpha').
        include_history: If True, also return invalidated (historical) relationships,
            annotated with their invalidation timestamp.
    """
    return await get_entity_neighborhood(entity_name, include_history)


MEMORY_TOOLS = [search_knowledge, save_fact, search_graph, get_graph_neighborhood]


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
