"""Memory sub-agent — handles explicit knowledge base queries and updates."""

import logging
from datetime import datetime

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from agents.personal_assistant.graphiti_store import (
    add_episode,
    get_entity_context,
    search_memory,
    search_nodes,
)
from agents.personal_assistant.prompts import MEMORY_SYSTEM_PROMPT
from agents.personal_assistant.state import AgentState
from core import get_model, settings

logger = logging.getLogger(__name__)


@tool
async def search_knowledge(query: str, include_history: bool = False) -> str:
    """Search the personal knowledge graph for facts about a person, project, or topic.

    Performs hybrid search (semantic + keyword + graph traversal).

    Args:
        query: Natural language search query.
        include_history: If True, also return superseded (historical) facts annotated
            with their invalidation time. Use when the user asks about past state or
            how something has changed over time.
    """
    result = await search_memory(query, num_results=10, include_history=include_history)
    return result or "No relevant facts found."


@tool
async def get_entity(entity_name: str) -> str:
    """Get all current facts and relationships for a specific named entity.

    Use this to retrieve a complete picture of a person, project, or organisation.

    Args:
        entity_name: The entity's canonical name (e.g. 'Pedro', 'Project Alpha').
    """
    return await get_entity_context(entity_name)


@tool
async def find_entities(query: str) -> str:
    """Search for entities (people, projects, organisations, topics) by name or description.

    Args:
        query: Name or partial name to search for.
    """
    return await search_nodes(query, limit=10)


@tool
async def remember(content: str) -> str:
    """Explicitly store a new fact or piece of information in the knowledge graph.

    Use when the user asks you to remember something. Graphiti will extract
    entities and relationships automatically.

    Args:
        content: Full self-contained statement to remember
            (e.g. 'Pedro prefers dark mode and uses VSCode').
    """
    await add_episode(content)
    return f"Stored: {content}"


MEMORY_TOOLS = [search_knowledge, get_entity, find_entities, remember]


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
