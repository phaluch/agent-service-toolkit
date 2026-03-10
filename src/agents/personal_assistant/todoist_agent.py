"""Todoist sub-agent — handles task management via the Todoist MCP integration."""

import logging
from datetime import datetime

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from agents.personal_assistant.prompts import TODOIST_SYSTEM_PROMPT
from agents.personal_assistant.state import AgentState
from agents.personal_assistant.todoist_tools import get_todoist_tools
from core import get_model, settings

logger = logging.getLogger(__name__)


async def respond(state: AgentState, config: RunnableConfig) -> AgentState:
    model_name = (
        config["configurable"].get("todoist_model")
        or config["configurable"].get("model")
        or settings.DEFAULT_MODEL
    )
    m = get_model(model_name)

    tools = await get_todoist_tools()
    if tools:
        m = m.bind_tools(tools)

    context = state.get("retrieved_context", "")
    context_section = f"\n## Context from your knowledge base:\n{context}" if context else ""
    system = TODOIST_SYSTEM_PROMPT.format(
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

    tools = await get_todoist_tools()
    tool_map = {t.name: t for t in tools}

    results = []
    for call in last_msg.tool_calls:
        tool = tool_map.get(call["name"])
        if tool is None:
            logger.warning(f"Tool {call['name']} not found")
            continue
        try:
            result = await tool.ainvoke(call["args"], config)
            results.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        except Exception as e:
            logger.error(f"Todoist tool {call['name']} failed: {e}")
            results.append(ToolMessage(content=f"Tool call failed: {e}", tool_call_id=call["id"]))

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

todoist_agent = graph.compile()
