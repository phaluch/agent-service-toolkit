"""Todoist domain-worker node — manages tasks via the Todoist MCP integration.

Receives a scoped action_input dict from the executor (goal, optional context).
Has no knowledge of other workers or the global execution plan.
Returns a confirmation string stored in action_results[action_id].
"""

import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from agents.personal_assistant.prompts import TODOIST_WORKER_PROMPT
from agents.personal_assistant.state import AgentState
from agents.personal_assistant.todoist_tools import get_todoist_tools
from core import get_model, settings

logger = logging.getLogger(__name__)

# Hard limit on ReAct iterations to prevent runaway loops
_MAX_ITERATIONS = 10


async def todoist_worker(state: AgentState, config: RunnableConfig) -> AgentState:
    """Domain-expert worker: ReAct loop over Todoist MCP tools only."""
    action_id: str = state.get("action_id", "unknown")
    action_input: dict = state.get("action_input", {})

    goal: str = action_input.get("goal", "")
    context: str = action_input.get("context", "")

    model_name = (
        config["configurable"].get("todoist_model")
        or config["configurable"].get("model")
        or settings.DEFAULT_MODEL
    )

    # Tools fetched fresh per invocation (not relying on module-level cache)
    tools = await get_todoist_tools()

    llm = get_model(model_name)
    if tools:
        llm = llm.bind_tools(tools)

    tool_map = {t.name: t for t in tools}

    context_section = f"\n## Pre-fetched context:\n{context}" if context else ""
    system_prompt = TODOIST_WORKER_PROMPT.format(
        date=datetime.now().strftime("%B %d, %Y"),
        context_section=context_section,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=goal),
    ]

    result: str = ""

    for iteration in range(_MAX_ITERATIONS):
        response = await llm.ainvoke(messages, config)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            result = str(response.content)
            break

        tool_results: list[ToolMessage] = []
        for call in tool_calls:
            tool = tool_map.get(call["name"])
            if tool is None:
                logger.warning("todoist_worker: unknown tool '%s'", call["name"])
                tool_results.append(
                    ToolMessage(
                        content=f"Tool '{call['name']}' not available.",
                        tool_call_id=call["id"],
                    )
                )
                continue
            try:
                tool_result = await tool.ainvoke(call["args"], config)
                tool_results.append(
                    ToolMessage(content=str(tool_result), tool_call_id=call["id"])
                )
            except Exception as e:
                logger.error("todoist_worker: tool '%s' failed: %s", call["name"], e)
                tool_results.append(
                    ToolMessage(
                        content=f"Tool call failed: {e}", tool_call_id=call["id"]
                    )
                )

        messages.extend(tool_results)

        if iteration == _MAX_ITERATIONS - 1:
            logger.warning("todoist_worker: reached max iterations (%d)", _MAX_ITERATIONS)
            result = str(response.content) or "Reached maximum tool-call iterations."

    return {
        "action_results": {action_id: result},
        "completed_actions": {action_id},
    }
