"""Web search domain-worker node — research via Perplexity MCP integration.

Receives a scoped action_input dict from the executor (query, optional context).
Has no knowledge of other workers or the global execution plan.
Returns a structured findings string stored in action_results[action_id].
"""

import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from agents.personal_assistant.perplexity_tools import get_perplexity_tools
from agents.personal_assistant.prompts import WEB_SEARCH_WORKER_PROMPT
from agents.personal_assistant.state import AgentState
from core import get_model, settings

logger = logging.getLogger(__name__)

# Hard limit on ReAct iterations to prevent runaway loops
_MAX_ITERATIONS = 10


async def web_search_worker(state: AgentState, config: RunnableConfig) -> AgentState:
    """Domain-expert worker: ReAct loop over Perplexity tools only."""
    action_id: str = state.get("action_id", "unknown")
    action_input: dict = state.get("action_input", {})

    query: str = action_input.get("query", "")
    context: str = action_input.get("context", "")

    model_name = (
        config["configurable"].get("web_search_model")
        or config["configurable"].get("model")
        or settings.DEFAULT_MODEL
    )

    # Tools fetched fresh per invocation (module-level cache in perplexity_tools handles dedup)
    tools = await get_perplexity_tools()

    llm = get_model(model_name)
    if tools:
        llm = llm.bind_tools(tools)

    tool_map = {t.name: t for t in tools}

    context_section = f"\n## Background context:\n{context}" if context else ""
    system_prompt = WEB_SEARCH_WORKER_PROMPT.format(
        date=datetime.now().strftime("%B %d, %Y"),
        context_section=context_section,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
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
                logger.warning("web_search_worker: unknown tool '%s'", call["name"])
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
                logger.error("web_search_worker: tool '%s' failed: %s", call["name"], e)
                tool_results.append(
                    ToolMessage(
                        content=f"Tool call failed: {e}", tool_call_id=call["id"]
                    )
                )

        messages.extend(tool_results)

        if iteration == _MAX_ITERATIONS - 1:
            logger.warning("web_search_worker: reached max iterations (%d)", _MAX_ITERATIONS)
            result = str(response.content) or "Reached maximum tool-call iterations."

    return {
        "action_results": {action_id: result},
        "completed_actions": {action_id},
    }
