"""Graphiti domain-worker node — manages the personal knowledge graph.

Receives a scoped action_input dict from the executor (goal, optional entity_hints).
Has no knowledge of other workers or the global execution plan.
Returns a formatted facts string or confirmation stored in action_results[action_id].
"""

import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from agents.personal_assistant.memory_agent import MEMORY_TOOLS
from agents.personal_assistant.prompts import GRAPHITI_WORKER_PROMPT
from agents.personal_assistant.state import AgentState
from core import get_model, settings

logger = logging.getLogger(__name__)

# Hard limit on ReAct iterations to prevent runaway loops
_MAX_ITERATIONS = 10


async def graphiti_worker(state: AgentState, config: RunnableConfig) -> AgentState:
    """Domain-expert worker: ReAct loop over Graphiti knowledge graph tools only."""
    action_id: str = state.get("action_id", "unknown")
    action_input: dict = state.get("action_input", {})

    goal: str = action_input.get("goal", "")
    entity_hints: list[str] = action_input.get("entity_hints", [])

    model_name = (
        config["configurable"].get("graphiti_model")
        or config["configurable"].get("model")
        or settings.DEFAULT_MODEL
    )

    llm = get_model(model_name).bind_tools(MEMORY_TOOLS)
    tool_map = {t.name: t for t in MEMORY_TOOLS}

    hints_section = (
        f"\n## Entity hints:\n" + "\n".join(f"- {h}" for h in entity_hints)
        if entity_hints
        else ""
    )
    user_name = config["configurable"].get("user_name", "")
    user_section = (
        f"\n\n## User identity\n\n"
        f"The owner of this assistant is {user_name}. "
        f'When storing or retrieving facts that refer to the user themselves '
        f'("I", "me", "my"), always represent them as "USER" for ease of retrieval '
        f"in all remember() calls and entity lookups."
        if user_name
        else ""
    )
    system_prompt = GRAPHITI_WORKER_PROMPT.format(
        date=datetime.now().strftime("%B %d, %Y"),
        hints_section=hints_section,
        user_section=user_section,
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
            t = tool_map.get(call["name"])
            if t is None:
                logger.warning("graphiti_worker: unknown tool '%s'", call["name"])
                tool_results.append(
                    ToolMessage(
                        content=f"Tool '{call['name']}' not available.",
                        tool_call_id=call["id"],
                    )
                )
                continue
            try:
                tool_result = await t.ainvoke(call["args"], config)
                tool_results.append(
                    ToolMessage(content=str(tool_result), tool_call_id=call["id"])
                )
            except Exception as e:
                logger.error("graphiti_worker: tool '%s' failed: %s", call["name"], e)
                tool_results.append(
                    ToolMessage(
                        content=f"Tool call failed: {e}", tool_call_id=call["id"]
                    )
                )

        messages.extend(tool_results)

        if iteration == _MAX_ITERATIONS - 1:
            logger.warning("graphiti_worker: reached max iterations (%d)", _MAX_ITERATIONS)
            result = str(response.content) or "Reached maximum tool-call iterations."

    return {
        "action_results": {action_id: result},
        "completed_actions": {action_id},
    }
