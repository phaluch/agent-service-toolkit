"""Intake node — classifies request complexity for routing."""

import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from agents.personal_assistant.prompts import INTAKE_PROMPT
from agents.personal_assistant.state import AgentState
from core import get_model, settings

logger = logging.getLogger(__name__)

_HISTORY_WINDOW = 8  # number of recent messages to include as context


class IntakeOutput(BaseModel):
    complexity: Literal["simple", "complex"]
    reasoning: str


def _build_history_snippet(state: AgentState) -> str:
    """Return a formatted conversation snippet excluding the last message, or empty string."""
    recent = state["messages"][-_HISTORY_WINDOW:]
    lines = []
    for m in recent[:-1]:  # exclude the current (last) message
        if isinstance(m, HumanMessage) and isinstance(m.content, str):
            lines.append(f"[User]: {m.content}")
        elif isinstance(m, AIMessage) and isinstance(m.content, str):
            lines.append(f"[Assistant]: {m.content}")
    return "\n".join(lines)


async def intake(state: AgentState, config: RunnableConfig) -> AgentState:
    """Classify whether the request needs the Decomposer+Coordinator path or just Coordinator."""
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not human_messages:
        return {"complexity": "simple"}

    last_message = human_messages[-1].content
    if not isinstance(last_message, str):
        return {"complexity": "simple"}

    history = _build_history_snippet(state)
    if history:
        user_content = (
            f"Recent conversation:\n{history}\n\nLatest message to classify: {last_message}"
        )
    else:
        user_content = last_message

    model_name = (
        config["configurable"].get("intake_model")
        or config["configurable"].get("model")
        or settings.DEFAULT_MODEL
    )
    m = get_model(model_name).with_structured_output(IntakeOutput)

    try:
        result: IntakeOutput = await m.ainvoke(
            [SystemMessage(content=INTAKE_PROMPT), HumanMessage(content=user_content)]
        )
        logger.debug("Intake classified as %r: %s", result.complexity, result.reasoning)
        return {"complexity": result.complexity}
    except Exception as e:
        logger.error("Intake classification failed: %s", e)
        return {"complexity": "complex"}
