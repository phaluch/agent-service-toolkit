"""Decomposer node — fragments a complex request into typed, self-contained pieces."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from agents.personal_assistant.prompts import DECOMPOSER_PROMPT
from agents.personal_assistant.state import AgentState, Fragment
from core import get_model, settings

logger = logging.getLogger(__name__)


class DecomposerOutput(BaseModel):
    fragments: list[Fragment]


async def decomposer(state: AgentState, config: RunnableConfig) -> AgentState:
    """Break a complex request into typed fragments for the Coordinator to plan around."""
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not human_messages:
        return {"fragments": []}

    last_message = human_messages[-1].content
    if not isinstance(last_message, str):
        return {"fragments": []}

    model_name = (
        config["configurable"].get("decomposer_model")
        or config["configurable"].get("model")
        or settings.DEFAULT_MODEL
    )
    m = get_model(model_name).with_structured_output(DecomposerOutput)

    try:
        result: DecomposerOutput = await m.ainvoke(
            [SystemMessage(content=DECOMPOSER_PROMPT), HumanMessage(content=last_message)]
        )
        logger.debug("Decomposer produced %d fragment(s)", len(result.fragments))
        return {"fragments": result.fragments}
    except Exception as e:
        logger.error("Decomposer failed: %s", e)
        return {"fragments": []}
