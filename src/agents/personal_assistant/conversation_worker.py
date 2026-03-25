"""Conversation domain-worker node — pure reasoning/generation, no tools.

Receives a scoped action_input dict from the executor (goal, optional context).
Context is pre-resolved by the Coordinator; no external lookups needed.
Returns a generated text response stored in action_results[action_id].
"""

import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agents.personal_assistant.prompts import CONVERSATION_WORKER_PROMPT
from agents.personal_assistant.state import AgentState
from core import get_model, settings

logger = logging.getLogger(__name__)


async def conversation_worker(state: AgentState, config: RunnableConfig) -> AgentState:
    """Domain-expert worker: single LLM call for conversational generation."""
    action_id: str = state.get("action_id", "unknown")
    action_input: dict = state.get("action_input", {})

    goal: str = action_input.get("goal", "")
    context: str = action_input.get("context", "")

    model_name = (
        config["configurable"].get("conversation_model")
        or config["configurable"].get("model")
        or settings.DEFAULT_MODEL
    )

    llm = get_model(model_name)

    context_section = f"\n## Pre-fetched context:\n{context}" if context else ""
    system_prompt = CONVERSATION_WORKER_PROMPT.format(
        date=datetime.now().strftime("%B %d, %Y"),
        context_section=context_section,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=goal),
    ]

    response = await llm.ainvoke(messages, config)
    result = str(response.content)

    return {
        "action_results": {action_id: result},
        "completed_actions": {action_id},
    }
