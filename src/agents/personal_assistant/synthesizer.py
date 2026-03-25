"""Synthesizer node — produces the final user-facing response from action results."""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agents.personal_assistant.prompts import SYNTHESIZER_PROMPT
from agents.personal_assistant.state import AgentState
from core import get_model, settings

logger = logging.getLogger(__name__)


async def synthesizer(state: AgentState, config: RunnableConfig) -> AgentState:
    """Combine action_results with conversation history into a single coherent reply."""
    model_name = (
        config["configurable"].get("synthesizer_model")
        or config["configurable"].get("model")
        or settings.DEFAULT_MODEL
    )
    llm = get_model(model_name)

    action_results: dict[str, str] = state.get("action_results") or {}

    if action_results:
        results_text = "\n\n".join(
            f"Result:\n{result}" for result in action_results.values()
        )
        user_facing_results = f"Worker results:\n{results_text}"
    else:
        user_facing_results = "No external results were retrieved."

    # Build a compact conversation excerpt so the LLM has original user intent
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    last_human = human_messages[-1].content if human_messages else ""

    synthesis_input = (
        f"User request: {last_human}\n\n{user_facing_results}"
        if last_human
        else user_facing_results
    )

    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=SYNTHESIZER_PROMPT),
                HumanMessage(content=synthesis_input),
            ]
        )
        logger.debug("Synthesizer produced response (%d chars)", len(str(response.content)))
        return {"messages": [AIMessage(content=response.content)]}
    except Exception as e:
        logger.error("Synthesizer failed: %s", e)
        return {"messages": [AIMessage(content="I encountered an error while preparing your response. Please try again.")]}
