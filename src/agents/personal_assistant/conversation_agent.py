"""General conversation sub-agent."""

import logging
from datetime import datetime

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from agents.personal_assistant.prompts import CONVERSATION_SYSTEM_PROMPT
from agents.personal_assistant.state import AgentState
from core import get_model, settings

logger = logging.getLogger(__name__)


async def respond(state: AgentState, config: RunnableConfig) -> AgentState:
    model_name = (
        config["configurable"].get("conversation_model")
        or config["configurable"].get("model")
        or settings.DEFAULT_MODEL
    )
    m = get_model(model_name)

    context = state.get("retrieved_context", "")
    system = CONVERSATION_SYSTEM_PROMPT.format(date=datetime.now().strftime("%B %d, %Y"))
    if context:
        system += f"\n\n## Context from your knowledge base:\n{context}"

    messages = [SystemMessage(content=system)] + state["messages"]
    response = await m.ainvoke(messages, config)
    return {"messages": [response]}


graph = StateGraph(AgentState)
graph.add_node("respond", respond)
graph.set_entry_point("respond")
graph.add_edge("respond", END)

conversation_agent = graph.compile()
