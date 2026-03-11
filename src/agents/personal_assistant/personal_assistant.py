"""Personal assistant — supervisor graph that routes to specialized sub-agents."""

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from agents.personal_assistant.conversation_agent import conversation_agent
from agents.personal_assistant.knowledge_store import retrieve_facts, store_facts
from agents.personal_assistant.memory_agent import memory_agent
from agents.personal_assistant.prompts import CLASSIFIER_PROMPT, EXTRACTION_PROMPT
from agents.personal_assistant.state import AgentState
from agents.personal_assistant.todoist_agent import todoist_agent
from agents.personal_assistant.web_search_agent import web_search_agent
from core import get_model, settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared memory nodes (run before routing, benefit all sub-agents)
# ---------------------------------------------------------------------------


class KnowledgeFact(BaseModel):
    content: str
    entity_type: Literal["person", "project", "process", "general"]
    entity_name: str


class ExtractionOutput(BaseModel):
    facts: list[KnowledgeFact]


async def extract_and_store(state: AgentState, config: RunnableConfig) -> AgentState:
    """Extract stable knowledge from the latest human message and store it."""
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not human_messages:
        return {}

    last_message = human_messages[-1].content
    if not isinstance(last_message, str) or not last_message.strip():
        return {}

    model_name = (
        config["configurable"].get("extraction_model")
        or config["configurable"].get("model")
        or settings.DEFAULT_MODEL
    )
    m = get_model(model_name).with_structured_output(ExtractionOutput)

    try:
        result: ExtractionOutput = await m.ainvoke(
            [SystemMessage(content=EXTRACTION_PROMPT), HumanMessage(content=last_message)]
        )
        if result.facts:
            await store_facts(
                [
                    {"content": f.content, "entity_type": f.entity_type, "entity_name": f.entity_name}
                    for f in result.facts
                ]
            )
    except Exception as e:
        logger.error(f"Knowledge extraction failed: {e}")

    return {}


async def retrieve_context(state: AgentState, config: RunnableConfig) -> AgentState:
    """Retrieve relevant knowledge for the latest human message."""
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not human_messages:
        return {"retrieved_context": ""}

    query = human_messages[-1].content
    if not isinstance(query, str):
        return {"retrieved_context": ""}

    docs = await retrieve_facts(query)
    if not docs:
        return {"retrieved_context": ""}

    context = "\n\n".join(
        f"[{doc.metadata.get('entity_type', 'general')}] "
        f"(stored: {doc.metadata.get('insertion_time', 'unknown')}) {doc.page_content}"
        for doc in docs
    )
    return {"retrieved_context": context}


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------


class IntentOutput(BaseModel):
    intent: Literal["todoist", "memory", "web_search", "general"]
    reasoning: str


async def classify_intent(state: AgentState, config: RunnableConfig) -> AgentState:
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not human_messages:
        return {"intent": "general"}

    last_message = human_messages[-1].content
    if not isinstance(last_message, str):
        return {"intent": "general"}

    model_name = (
        config["configurable"].get("classifier_model")
        or config["configurable"].get("model")
        or settings.DEFAULT_MODEL
    )
    m = get_model(model_name).with_structured_output(IntentOutput)

    try:
        result: IntentOutput = await m.ainvoke(
            [SystemMessage(content=CLASSIFIER_PROMPT), HumanMessage(content=last_message)]
        )
        logger.debug(f"Intent classified as '{result.intent}': {result.reasoning}")
        return {"intent": result.intent}
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return {"intent": "general"}


def route_intent(state: AgentState) -> str:
    return f"{state.get('intent', 'general')}_agent"


# ---------------------------------------------------------------------------
# Supervisor graph
# ---------------------------------------------------------------------------

agent = StateGraph(AgentState)

agent.add_node("retrieve_context", retrieve_context)
agent.add_node("extract_and_store", extract_and_store)
agent.add_node("classify_intent", classify_intent)
agent.add_node("todoist_agent", todoist_agent)
agent.add_node("memory_agent", memory_agent)
agent.add_node("conversation_agent", conversation_agent)
agent.add_node("web_search_agent", web_search_agent)

agent.set_entry_point("retrieve_context")
agent.add_edge("retrieve_context", "extract_and_store")
agent.add_edge("extract_and_store", "classify_intent")
agent.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "todoist_agent": "todoist_agent",
        "memory_agent": "memory_agent",
        "conversation_agent": "conversation_agent",
        "web_search_agent": "web_search_agent",
    },
)
agent.add_edge("todoist_agent", END)
agent.add_edge("memory_agent", END)
agent.add_edge("conversation_agent", END)
agent.add_edge("web_search_agent", END)

personal_assistant = agent.compile()
