"""Personal assistant — supervisor graph that routes to specialized sub-agents."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from pydantic import BaseModel

from agents.personal_assistant.conversation_agent import conversation_agent
from agents.personal_assistant.graphiti_store import add_episode, search_memory
from agents.personal_assistant.memory_agent import memory_agent
from agents.personal_assistant.prompts import CLASSIFIER_PROMPT
from agents.personal_assistant.state import AgentState, IntentLiteral
from agents.personal_assistant.todoist_agent import todoist_agent
from agents.personal_assistant.web_search_agent import web_search_agent
from core import get_model, settings

logger = logging.getLogger(__name__)

# Intents that are pre-processing steps, not routable sub-agents
_MEMORY_OPS: set[str] = {"retrieve_context", "extract_and_store"}


# ---------------------------------------------------------------------------
# Memory nodes
# ---------------------------------------------------------------------------


async def extract_and_store(state: AgentState, config: RunnableConfig) -> AgentState:
    """Ingest the latest human message into the Graphiti knowledge graph.

    Graphiti handles entity extraction, relationship detection, embedding, and
    temporal supersession internally — no manual extraction prompt needed.
    """
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not human_messages:
        return {}

    last_message = human_messages[-1].content
    if not isinstance(last_message, str) or not last_message.strip():
        return {}

    user_id = config["configurable"].get("user_id", "default")
    model = config["configurable"].get("entity_extraction_model") or config[
        "configurable"
    ].get("model")

    try:
        await add_episode(last_message, group_id=user_id, model=model)
        logger.info("EXTRACT_AND_STORE: ingested message (%d chars) for user %r", len(last_message), user_id)
    except Exception as e:
        logger.error("EXTRACT_AND_STORE failed: %s", e)

    return {}


async def retrieve_context(state: AgentState, config: RunnableConfig) -> AgentState:
    """Retrieve relevant knowledge for the latest human message."""
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not human_messages:
        return {"retrieved_context": ""}

    query = human_messages[-1].content
    if not isinstance(query, str):
        return {"retrieved_context": ""}

    user_id = config["configurable"].get("user_id", "default")
    model = config["configurable"].get("entity_extraction_model") or config[
        "configurable"
    ].get("model")
    logger.info("RETRIEVE_CONTEXT: query=%r user=%r", query[:120], user_id)

    context = await search_memory(query, num_results=10, group_id=user_id, model=model)

    if context:
        logger.info("RETRIEVE_CONTEXT: %d char(s) returned", len(context))
    else:
        logger.info("RETRIEVE_CONTEXT: no context found")

    return {"retrieved_context": context}


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------


class IntentOutput(BaseModel):
    intents: list[IntentLiteral]
    reasoning: str


async def classify_intent(state: AgentState, config: RunnableConfig) -> AgentState:
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not human_messages:
        return {"intents": ["general"]}

    last_message = human_messages[-1].content
    if not isinstance(last_message, str):
        return {"intents": ["general"]}

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
        logger.debug("Intents classified as %r: %s", result.intents, result.reasoning)
        return {"intents": result.intents}
    except Exception as e:
        logger.error("Intent classification failed: %s", e)
        return {"intents": ["general"]}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def dispatch_agents(state: AgentState) -> list[Send]:
    """Edge routing function: fan-out to extract_and_store and each agent in parallel."""
    intents = state.get("intents", ["general"])
    sends = []
    if "extract_and_store" in intents:
        sends.append(Send("extract_and_store", state))
    for intent in intents:
        if intent not in _MEMORY_OPS:
            sends.append(Send(f"{intent}_agent", state))
    return sends


def route_after_classify(state: AgentState) -> str | list[Send]:
    """If retrieval was requested, run it first; otherwise fan-out immediately."""
    if "retrieve_context" in state.get("intents", []):
        return "retrieve_context"
    return dispatch_agents(state)


# ---------------------------------------------------------------------------
# Supervisor graph
# ---------------------------------------------------------------------------

agent = StateGraph(AgentState)

agent.add_node("classify_intent", classify_intent)
agent.add_node("retrieve_context", retrieve_context)
agent.add_node("extract_and_store", extract_and_store)
agent.add_node("todoist_agent", todoist_agent)
agent.add_node("memory_agent", memory_agent)
agent.add_node("general_agent", conversation_agent)
agent.add_node("web_search_agent", web_search_agent)

agent.set_entry_point("classify_intent")
agent.add_conditional_edges("classify_intent", route_after_classify)
agent.add_conditional_edges("retrieve_context", dispatch_agents)

# dispatch_agents uses Send — no explicit edges needed for the agent nodes
agent.add_edge("todoist_agent", END)
agent.add_edge("memory_agent", END)
agent.add_edge("general_agent", END)
agent.add_edge("web_search_agent", END)
agent.add_edge("extract_and_store", END)

personal_assistant = agent.compile()
