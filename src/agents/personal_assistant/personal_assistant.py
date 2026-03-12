"""Personal assistant — supervisor graph that routes to specialized sub-agents."""

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from pydantic import BaseModel

from agents.personal_assistant.conversation_agent import conversation_agent
from agents.personal_assistant.graph_store import (
    invalidate_relationships,
    upsert_entities,
    upsert_relationships,
)
from agents.personal_assistant.knowledge_store import invalidate_facts, retrieve_facts, store_facts
from agents.personal_assistant.memory_agent import memory_agent
from agents.personal_assistant.prompts import CLASSIFIER_PROMPT, GRAPH_EXTRACTION_PROMPT
from agents.personal_assistant.state import AgentState, IntentLiteral
from agents.personal_assistant.todoist_agent import todoist_agent
from agents.personal_assistant.web_search_agent import web_search_agent
from core import get_model, settings

logger = logging.getLogger(__name__)

# Intents that are pre-processing steps, not routable sub-agents
_MEMORY_OPS: set[str] = {"retrieve_context", "extract_and_store"}


# ---------------------------------------------------------------------------
# Shared memory nodes (triggered only when the classifier selects them)
# ---------------------------------------------------------------------------


class KnowledgeFact(BaseModel):
    content: str
    entity_type: Literal["person", "project", "process", "general"]
    entity_name: str


class GraphEntity(BaseModel):
    name: str
    entity_type: Literal["person", "project", "organization", "topic", "process"]
    properties: dict[str, str] = {}


class GraphRelationship(BaseModel):
    source: str
    source_type: str
    target: str
    target_type: str
    rel_type: Literal[
        "WORKS_ON",
        "WORKS_AT",
        "KNOWS",
        "USES",
        "INTERESTED_IN",
        "PART_OF",
        "INVOLVES",
        "RELATED_TO",
        "MENTIONS",
    ]
    properties: dict[str, str] = {}


class FactInvalidation(BaseModel):
    entity_name: str
    reason: str


class RelationshipInvalidation(BaseModel):
    source: str
    source_type: str
    rel_type: Literal[
        "WORKS_ON",
        "WORKS_AT",
        "KNOWS",
        "USES",
        "INTERESTED_IN",
        "PART_OF",
        "INVOLVES",
        "RELATED_TO",
        "MENTIONS",
    ]
    target_type: str
    target: str | None = None  # None = invalidate all of this rel_type from source
    reason: str


class ExtractionOutput(BaseModel):
    facts: list[KnowledgeFact] = []
    entities: list[GraphEntity] = []
    relationships: list[GraphRelationship] = []
    invalidate_facts: list[FactInvalidation] = []
    invalidate_relationships: list[RelationshipInvalidation] = []


async def extract_and_store(state: AgentState, config: RunnableConfig) -> AgentState:
    """Extract stable knowledge from the latest human message and store it in both stores.

    Before invoking the extraction LLM, retrieves existing facts for the entities
    mentioned in the message so the LLM can decide what to invalidate.
    """
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

    # Retrieve existing knowledge (including historical) so the LLM can decide
    # which facts/relationships are superseded by the new message.
    existing_docs = await retrieve_facts(last_message, k=10, include_history=False)
    existing_context = (
        "\n".join(
            f"[{d.metadata.get('entity_name', '')}] {d.page_content}"
            for d in existing_docs
        )
        if existing_docs
        else "None"
    )

    prompt_content = f"EXISTING FACTS:\n{existing_context}\n\nNEW MESSAGE:\n{last_message}"

    try:
        result: ExtractionOutput = await m.ainvoke(
            [SystemMessage(content=GRAPH_EXTRACTION_PROMPT), HumanMessage(content=prompt_content)]
        )

        # 1. Process invalidations first
        if result.invalidate_facts:
            for inv in result.invalidate_facts:
                n = await invalidate_facts(inv.entity_name)
                logger.info(
                    "Invalidated %d facts for '%s' — reason: %s", n, inv.entity_name, inv.reason
                )

        if result.invalidate_relationships:
            n = await invalidate_relationships(
                [
                    {
                        "source": r.source,
                        "source_type": r.source_type,
                        "rel_type": r.rel_type,
                        "target_type": r.target_type,
                        "target": r.target,
                    }
                    for r in result.invalidate_relationships
                ]
            )
            logger.info("Invalidated %d graph relationship(s)", n)

        # 2. Store new knowledge
        if result.facts:
            await store_facts(
                [
                    {"content": f.content, "entity_type": f.entity_type, "entity_name": f.entity_name}
                    for f in result.facts
                ]
            )

        if result.entities:
            n = await upsert_entities(
                [{"name": e.name, "entity_type": e.entity_type, "properties": e.properties} for e in result.entities]
            )
            logger.info(f"Upserted {n} graph entities")

        if result.relationships:
            n = await upsert_relationships(
                [
                    {
                        "source": r.source,
                        "source_type": r.source_type,
                        "target": r.target,
                        "target_type": r.target_type,
                        "rel_type": r.rel_type,
                        "properties": r.properties,
                    }
                    for r in result.relationships
                ]
            )
            logger.info(f"Upserted {n} graph relationships")

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
        logger.debug(f"Intents classified as {result.intents!r}: {result.reasoning}")
        return {"intents": result.intents}
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
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
