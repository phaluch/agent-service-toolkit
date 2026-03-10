"""Personal assistant agent with proactive knowledge store retrieval."""

import logging
from datetime import datetime
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph
from pydantic import BaseModel

from agents.personal_assistant.knowledge_store import retrieve_facts, store_facts
from agents.personal_assistant.todoist_tools import get_todoist_tools
from core import get_model, settings

logger = logging.getLogger(__name__)

current_date = datetime.now().strftime("%B %d, %Y")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(MessagesState, total=False):
    retrieved_context: str


# ---------------------------------------------------------------------------
# Structured extraction schema
# ---------------------------------------------------------------------------


class KnowledgeFact(BaseModel):
    content: str
    """Full self-contained sentence describing the fact."""
    entity_type: Literal["person", "project", "process", "general"]
    entity_name: str
    """Primary entity label, e.g. 'Paulo' or 'Project Alpha'."""


class ExtractionOutput(BaseModel):
    facts: list[KnowledgeFact]


EXTRACTION_PROMPT = """You analyze messages to extract stable, long-term knowledge worth storing in a personal knowledge base.

Extract facts that are:
- Descriptions of people (who they are, their role, relationship to the user)
- Project scope, status, or important context
- Process notes, decisions, or ways of doing things
- Any information the user would want to recall months from now

Do NOT extract:
- Ephemeral or time-sensitive information (e.g. "I'm tired today")
- Simple questions or requests
- Things already universally known

For each fact, write a complete, self-contained sentence understandable without the original message.
Return an empty list if nothing worth storing is found.
"""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def extract_and_store(state: AgentState, config: RunnableConfig) -> AgentState:
    """Extract stable knowledge from the latest human message and store it."""
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not human_messages:
        return {}

    last_message = human_messages[-1].content
    if not isinstance(last_message, str) or not last_message.strip():
        return {}

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    extraction_model = m.with_structured_output(ExtractionOutput)

    try:
        result: ExtractionOutput = await extraction_model.ainvoke(
            [SystemMessage(content=EXTRACTION_PROMPT), HumanMessage(content=last_message)]
        )
        if result.facts:
            await store_facts(
                [
                    {
                        "content": f.content,
                        "entity_type": f.entity_type,
                        "entity_name": f.entity_name,
                    }
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
        f"[{doc.metadata.get('entity_type', 'general')}] (stored: {doc.metadata.get('insertion_time', 'unknown')}) {doc.page_content}"
        for doc in docs
    )
    return {"retrieved_context": context}


SYSTEM_PROMPT = f"""You are a personal assistant. Today's date is {current_date}.

You have access to a personal knowledge base containing information about the user's contacts,
projects, and processes. Use this context to give informed, personalized responses.

When context is available:
- Reference it naturally without quoting it verbatim
- Connect relevant pieces (e.g. link a person to a project they're involved in)
- If you just stored new information from the user's message, briefly acknowledge it

Be conversational, direct, and helpful.
"""


async def respond(state: AgentState, config: RunnableConfig) -> AgentState:
    """Generate a response, injecting retrieved context into the system prompt."""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))

    todoist_tools = await get_todoist_tools()
    if todoist_tools:
        m = m.bind_tools(todoist_tools)

    context = state.get("retrieved_context", "")
    if context:
        system_content = (
            SYSTEM_PROMPT
            + "\n\n## Context from your knowledge base:\n"
            + context
        )
    else:
        system_content = SYSTEM_PROMPT

    messages = [SystemMessage(content=system_content)] + state["messages"]
    response = await m.ainvoke(messages, config)
    return {"messages": [response]}


async def execute_tools(state: AgentState, config: RunnableConfig) -> AgentState:
    """Execute Todoist tool calls from the last AI message."""
    last_msg = state["messages"][-1]
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return {}

    tools = await get_todoist_tools()
    tool_map = {t.name: t for t in tools}

    results = []
    for call in last_msg.tool_calls:
        tool = tool_map.get(call["name"])
        if tool is None:
            logger.warning(f"Tool {call['name']} not found in Todoist tools")
            continue
        try:
            result = await tool.ainvoke(call["args"], config)
            results.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        except Exception as e:
            logger.error(f"Todoist tool {call['name']} failed: {e}")
            results.append(ToolMessage(content=f"Tool call failed: {e}", tool_call_id=call["id"]))

    return {"messages": results}


def route_after_respond(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "execute_tools"
    return END


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

agent = StateGraph(AgentState)
agent.add_node("retrieve_context", retrieve_context)
agent.add_node("extract_and_store", extract_and_store)
agent.add_node("respond", respond)
agent.add_node("execute_tools", execute_tools)

agent.set_entry_point("retrieve_context")
agent.add_edge("retrieve_context", "extract_and_store")
agent.add_edge("extract_and_store", "respond")
agent.add_conditional_edges("respond", route_after_respond, ["execute_tools", END])
agent.add_edge("execute_tools", "respond")

personal_assistant = agent.compile()
