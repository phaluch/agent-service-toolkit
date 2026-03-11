"""Shared state schema for the personal assistant multi-agent graph."""

from typing import Literal

from langgraph.graph import MessagesState

IntentLiteral = Literal[
    "todoist",
    "memory",
    "web_search",
    "general",
    "retrieve_context",
    "extract_and_store",
]


class AgentState(MessagesState, total=False):
    retrieved_context: str
    intents: list[IntentLiteral]
