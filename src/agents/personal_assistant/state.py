"""Shared state schema for the personal assistant multi-agent graph."""

from typing import Literal

from langgraph.graph import MessagesState


class AgentState(MessagesState, total=False):
    retrieved_context: str
    intent: Literal["todoist", "memory", "general"]
