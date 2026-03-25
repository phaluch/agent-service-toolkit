"""Shared state schema for the personal assistant multi-agent graph."""

from typing import Annotated, Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Worker type enum
# ---------------------------------------------------------------------------

WorkerLiteral = Literal["todoist", "graphiti", "web_search", "general"]

# ---------------------------------------------------------------------------
# Backward-compat alias — remove when personal_assistant.py is rebuilt (TASK-13)
# ---------------------------------------------------------------------------

IntentLiteral = Literal[
    "todoist",
    "memory",
    "web_search",
    "general",
    "retrieve_context",
    "extract_and_store",
]

# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------


class Fragment(BaseModel):
    """A self-contained sub-goal extracted from a complex request."""

    type: Literal["task", "memory_store", "memory_query", "web_search", "general"]
    content: str
    entities: list[str] = Field(default_factory=list)


class Action(BaseModel):
    """A single unit of work in the execution plan."""

    id: str
    tool: WorkerLiteral
    input: dict
    depends_on: list[str] = Field(default_factory=list)
    reason: str


# ---------------------------------------------------------------------------
# Reducers for parallel-safe state merging
# ---------------------------------------------------------------------------


def merge_results(left: dict[str, str] | None, right: dict[str, str]) -> dict[str, str]:
    """Merge two action-result dicts; safe for concurrent writes."""
    return {**(left or {}), **right}


def union_reducer(left: set[str] | None, right: set[str]) -> set[str]:
    """Union two completed-action sets; safe for concurrent writes."""
    return (left or set()) | right


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class AgentState(MessagesState, total=False):
    # --- Backward-compat fields: used by the old personal_assistant.py graph.
    # Remove when TASK-13 replaces the graph.
    intents: list[IntentLiteral]
    retrieved_context: str
    # --- New planner-executor fields ---
    complexity: Literal["simple", "complex"]
    fragments: list[Fragment]
    execution_plan: list[Action]
    action_results: Annotated[dict[str, str], merge_results]
    completed_actions: Annotated[set[str], union_reducer]
    started_actions: Annotated[set[str], union_reducer]
    # Worker-scoped fields — populated by executor via Send
    action_id: str
    action_input: dict
