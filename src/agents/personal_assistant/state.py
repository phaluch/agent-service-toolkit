"""Shared state schema for the personal assistant multi-agent graph."""

from typing import Annotated, Literal, Union

from langgraph.graph import MessagesState
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Worker type enum
# ---------------------------------------------------------------------------

WorkerLiteral = Literal["todoist", "graphiti", "web_search", "general"]

# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------


class Fragment(BaseModel):
    """A self-contained sub-goal extracted from a complex request."""

    type: Literal["task", "memory_store", "memory_query", "web_search", "general"]
    content: str
    entities: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-tool input models — closed schemas, satisfies OpenAI strict mode
# ---------------------------------------------------------------------------


class TodoistInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    context: str = ""


class GraphitiInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    entity_hints: list[str] = Field(default_factory=list)


class WebSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    context: str = ""


class GeneralInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    context: str = ""


# ---------------------------------------------------------------------------
# Per-tool action models — discriminated on `tool` for OpenAI strict mode
# ---------------------------------------------------------------------------


class TodoistAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    tool: Literal["todoist"]
    input: TodoistInput
    depends_on: list[str] = Field(default_factory=list)
    reason: str


class GraphitiAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    tool: Literal["graphiti"]
    input: GraphitiInput
    depends_on: list[str] = Field(default_factory=list)
    reason: str


class WebSearchAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    tool: Literal["web_search"]
    input: WebSearchInput
    depends_on: list[str] = Field(default_factory=list)
    reason: str


class GeneralAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    tool: Literal["general"]
    input: GeneralInput
    depends_on: list[str] = Field(default_factory=list)
    reason: str


# Discriminated union — `tool` selects the correct branch unambiguously
Action = Annotated[
    Union[TodoistAction, GraphitiAction, WebSearchAction, GeneralAction],
    Field(discriminator="tool"),
]

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
    # Orchestration fields
    complexity: Literal["simple", "complex"]
    fragments: list[Fragment]
    execution_plan: list[Action]
    action_results: Annotated[dict[str, str], merge_results]
    completed_actions: Annotated[set[str], union_reducer]
    started_actions: Annotated[set[str], union_reducer]
    # Worker-scoped fields — populated by executor via Send
    action_id: str
    action_input: dict
