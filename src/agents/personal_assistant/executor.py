"""Executor node — DAG engine that fans out ExecutionPlan actions via Send."""

import logging
import re

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send

from agents.personal_assistant.state import AgentState

logger = logging.getLogger(__name__)

# Maps WorkerLiteral tool names to LangGraph node names (wired in TASK-13)
_WORKER_NODE: dict[str, str] = {
    "todoist": "todoist_worker",
    "graphiti": "graphiti_worker",
    "web_search": "web_search_worker",
    "general": "conversation_worker",
}

_TEMPLATE_RE = re.compile(r"\{\{(\w+)\.result\}\}")


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------


def _resolve(value: object, results: dict[str, str]) -> object:
    """Recursively replace ``{{action_id.result}}`` placeholders in *value*."""
    if isinstance(value, str):
        return _TEMPLATE_RE.sub(
            lambda m: results.get(m.group(1), m.group(0)), value
        )
    if isinstance(value, dict):
        return {k: _resolve(v, results) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(item, results) for item in value]
    return value


def _resolve_input(action_input: dict, results: dict[str, str]) -> dict:
    """Return a copy of *action_input* with all templates resolved."""
    resolved = _resolve(action_input, results)
    assert isinstance(resolved, dict)
    return resolved


# ---------------------------------------------------------------------------
# Executor node
# ---------------------------------------------------------------------------


async def executor(state: AgentState, config: RunnableConfig) -> Command:
    """Dispatch all currently-ready actions from the ExecutionPlan via Send.

    "Ready" means every action listed in ``depends_on`` has completed AND the
    action has not yet been dispatched (not in ``started_actions``).
    """
    plan = state.get("execution_plan") or []
    completed: set[str] = state.get("completed_actions") or set()
    started: set[str] = state.get("started_actions") or set()
    results: dict[str, str] = state.get("action_results") or {}

    if not plan:
        logger.debug("Executor: empty plan, routing to synthesizer")
        return Command(goto="synthesizer")

    plan_ids = {a.id for a in plan}
    if plan_ids <= completed:
        logger.debug("Executor: all %d action(s) complete, routing to synthesizer", len(plan))
        return Command(goto="synthesizer")

    ready = [
        a
        for a in plan
        if a.id not in started and all(dep in completed for dep in a.depends_on)
    ]

    if not ready:
        # All un-completed actions are in-flight — executor_router will re-enter
        # us once a worker finishes and something becomes unblocked.
        logger.debug(
            "Executor: no ready actions (in-flight: %s)",
            started - completed,
        )
        return Command(goto=[])

    sends: list[Send] = []
    newly_started: set[str] = set()

    for action in ready:
        resolved_input = _resolve_input(action.input, results)
        node = _WORKER_NODE[action.tool]
        sends.append(
            Send(node, {"action_id": action.id, "action_input": resolved_input})
        )
        newly_started.add(action.id)
        logger.debug(
            "Executor dispatching action '%s' → '%s' (input keys: %s)",
            action.id,
            node,
            list(resolved_input.keys()),
        )

    return Command(goto=sends, update={"started_actions": newly_started})


# ---------------------------------------------------------------------------
# Executor router (conditional edge called after every worker completion)
# ---------------------------------------------------------------------------


def executor_router(state: AgentState) -> str:
    """Return the next node to visit after a worker finishes.

    - ``"synthesizer"`` when every action in the plan has completed.
    - ``"executor"`` otherwise, to dispatch newly-unblocked actions.
    """
    plan = state.get("execution_plan") or []
    if not plan:
        return "synthesizer"

    plan_ids = {a.id for a in plan}
    completed: set[str] = state.get("completed_actions") or set()

    if plan_ids <= completed:
        return "synthesizer"

    return "executor"
