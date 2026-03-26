"""Coordinator node — translates state into a validated ExecutionPlan."""

import logging
from collections import deque

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, model_validator

from agents.personal_assistant.prompts import COORDINATOR_PROMPT
from agents.personal_assistant.state import Action, AgentState
from core import get_model, settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output model
# ---------------------------------------------------------------------------


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[Action]

    @model_validator(mode="after")
    def validate_depends_on(self) -> "ExecutionPlan":
        ids = {a.id for a in self.actions}
        for action in self.actions:
            for dep in action.depends_on:
                if dep not in ids:
                    raise ValueError(
                        f"Action '{action.id}' depends_on unknown action '{dep}'"
                    )
        # Cycle detection via Kahn's topological sort
        indegree: dict[str, int] = {a.id: 0 for a in self.actions}
        adj: dict[str, list[str]] = {a.id: [] for a in self.actions}
        for action in self.actions:
            for dep in action.depends_on:
                adj[dep].append(action.id)
                indegree[action.id] += 1
        queue: deque[str] = deque(aid for aid, deg in indegree.items() if deg == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for neighbour in adj[node]:
                indegree[neighbour] -= 1
                if indegree[neighbour] == 0:
                    queue.append(neighbour)
        if visited != len(self.actions):
            raise ValueError("depends_on graph contains a cycle")
        return self


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def coordinator(state: AgentState, config: RunnableConfig) -> AgentState:
    """Produce an ExecutionPlan from raw messages (simple) or fragments (complex)."""
    model_name = (
        config["configurable"].get("coordinator_model")
        or config["configurable"].get("model")
        or settings.DEFAULT_MODEL
    )
    llm = get_model(model_name).with_structured_output(ExecutionPlan, method="function_calling")

    user_name = config["configurable"].get("user_name", "")
    user_section = (
        f"\n## User identity\n\n"
        f"The owner of this assistant is {user_name}. "
        f"When creating a graphiti action that concerns the user personally "
        f"(their preferences, goals, habits, or relationships), always include "
        f'"{user_name}" in entity_hints so the graphiti worker uses a consistent '
        f"entity name in the knowledge graph."
        if user_name
        else ""
    )
    coordinator_prompt = COORDINATOR_PROMPT.replace("{user_section}", user_section)

    complexity = state.get("complexity", "simple")
    if complexity == "complex" and state.get("fragments"):
        fragments_text = "\n".join(
            f"  [{f.type}] {f.content}"
            + (f"  (entities: {', '.join(f.entities)})" if f.entities else "")
            for f in state["fragments"]
        )
        user_content = f"Fragments to plan:\n{fragments_text}"
    else:
        human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if not human_messages:
            return {"execution_plan": []}
        last = human_messages[-1].content
        if not isinstance(last, str):
            return {"execution_plan": []}
        # Include recent conversation so follow-up messages ("remove the duplicate",
        # "do it", "cancel that") can be resolved to the correct worker and entity.
        recent = state["messages"][-8:]
        history_lines = []
        for m in recent[:-1]:
            if isinstance(m, HumanMessage) and isinstance(m.content, str):
                history_lines.append(f"[User]: {m.content}")
            elif isinstance(m, AIMessage) and isinstance(m.content, str):
                history_lines.append(f"[Assistant]: {m.content}")
        if history_lines:
            user_content = (
                "Recent conversation:\n"
                + "\n".join(history_lines)
                + f"\n\nLatest user request: {last}"
            )
        else:
            user_content = last

    try:
        result: ExecutionPlan = await llm.ainvoke(
            [SystemMessage(content=coordinator_prompt), HumanMessage(content=user_content)]
        )
        logger.debug("Coordinator produced %d action(s)", len(result.actions))
        # Reset per-invocation tracking so stale results from a prior run don't
        # cause the executor to skip actions that share the same ID.
        return {
            "execution_plan": result.actions,
            "action_results": None,
            "completed_actions": None,
            "started_actions": None,
        }
    except Exception as e:
        logger.error("Coordinator failed: %s", e)
        return {"execution_plan": []}
