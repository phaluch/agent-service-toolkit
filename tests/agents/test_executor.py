"""Unit tests for the executor DAG engine (TASK-05)."""

import pytest
from unittest.mock import AsyncMock, patch

from agents.personal_assistant.executor import (
    _resolve,
    _resolve_input,
    executor,
    executor_router,
)
from agents.personal_assistant.state import Action, AgentState


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------


class TestResolve:
    def test_plain_string_unchanged(self):
        assert _resolve("hello", {}) == "hello"

    def test_single_template_replaced(self):
        result = _resolve("{{a1.result}}", {"a1": "done"})
        assert result == "done"

    def test_template_embedded_in_string(self):
        result = _resolve("Use this: {{a1.result}} then carry on", {"a1": "VALUE"})
        assert result == "Use this: VALUE then carry on"

    def test_missing_key_left_intact(self):
        result = _resolve("{{missing.result}}", {})
        assert result == "{{missing.result}}"

    def test_nested_dict_resolved(self):
        result = _resolve({"goal": "do {{a1.result}}", "extra": 42}, {"a1": "task"})
        assert result == {"goal": "do task", "extra": 42}

    def test_list_resolved(self):
        result = _resolve(["x", "{{a1.result}}"], {"a1": "y"})
        assert result == ["x", "y"]

    def test_non_string_passthrough(self):
        assert _resolve(123, {}) == 123
        assert _resolve(None, {}) is None


class TestResolveInput:
    def test_resolves_dict_values(self):
        action_input = {"goal": "Create a task about {{mem.result}}"}
        resolved = _resolve_input(action_input, {"mem": "Paulo's project"})
        assert resolved == {"goal": "Create a task about Paulo's project"}

    def test_returns_copy(self):
        action_input = {"goal": "plain text"}
        resolved = _resolve_input(action_input, {})
        assert resolved == action_input
        assert resolved is not action_input


# ---------------------------------------------------------------------------
# executor_router
# ---------------------------------------------------------------------------


class TestExecutorRouter:
    def _make_plan(self, *ids: str) -> list[Action]:
        return [
            Action(id=aid, tool="general", input={}, depends_on=[], reason="test")
            for aid in ids
        ]

    def test_empty_plan_routes_to_synthesizer(self):
        state: AgentState = {"messages": [], "execution_plan": []}
        assert executor_router(state) == "synthesizer"

    def test_all_completed_routes_to_synthesizer(self):
        state: AgentState = {
            "messages": [],
            "execution_plan": self._make_plan("a1", "a2"),
            "completed_actions": {"a1", "a2"},
        }
        assert executor_router(state) == "synthesizer"

    def test_partial_completion_routes_to_executor(self):
        state: AgentState = {
            "messages": [],
            "execution_plan": self._make_plan("a1", "a2"),
            "completed_actions": {"a1"},
        }
        assert executor_router(state) == "executor"

    def test_no_completed_routes_to_executor(self):
        state: AgentState = {
            "messages": [],
            "execution_plan": self._make_plan("a1"),
        }
        assert executor_router(state) == "executor"


# ---------------------------------------------------------------------------
# executor node (async)
# ---------------------------------------------------------------------------


def _action(aid: str, tool: str = "general", depends_on: list[str] | None = None, **kw) -> Action:
    return Action(
        id=aid,
        tool=tool,
        input=kw.get("input", {"goal": "do it"}),
        depends_on=depends_on or [],
        reason="test",
    )


@pytest.mark.asyncio
class TestExecutorNode:
    async def _run(self, state: AgentState):
        config = {"configurable": {}}
        return await executor(state, config)

    async def test_empty_plan_goes_to_synthesizer(self):
        state: AgentState = {"messages": [], "execution_plan": []}
        cmd = await self._run(state)
        assert cmd.goto == "synthesizer"

    async def test_all_done_goes_to_synthesizer(self):
        plan = [_action("a1")]
        state: AgentState = {
            "messages": [],
            "execution_plan": plan,
            "completed_actions": {"a1"},
            "started_actions": {"a1"},
        }
        cmd = await self._run(state)
        assert cmd.goto == "synthesizer"

    async def test_dispatches_ready_action(self):
        plan = [_action("a1", tool="todoist")]
        state: AgentState = {"messages": [], "execution_plan": plan}
        cmd = await self._run(state)

        from langgraph.types import Send as _Send

        assert len(cmd.goto) == 1
        send = cmd.goto[0]
        assert isinstance(send, _Send)
        assert send.node == "todoist_worker"
        assert send.arg["action_id"] == "a1"

    async def test_marks_dispatched_as_started(self):
        plan = [_action("a1", tool="graphiti")]
        state: AgentState = {"messages": [], "execution_plan": plan}
        cmd = await self._run(state)
        assert "a1" in cmd.update["started_actions"]

    async def test_already_started_not_redispatched(self):
        plan = [_action("a1", tool="todoist"), _action("a2", tool="web_search")]
        state: AgentState = {
            "messages": [],
            "execution_plan": plan,
            "started_actions": {"a1"},
        }
        cmd = await self._run(state)
        nodes = [s.node for s in cmd.goto]
        assert "todoist_worker" not in nodes
        assert "web_search_worker" in nodes

    async def test_dependency_blocks_dispatch(self):
        plan = [
            _action("a1", tool="graphiti"),
            _action("a2", tool="todoist", depends_on=["a1"]),
        ]
        state: AgentState = {"messages": [], "execution_plan": plan}
        cmd = await self._run(state)
        nodes = [s.node for s in cmd.goto]
        assert "graphiti_worker" in nodes
        assert "todoist_worker" not in nodes

    async def test_dependency_unblocked_after_completion(self):
        plan = [
            _action("a1", tool="graphiti"),
            _action("a2", tool="todoist", depends_on=["a1"]),
        ]
        state: AgentState = {
            "messages": [],
            "execution_plan": plan,
            "started_actions": {"a1"},
            "completed_actions": {"a1"},
            "action_results": {"a1": "some context"},
        }
        cmd = await self._run(state)
        nodes = [s.node for s in cmd.goto]
        assert "todoist_worker" in nodes
        assert "graphiti_worker" not in nodes

    async def test_template_resolved_in_input(self):
        plan = [
            _action("a1", tool="graphiti"),
            _action(
                "a2",
                tool="todoist",
                depends_on=["a1"],
                input={"goal": "Create task about {{a1.result}}"},
            ),
        ]
        state: AgentState = {
            "messages": [],
            "execution_plan": plan,
            "started_actions": {"a1"},
            "completed_actions": {"a1"},
            "action_results": {"a1": "Paulo's sprint"},
        }
        cmd = await self._run(state)
        assert len(cmd.goto) == 1
        payload = cmd.goto[0].arg
        assert payload["action_input"]["goal"] == "Create task about Paulo's sprint"

    async def test_parallel_independent_actions_dispatched_together(self):
        plan = [
            _action("a1", tool="web_search"),
            _action("a2", tool="todoist"),
        ]
        state: AgentState = {"messages": [], "execution_plan": plan}
        cmd = await self._run(state)
        assert len(cmd.goto) == 2
        nodes = {s.node for s in cmd.goto}
        assert nodes == {"web_search_worker", "todoist_worker"}

    async def test_general_tool_maps_to_conversation_worker(self):
        plan = [_action("a1", tool="general")]
        state: AgentState = {"messages": [], "execution_plan": plan}
        cmd = await self._run(state)
        assert cmd.goto[0].node == "conversation_worker"

    async def test_no_ready_returns_empty_goto(self):
        # a1 started but not yet complete; a2 depends on a1 → nothing to dispatch
        plan = [
            _action("a1", tool="graphiti"),
            _action("a2", tool="todoist", depends_on=["a1"]),
        ]
        state: AgentState = {
            "messages": [],
            "execution_plan": plan,
            "started_actions": {"a1"},
            "completed_actions": set(),
        }
        cmd = await self._run(state)
        assert cmd.goto == []
