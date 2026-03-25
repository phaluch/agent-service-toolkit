"""Personal assistant — Planner → Coordinator → Domain Workers graph.

Supported configurable keys (pass via agent_config in the API, or RunnableConfig directly):

  user_name     (str) — canonical name of the assistant's owner, e.g. "Paulo".
                        Used by the coordinator and graphiti/conversation workers to
                        ensure the user is represented by a consistent entity name
                        in the knowledge graph and in generated responses.

  user_context  (str) — optional free-text persona/background for the conversation
                        worker, e.g. "Paulo is a software developer who prefers
                        concise, technical answers."

  model         (str) — default model for all nodes (overridden per-node by
                        coordinator_model, graphiti_model, etc.).
"""

import logging

from langgraph.graph import END, StateGraph

from agents.personal_assistant.conversation_worker import conversation_worker
from agents.personal_assistant.coordinator import coordinator
from agents.personal_assistant.decomposer import decomposer
from agents.personal_assistant.executor import executor, executor_router
from agents.personal_assistant.graphiti_worker import graphiti_worker
from agents.personal_assistant.intake import intake
from agents.personal_assistant.state import AgentState
from agents.personal_assistant.synthesizer import synthesizer
from agents.personal_assistant.todoist_worker import todoist_worker
from agents.personal_assistant.web_search_worker import web_search_worker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def intake_router(state: AgentState) -> str:
    """Route to decomposer for complex requests, directly to coordinator for simple ones."""
    if state.get("complexity") == "complex":
        return "decomposer"
    return "coordinator"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

agent = StateGraph(AgentState)

# Orchestration nodes
agent.add_node("intake", intake)
agent.add_node("decomposer", decomposer)
agent.add_node("coordinator", coordinator)
agent.add_node("executor", executor)
agent.add_node("synthesizer", synthesizer)

# Domain worker nodes
agent.add_node("todoist_worker", todoist_worker)
agent.add_node("graphiti_worker", graphiti_worker)
agent.add_node("web_search_worker", web_search_worker)
agent.add_node("conversation_worker", conversation_worker)

# Entry point
agent.set_entry_point("intake")

# intake → simple → coordinator, complex → decomposer
agent.add_conditional_edges(
    "intake",
    intake_router,
    {"coordinator": "coordinator", "decomposer": "decomposer"},
)

# decomposer → coordinator
agent.add_edge("decomposer", "coordinator")

# coordinator → executor
agent.add_edge("coordinator", "executor")

# executor fans out via Command/Send to workers (no explicit edge needed)
# Each worker routes through executor_router on completion
_WORKER_NODES = [
    "todoist_worker",
    "graphiti_worker",
    "web_search_worker",
    "conversation_worker",
]
for _worker in _WORKER_NODES:
    agent.add_conditional_edges(
        _worker,
        executor_router,
        {"synthesizer": "synthesizer", "executor": "executor"},
    )

# synthesizer is terminal
agent.add_edge("synthesizer", END)

personal_assistant = agent.compile()
