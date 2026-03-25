# Roadmap: Planner-Executor Architecture

Refactor the `personal_assistant` from a flat supervisor-router pattern into a
**Planner → Coordinator → Domain Workers** architecture.

**Core principle:** the Coordinator owns all cross-domain decisions (what flows
where, in what order). Domain workers are autonomous within their own silo and
have no awareness of other workers.

## Development environment note

This project runs entirely in Docker. Do **not** run bare `python`, `pytest`, or similar
commands directly on the host — they will use the wrong interpreter or environment.
Always use `docker compose exec <service> <command>` to run commands inside the correct
container.

---

## Architecture target

```
START
  │
  ▼
[INTAKE]          simple vs. complex routing
  │
  ├─ simple ──────────────────────────────────────────────► [COORDINATOR]
  │                                                               │
  └─ complex ──► [DECOMPOSER] ──► [COORDINATOR]                  │
                                       │                         │
                                       ▼                         ▼
                                  ExecutionPlan ──► [EXECUTOR] (DAG engine)
                                                        │
                          ┌─────────────┬───────────────┼────────────────┐
                          ▼             ▼               ▼                ▼
                   todoist_worker  graphiti_worker  web_search_worker  conversation_worker
                   (domain expert) (domain expert)  (domain expert)   (domain expert)
                          │             │               │                │
                          └─────────────┴───────────────┴────────────────┘
                                                   │  fan-in
                                                   ▼
                                         [SYNTHESIZER]
                                                   │
                                                  END
```

**Worker contract:** each worker receives `{goal, context?, ...}` scoped to its
domain. It has a ReAct loop over its own tools. It does not know other workers
exist. Cross-domain data flows are resolved by the Coordinator via `depends_on`.

---

## Dependency graph (parallelization map)

```
TASK-01 (state schema)
   ├── TASK-02 (intake)        ─────────────────────────► TASK-13 (assembly)
   ├── TASK-03 (decomposer)    ──► TASK-04 (coordinator)       │
   ├── TASK-04 (coordinator)   ──► TASK-11 (coord prompt) ─────┤
   ├── TASK-05 (executor)      ─────────────────────────► TASK-13
   ├── TASK-06 (synthesizer)   ─────────────────────────► TASK-13
   ├── TASK-07 (todoist worker)──► TASK-12 (worker prompts)    │
   ├── TASK-08 (graphiti worker)─► TASK-12                ─────┤
   ├── TASK-09 (web search worker) TASK-12                ─────┤
   └── TASK-10 (conv worker)   ──► TASK-12                ─────┘
                                                               │
                                                    TASK-13 ──► TASK-14 ──► TASK-15
```

**Two parallel tracks after TASK-01:**
- **Track A** — Orchestration: TASK-02, 03, 04, 05, 06
- **Track B** — Workers: TASK-07, 08, 09, 10

Both tracks converge at TASK-13 (assembly).

---

## Branching strategy

### Long-lived integration branch

`main` stays stable. `feat/planner-executor` is the staging area for the entire
refactor. It only merges to `main` when TASK-15 (integration tests) is green.

### Task branches: `pe/<task-id>-<slug>`

All task branches use the short `pe/` prefix for scannability and branch off
`feat/planner-executor`. Examples:

```
feat/planner-executor
  ├── pe/task-01-state-schema
  ├── pe/task-02-intake
  ├── pe/task-03-decomposer
  ├── pe/task-04-coordinator      ← branches off pe/task-03-decomposer (see below)
  └── ...
```

### Dependency rule

When TASK-X depends on TASK-Y and TASK-Y hasn't merged yet, branch off TASK-Y's
branch rather than waiting:

```
feat/planner-executor
  └── pe/task-03-decomposer
        └── pe/task-04-coordinator   ← PR targets pe/task-03-decomposer
```

The PR for TASK-04 targets `pe/task-03-decomposer` first, which rebases cleanly
into `feat/planner-executor` once TASK-03 merges. Avoids long idle waits.

---

## EPIC 0 — Foundation

### TASK-01 · New state schema & data models

**Everything depends on this. Start here.**

Replace `intents` and `retrieved_context` with an execution-plan-centric schema.
Add the `Action` Pydantic model and custom LangGraph reducers for concurrent writes.

**Files:** `state.py`

**Acceptance criteria:**
- [ ] `Action` model has: `id`, `tool`, `input: dict`, `depends_on: list[str]`, `reason`
- [ ] `tool` is a `Literal` enum of all available workers (todoist, graphiti, web_search, general)
- [ ] `AgentState` has `complexity`, `execution_plan`, `action_results`, `completed_actions`
- [ ] `action_results` uses a custom `merge_results` reducer (dict merge, safe for parallel writes)
- [ ] `completed_actions` uses a custom `union_reducer` (set union, safe for parallel writes)
- [ ] `intents` and `retrieved_context` are removed
- [ ] Old imports compile without error (fields may be unused until workers are refactored)

**Size:** S

---

## EPIC 1 — Orchestration Layer

Tasks in this epic can be developed in parallel with EPIC 2.
They depend only on TASK-01.

---

### TASK-02 · Intake node

Replaces `classify_intent`. Determines routing complexity — does this request
need Decomposer + Coordinator, or can it go straight to Coordinator?

**Files:** new `intake.py`
**Depends on:** TASK-01

**Acceptance criteria:**
- [ ] Single LLM call with structured output: `IntakeOutput(complexity, reasoning)`
- [ ] `complexity: Literal["simple", "complex"]`
- [ ] "Simple" = single domain, no cross-domain data dependency
- [ ] "Complex" = multiple steps, data from one domain feeds another, or multiple domains
- [ ] Uses `intake_model` → `model` → `DEFAULT_MODEL` config priority
- [ ] Returns `{"complexity": ...}` into state

**Size:** S

---

### TASK-03 · Decomposer node

Only activated for `complexity == "complex"`. Fragments a composite message
into typed, self-contained pieces that the Coordinator can plan around.

**Files:** new `decomposer.py`
**Depends on:** TASK-01

**Acceptance criteria:**
- [ ] Structured output: `DecomposerOutput(fragments: list[Fragment])`
- [ ] `Fragment` has: `type` (task, memory_store, memory_query, web_search, general), `content`, `entities: list[str]`
- [ ] Output written to `state["fragments"]` (add field to AgentState)
- [ ] Does not call any tools — pure LLM classification
- [ ] Uses `decomposer_model` → `model` → `DEFAULT_MODEL`

**Size:** M

---

### TASK-04 · Coordinator node

The brain. Receives state (with fragments if complex, or raw messages if simple)
and produces an `ExecutionPlan` with explicit dependency graph.

**Files:** new `coordinator.py`
**Depends on:** TASK-01, TASK-02, TASK-03

**Acceptance criteria:**
- [ ] Structured output: `ExecutionPlan(actions: list[Action])`
- [ ] System prompt lists all available workers with their capabilities and limitations
- [ ] System prompt documents `depends_on` semantics and `{{action_id.result}}` template syntax
- [ ] For "simple" path: receives messages directly, produces 1-action plan
- [ ] For "complex" path: receives fragments from Decomposer, produces multi-action plan
- [ ] `depends_on` is validated: no cycles, no references to non-existent action IDs
- [ ] Uses `coordinator_model` → `model` → `DEFAULT_MODEL`

**Size:** L (prompt engineering is the hard part — see TASK-11)

---

### TASK-05 · Executor node (DAG engine)

Resolves the `ExecutionPlan` dependency graph. Dispatches actions in topological
order, parallelizing independent actions via `Send`.

**Files:** new `executor.py`
**Depends on:** TASK-01 (worker node names needed for `Send` targets)

**Acceptance criteria:**
- [ ] On each entry: compute "ready" actions (all `depends_on` satisfied, not yet started)
- [ ] Resolves `{{action_id.result}}` templates in `action.input` before dispatching
- [ ] Fan-out: dispatches all ready actions simultaneously via `Send`
- [ ] Re-enters after each fan-in batch; checks for newly unblocked actions
- [ ] Terminates when `completed_actions == {a.id for a in execution_plan}` → routes to `synthesizer`
- [ ] If a worker fails, stores the error string in `action_results[action_id]` and continues
- [ ] Does not call any LLM

**Size:** L (most technically complex task)

**Notes:**
Template resolution must handle nested references and missing keys gracefully.
The loop re-entry pattern requires careful LangGraph edge wiring — use a
`executor_router` conditional edge that fires after every worker completion.

---

### TASK-06 · Response synthesizer

Final node. Takes all `action_results` + conversation history and produces a
single coherent response.

**Files:** new `synthesizer.py`
**Depends on:** TASK-01

**Acceptance criteria:**
- [ ] Receives full message history + `action_results` dict
- [ ] Single LLM call (no tools)
- [ ] Formats multi-action results into a natural, unified response
- [ ] Does not expose internal action IDs or plan structure to the user
- [ ] Uses `synthesizer_model` → `model` → `DEFAULT_MODEL`

**Size:** S

---

## EPIC 2 — Domain Workers

All tasks in this epic are **fully independent of each other**.
They depend only on TASK-01 and can run in parallel with EPIC 1.

Each worker:
- Receives a scoped `action_input` dict (`goal`, optional `context`), not the full `AgentState`
- Has a ReAct loop over its own tools only
- Has no knowledge of other workers or the global plan
- Returns a `str` result that goes into `action_results[action_id]`

---

### TASK-07 · Todoist worker

Refactor `todoist_agent.py` into a domain-expert worker.

**Files:** refactor `todoist_agent.py` → `todoist_worker.py`
**Depends on:** TASK-01

**Acceptance criteria:**
- [ ] Input contract: `action_input: dict` with `goal: str` and optional `context: str`
- [ ] Does NOT receive global state fields (no `retrieved_context`, no `messages`)
- [ ] ReAct loop over Todoist MCP tools only
- [ ] Can make multiple tool calls to accomplish the goal (e.g. list projects → create task)
- [ ] System prompt: Todoist-domain-only, includes user's workspace conventions (see TASK-12)
- [ ] Returns confirmation string with task ID / URL
- [ ] `get_todoist_tools()` called once at node entry, not cached globally

**Size:** M

---

### TASK-08 · Graphiti worker

Refactor `memory_agent.py` into a domain-expert worker.

**Files:** refactor `memory_agent.py` → `graphiti_worker.py`
**Depends on:** TASK-01

**Acceptance criteria:**
- [ ] Input contract: `action_input: dict` with `goal: str`, optional `entity_hints: list[str]`
- [ ] ReAct loop over Graphiti tools only: `search_knowledge`, `get_entity`, `find_entities`, `remember`
- [ ] Can combine multiple tool strategies to satisfy the goal (entity lookup + semantic search)
- [ ] System prompt: knowledge graph reasoning, scoped to memory operations only (see TASK-12)
- [ ] Returns formatted facts or confirmation string

**Size:** M

---

### TASK-09 · Web search worker

Refactor `web_search_agent.py` into a domain-expert worker.

**Files:** refactor `web_search_agent.py` → `web_search_worker.py`
**Depends on:** TASK-01

**Acceptance criteria:**
- [ ] Input contract: `action_input: dict` with `query: str`, optional `context: str`
- [ ] ReAct loop over Perplexity tools only
- [ ] Can reformulate queries and do follow-up searches within its loop
- [ ] System prompt: research methodology, citation standards (see TASK-12)
- [ ] Returns structured findings string with sources

**Size:** M

---

### TASK-10 · Conversation worker

Refactor `conversation_agent.py` into a domain-expert worker.

**Files:** refactor `conversation_agent.py` → `conversation_worker.py`
**Depends on:** TASK-01

**Acceptance criteria:**
- [ ] Input contract: `action_input: dict` with `goal: str`, optional `context: str`
- [ ] Single LLM call (no tools — pure reasoning/generation)
- [ ] Context is pre-resolved by Coordinator (no need to fetch it)
- [ ] System prompt: conversational, personalized, scoped to generation (see TASK-12)

**Size:** S

---

## EPIC 3 — Prompt Engineering

Prompt quality determines system quality. These tasks are iterative.

---

### TASK-11 · Coordinator system prompt

**Files:** `prompts.py`
**Depends on:** TASK-04 + all worker tasks (need to know full tool inventory)

**Acceptance criteria:**
- [ ] Describes each available worker with: name, what it can do, what it cannot do
- [ ] Documents `depends_on` semantics with concrete examples
- [ ] Documents `{{action_id.result}}` template syntax
- [ ] Includes few-shot examples covering: single action, parallel independent actions, sequential dependent actions
- [ ] The Paulo example works: graphiti lookup → todoist create with `depends_on`
- [ ] Simple requests produce 1-action plans (not unnecessarily complex)
- [ ] `extract_and_store` is just a `graphiti` action in the plan, not special-cased

**Size:** L (iterative — expect multiple refinement cycles against real inputs)

---

### TASK-12 · Worker system prompts

**Files:** `prompts.py`
**Depends on:** TASK-07, 08, 09, 10

**Acceptance criteria:**
- [ ] Each prompt has explicit domain scope ("you have no knowledge of other systems")
- [ ] Each prompt documents the input contract the worker receives
- [ ] Todoist prompt: workspace conventions, project/label/priority guidance
- [ ] Graphiti prompt: multi-hop search strategy, when to combine search types
- [ ] Web search prompt: query reformulation, citation standards
- [ ] Conversation prompt: personalization, tone, how to use pre-fetched context

**Size:** M

---

## EPIC 4 — Assembly & Migration

---

### TASK-13 · Rebuild personal_assistant.py

Wire all new nodes into the final graph.

**Files:** `personal_assistant.py`
**Depends on:** TASK-02, 03, 04, 05, 06, 07, 08, 09, 10

**Acceptance criteria:**
- [ ] Graph topology matches the architecture diagram at the top of this document
- [ ] `intake` → conditional edge: simple → `coordinator`, complex → `decomposer` → `coordinator`
- [ ] `coordinator` → `executor`
- [ ] `executor` fans out via `Send` to worker nodes based on `action.tool`
- [ ] Each worker node calls `executor_router` on completion (to re-enter executor)
- [ ] `executor_router` routes to `synthesizer` when plan is complete
- [ ] `personal_assistant = graph.compile()` — no other changes to `agents.py`

**Size:** M

---

### TASK-14 · Migrate extract_and_store

Remove the hardcoded `extract_and_store` side-effect node. Memory writes become
a regular `graphiti` action in the Coordinator's plan.

**Files:** `personal_assistant.py`, update `coordinator.py` prompt
**Depends on:** TASK-08, TASK-11, TASK-13

**Acceptance criteria:**
- [ ] `extract_and_store` node removed from graph
- [ ] Coordinator prompt instructs when to include a `graphiti` memory-write action
- [ ] End-to-end test: "remember that X" produces a plan with a `graphiti` `remember` action

**Size:** S

---

### TASK-15 · Integration test suite

**Files:** new `tests/test_personal_assistant_e2e.py`
**Depends on:** TASK-13

**Scenarios to cover:**
- [ ] Simple single-domain request (e.g. "create a task for tomorrow")
- [ ] Cross-domain with dependency (e.g. "create task about Paulo" → graphiti lookup feeds todoist)
- [ ] Parallel independent actions (e.g. "search for X and create a task for Y")
- [ ] Memory write + agent action in same request
- [ ] Graceful degradation when a worker errors (plan continues, error surfaced in synthesis)
- [ ] Simple conversational request produces 1-action plan (conversation worker)

**Size:** M

---

## Summary table

| Task | Title | Epic | Depends on | Size | Track |
|------|-------|------|------------|------|-------|
| TASK-01 | State schema | 0 — Foundation | — | S | Blocker |
| TASK-02 | Intake node | 1 — Orchestration | 01 | S | A |
| TASK-03 | Decomposer node | 1 — Orchestration | 01 | M | A |
| TASK-04 | Coordinator node | 1 — Orchestration | 01, 02, 03 | L | A |
| TASK-05 | Executor (DAG) | 1 — Orchestration | 01 | L | A |
| TASK-06 | Synthesizer | 1 — Orchestration | 01 | S | A |
| TASK-07 | Todoist worker | 2 — Workers | 01 | M | B |
| TASK-08 | Graphiti worker | 2 — Workers | 01 | M | B |
| TASK-09 | Web search worker | 2 — Workers | 01 | M | B |
| TASK-10 | Conversation worker | 2 — Workers | 01 | S | B |
| TASK-11 | Coordinator prompt | 3 — Prompts | 04 + workers | L | A |
| TASK-12 | Worker prompts | 3 — Prompts | 07–10 | M | B |
| TASK-13 | Graph assembly | 4 — Assembly | all | M | Final |
| TASK-14 | Migrate extract_and_store | 4 — Assembly | 08, 11, 13 | S | Final |
| TASK-15 | Integration tests | 4 — Assembly | 13 | M | Final |
