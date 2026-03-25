# Agent Service Toolkit — Developer Instructions

You are working on the `agent-service-toolkit` project, a LangGraph-based personal assistant.

## Active refactor

We are building a Planner → Coordinator → Domain Workers architecture.
Roadmap: `docs/roadmap-planner-executor.md` — read it before starting any task.

## Branching rules

- Long-lived integration branch: `feat/planner-executor`
- Task branches: `pe/<task-id>-<slug>` (e.g. `pe/task-04-coordinator`)
- Branch off the parent task's branch if that task hasn't merged yet, not off `feat/planner-executor` directly
- One branch per task. Create it before writing any code.

## Commit style

Follow the pattern in recent commits:
```
feat(<scope>): short description (TASK-XX)
```
One commit per logical unit. Co-author line required:
```
Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

## Environment & Docker

The entire stack runs in Docker. The compose file is `compose.yaml` (not `docker-compose.yml`).

### How source code gets into the container

`agent_service` does **not** have a static volume mount for `src/`. Source files are synced
via `docker compose watch`, which maps:

| Host path     | Container path  |
|---------------|-----------------|
| `src/agents/` | `/app/agents/`  |
| `src/core/`   | `/app/core/`    |
| `src/schema/` | `/app/schema/`  |
| `src/service/`| `/app/service/` |
| `src/memory/` | `/app/memory/`  |

**Consequence:** if the stack was started with `docker compose up` (not `watch`), files you
create or edit on the host are **not visible inside the container** until you either:
- run `docker compose watch` (preferred for active development — syncs on save), or
- rebuild with `docker compose up --build agent_service`.

### Validating new files

**Quick syntax check (no Docker needed):**
```bash
python3 -c "import ast; ast.parse(open('src/agents/personal_assistant/my_node.py').read()); print('Syntax OK')"
```
The repo has a `.venv` with Python 3.12 that can parse files even without the project installed.

**Import/runtime check (requires watch or rebuild):**
```bash
# Only works if docker compose watch is running, or after a rebuild
docker compose exec agent_service python -c "from agents.personal_assistant.my_node import my_fn; print('OK')"
```

**Never run bare `python`, `pytest`, or similar commands on the host** — they will use the
wrong interpreter. All runtime checks must go through `docker compose exec <service> <cmd>`.

## Code patterns

Before implementing a node, read an existing one for conventions:
- `src/agents/personal_assistant/coordinator.py` — canonical async node pattern
- `src/agents/personal_assistant/state.py` — shared state schema
- `src/agents/personal_assistant/prompts.py` — where all prompts live (gitignored)
- `src/agents/personal_assistant/prompts.example.py` — checked-in template; add stub prompts here too

## Before writing any code

1. Read the task spec in `docs/roadmap-planner-executor.md`
2. Check which branch to base off (`git log --oneline <parent-branch> -5`)
3. Create the task branch (`git checkout -b pe/task-XX-slug`)
4. Implement
5. Syntax-check new files with the `python3 ast.parse` one-liner above
6. Commit with the required style
