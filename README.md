# Personal Assistant

A personal AI assistant for work and life, built on top of [agent-service-toolkit](https://github.com/JoshuaC215/agent-service-toolkit) by Joshua Carroll.

The assistant is a LangGraph supervisor graph that routes each message to a specialized sub-agent:

- **Conversation agent** — general chat and questions, with access to your personal knowledge base
- **Memory agent** — explicitly query or update what the assistant knows about you
- **Todoist agent** — manage tasks via the Todoist API

Passive knowledge extraction runs on every turn: facts you mention are automatically stored in a ChromaDB vector store and retrieved as context for future messages.

## Setup

1. Copy the example env file and fill in your keys:

   ```sh
   cp .env.example .env
   ```

   Key variables:

   | Variable | Required | Description |
   |---|---|---|
   | `ANTHROPIC_API_KEY` | Yes (or another LLM key) | LLM provider |
   | `DEFAULT_MODEL` | Recommended | e.g. `claude-sonnet-4-6` |
   | `TODOIST_API_KEY` | Optional | Enables task management |
   | `DATABASE_TYPE` | Optional | `sqlite` (default) or `postgres` |

2. Copy the prompts example and customize it (this file is gitignored):

   ```sh
   cp src/agents/personal_assistant/prompts.example.py src/agents/personal_assistant/prompts.py
   ```

## Running

**With Python:**

```sh
uv sync --frozen
source .venv/bin/activate

# Start the API service
python src/run_service.py

# In another shell, start the chat UI
streamlit run src/streamlit_app.py
```

**With Docker:**

```sh
docker compose watch
```

- Chat UI: http://localhost:8501
- API + docs: http://localhost:8080/redoc

## Project structure

```
src/
  agents/personal_assistant/
    personal_assistant.py   # Supervisor graph: extract → retrieve → classify → route
    conversation_agent.py   # General conversation
    memory_agent.py         # Explicit knowledge base queries and updates
    todoist_agent.py        # Task management via Todoist API
    knowledge_store.py      # ChromaDB vector store
    state.py                # Shared graph state
    prompts.py              # System prompts (gitignored — copy from prompts.example.py)
  service/service.py        # FastAPI service
  streamlit_app.py          # Streamlit chat interface
```

## License

MIT — see [LICENSE](LICENSE).
