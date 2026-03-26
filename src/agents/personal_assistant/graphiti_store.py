"""Graphiti-backed memory store for the personal assistant.

Replaces both knowledge_store.py (ChromaDB) and graph_store.py (Kuzu) with a
single temporal knowledge graph engine that handles entity extraction, embedding,
and temporal validity automatically.

Backend: Kuzu (embedded, no server required) via graphiti-core[kuzu]
LLM:     Follows the project's AllModelEnum selection (Anthropic / OpenAI / Gemini / Groq).
         Falls back to whichever API key is available when no model is specified.
Embedder: OpenAI (requires OPENAI_API_KEY)

Note: graphiti-core has neo4j as a core dependency (Python driver package only —
no Neo4j server is required when using the Kuzu driver).
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from schema.models import AllModelEnum, AnthropicModelName, OpenAIModelName

logger = logging.getLogger(__name__)


def _langsmith_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1")

GRAPHITI_DB_DIR = "./data/personal_assistant_graphiti.db"

# ---------------------------------------------------------------------------
# Anthropic model ID overrides
# Graphiti calls the Anthropic SDK directly so it needs the exact API model ID.
# Some enum values match the API ID; HAIKU_45 needs an explicit date suffix.
# ---------------------------------------------------------------------------
_ANTHROPIC_MODEL_IDS: dict[AnthropicModelName, str] = {
    AnthropicModelName.HAIKU_45: "claude-haiku-4-5-20251001",
    AnthropicModelName.SONNET_45: "claude-sonnet-4-5",
    AnthropicModelName.SONNET_46: "claude-sonnet-4-6",
}


# ---------------------------------------------------------------------------
# LangSmith tracing helpers
# ---------------------------------------------------------------------------


try:
    from langsmith import traceable as _traceable
except ImportError:
    import functools

    def _traceable(**_kw):  # type: ignore[misc]
        """No-op shim when langsmith is not installed."""

        def _decorator(fn):
            @functools.wraps(fn)
            async def _wrapper(*args, **kwargs):
                return await fn(*args, **kwargs)

            return _wrapper

        return _decorator


def _wrap_openai_client(api_key: str):
    """Return an AsyncOpenAI client wrapped with LangSmith tracing if enabled."""
    from openai import AsyncOpenAI

    raw = AsyncOpenAI(api_key=api_key)
    if _langsmith_enabled():
        try:
            from langsmith.wrappers import wrap_openai

            return wrap_openai(raw)
        except ImportError:
            logger.warning("GRAPHITI: langsmith not installed — OpenAI calls will not be traced")
    return raw


# ---------------------------------------------------------------------------
# LLM client factory (maps AllModelEnum → graphiti LLMClient)
# ---------------------------------------------------------------------------


def _build_graphiti_llm_client(model: AllModelEnum | None):
    """Return a graphiti LLM client for *model*, or the best available default.

    Only Anthropic and OpenAI are supported.  Any other provider falls back to
    whichever of those two has an API key configured.
    """
    from graphiti_core.llm_client.config import LLMConfig

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if isinstance(model, AnthropicModelName) and anthropic_key:
        from graphiti_core.llm_client.anthropic_client import AnthropicClient

        api_id = _ANTHROPIC_MODEL_IDS.get(model, "claude-haiku-4-5-20251001")
        logger.info("GRAPHITI LLM: Anthropic %s", api_id)
        return AnthropicClient(LLMConfig(api_key=anthropic_key, model=api_id))

    if isinstance(model, OpenAIModelName) and openai_key:
        from graphiti_core.llm_client.openai_client import OpenAIClient

        logger.info("GRAPHITI LLM: OpenAI %s", model.value)
        return OpenAIClient(
            LLMConfig(api_key=openai_key, model=model.value),
            client=_wrap_openai_client(openai_key),
        )

    # Fallback (model=None or unsupported provider): prefer Anthropic, then OpenAI
    if anthropic_key:
        from graphiti_core.llm_client.anthropic_client import AnthropicClient

        logger.info("GRAPHITI LLM: fallback → Anthropic claude-haiku-4-5-20251001")
        return AnthropicClient(
            LLMConfig(api_key=anthropic_key, model="claude-haiku-4-5-20251001")
        )

    from graphiti_core.llm_client.openai_client import OpenAIClient

    logger.info("GRAPHITI LLM: fallback → OpenAI default")
    return OpenAIClient(
        LLMConfig(api_key=openai_key or ""),
        client=_wrap_openai_client(openai_key or ""),
    )


def _build_embedder():
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

    return OpenAIEmbedder(
        OpenAIEmbedderConfig(api_key=os.getenv("OPENAI_API_KEY", ""))
    )


# ---------------------------------------------------------------------------
# Driver singleton + per-model Graphiti instance cache
# ---------------------------------------------------------------------------

_kuzu_driver = None  # KuzuDriver — initialized once, shared across instances
_graphiti_by_model: dict[str, Graphiti] = {}  # keyed by model string repr

_KUZU_FTS_QUERIES = [
    "CALL CREATE_FTS_INDEX('Episodic', 'episode_content', ['content', 'source', 'source_description']);",
    "CALL CREATE_FTS_INDEX('Entity', 'node_name_and_summary', ['name', 'summary']);",
    "CALL CREATE_FTS_INDEX('Community', 'community_name', ['name']);",
    "CALL CREATE_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', ['name', 'fact']);",
]


async def _ensure_driver():
    """Initialize the KuzuDriver singleton (idempotent)."""
    global _kuzu_driver
    if _kuzu_driver is not None:
        return

    Path(GRAPHITI_DB_DIR).parent.mkdir(parents=True, exist_ok=True)

    from graphiti_core.driver.kuzu_driver import KuzuDriver

    driver = KuzuDriver(db=GRAPHITI_DB_DIR)

    # Workaround: graphiti 0.28.x checks `self.driver._database` on every
    # add_episode call (Neo4j multi-DB logic), but KuzuDriver never sets
    # this attribute.  Setting it to the default group_id prevents the
    # AttributeError without changing any Kuzu behaviour.
    driver._database = "default"  # type: ignore[attr-defined]

    # Workaround: KuzuDriver.build_indices_and_constraints is a no-op in
    # graphiti 0.28.x, so FTS indexes are never created.  We create them
    # manually here; "already exists" errors are safe to ignore on restart.
    for q in _KUZU_FTS_QUERIES:
        try:
            await driver.execute_query(q)
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning("GRAPHITI: FTS index creation warning: %s", e)

    _kuzu_driver = driver
    logger.info("GRAPHITI: Kuzu driver ready at %s", GRAPHITI_DB_DIR)


async def get_graphiti(model: AllModelEnum | None = None) -> Graphiti:
    """Return a Graphiti instance for *model*, creating one if needed.

    The KuzuDriver (and therefore the database) is shared across all instances.
    Each distinct model gets its own Graphiti wrapper so the LLM client matches
    the caller's selection.
    """
    await _ensure_driver()

    key = str(model) if model is not None else "_default_"
    if key not in _graphiti_by_model:
        _graphiti_by_model[key] = Graphiti(
            graph_driver=_kuzu_driver,
            llm_client=_build_graphiti_llm_client(model),
            embedder=_build_embedder(),
        )
        logger.info("GRAPHITI: created instance for model=%s", key)

    return _graphiti_by_model[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@_traceable(name="graphiti:add_episode", run_type="tool")
async def add_episode(
    content: str,
    group_id: str = "default",
    model: AllModelEnum | None = None,
) -> None:
    """Ingest a message as an episode.

    Graphiti handles entity extraction, relationship detection, embedding, and
    temporal supersession internally — no manual extraction prompt needed.
    """
    g = await get_graphiti(model)
    name = f"msg_{uuid.uuid4().hex[:12]}"
    try:
        await g.add_episode(
            name=name,
            episode_body=content,
            source=EpisodeType.message,
            source_description="user conversation",
            group_id=group_id,
            reference_time=datetime.now(timezone.utc),
        )
        logger.info("GRAPHITI add_episode: %r (%d chars)", name, len(content))
    except Exception:
        logger.exception("GRAPHITI add_episode failed for %r", name)


@_traceable(name="graphiti:search_memory", run_type="retriever")
async def search_memory(
    query: str,
    num_results: int = 10,
    group_id: str = "default",
    include_history: bool = False,
    model: AllModelEnum | None = None,
) -> str:
    """Hybrid search (BM25 + cosine + graph) across the knowledge graph.

    Returns a formatted block of facts with temporal metadata, ready to inject
    into an agent's context window.  By default only current (valid) edges are
    returned; pass include_history=True to include superseded facts annotated
    with their invalidation time.
    """
    g = await get_graphiti(model)
    try:
        edges = await g.search(query, num_results=num_results, group_ids=[group_id])
        if not edges:
            logger.info("GRAPHITI search: no results for %r", query[:80])
            return ""

        lines: list[str] = []
        for edge in edges:
            # Skip historical facts unless explicitly requested
            if not include_history and edge.invalid_at is not None:
                continue
            timestamp = edge.valid_at.isoformat() if edge.valid_at else "unknown"
            suffix = ""
            if edge.invalid_at is not None:
                suffix = f" [HISTORICAL — superseded: {edge.invalid_at.isoformat()}]"
            lines.append(f"(since: {timestamp}{suffix}) {edge.fact}")

        logger.info(
            "GRAPHITI search: query=%r → %d/%d result(s) (include_history=%s)",
            query[:80],
            len(lines),
            len(edges),
            include_history,
        )
        return "\n".join(lines)
    except Exception:
        logger.exception("GRAPHITI search failed")
        return ""


@_traceable(name="graphiti:search_nodes", run_type="retriever")
async def search_nodes(
    query: str,
    group_id: str = "default",
    limit: int = 10,
    model: AllModelEnum | None = None,
) -> str:
    """Search for entity nodes by name — returns unique entity names found in matching edges."""
    g = await get_graphiti(model)
    try:
        edges = await g.search(query, num_results=limit * 2, group_ids=[group_id])
        if not edges:
            return f"No entities matching '{query}' found."

        # Collect unique source/target names from matching edges
        seen: set[str] = set()
        lines: list[str] = []
        for edge in edges:
            for name in (edge.source_node_name, edge.target_node_name):
                if name and name not in seen and query.lower() in name.lower():
                    seen.add(name)
                    lines.append(name)
            if len(lines) >= limit:
                break

        if not lines:
            # Fallback: return all unique entity names from the results
            for edge in edges:
                for name in (edge.source_node_name, edge.target_node_name):
                    if name and name not in seen:
                        seen.add(name)
                        lines.append(name)
                if len(lines) >= limit:
                    break

        logger.info("GRAPHITI node_search: %r → %d result(s)", query[:80], len(lines))
        return "\n".join(lines) if lines else f"No entities matching '{query}' found."
    except Exception:
        logger.exception("GRAPHITI node_search failed")
        return f"Entity search failed for '{query}'."


@_traceable(name="graphiti:get_entity_context", run_type="retriever")
async def get_entity_context(
    entity_name: str,
    group_id: str = "default",
    model: AllModelEnum | None = None,
) -> str:
    """Return all current facts/relationships involving a named entity."""
    g = await get_graphiti(model)
    try:
        edges = await g.search(
            entity_name,
            num_results=20,
            group_ids=[group_id],
        )
        if not edges:
            return f"No information found for '{entity_name}'."

        current = [e for e in edges if e.invalid_at is None]
        if not current:
            return f"No current facts found for '{entity_name}' (all historical)."

        lines = [f"  {e.fact}" for e in current]
        logger.info(
            "GRAPHITI entity_context: %r → %d fact(s)", entity_name, len(lines)
        )
        return f"[{entity_name}]\n" + "\n".join(lines)
    except Exception:
        logger.exception("GRAPHITI entity_context failed")
        return f"Context retrieval failed for '{entity_name}'."
