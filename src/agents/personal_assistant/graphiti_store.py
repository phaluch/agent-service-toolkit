"""Graphiti-backed memory store for the personal assistant.

Replaces both knowledge_store.py (ChromaDB) and graph_store.py (Kuzu) with a
single temporal knowledge graph engine that handles entity extraction, embedding,
and temporal validity automatically.

Backend: Kuzu (embedded, no server required) via graphiti-core[kuzu]
LLM:     Anthropic if ANTHROPIC_API_KEY is set, otherwise OpenAI
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

logger = logging.getLogger(__name__)

GRAPHITI_DB_DIR = "./data/personal_assistant_graphiti"

# ---------------------------------------------------------------------------
# LLM + embedder construction
# ---------------------------------------------------------------------------


def _build_llm_client():
    """Return an Anthropic client if the key is available, otherwise OpenAI."""
    if os.getenv("ANTHROPIC_API_KEY"):
        from graphiti_core.llms.anthropic import AnthropicClient
        from graphiti_core.llms.config import LLMConfig

        logger.info("GRAPHITI: using Anthropic LLM client")
        return AnthropicClient(LLMConfig(api_key=os.environ["ANTHROPIC_API_KEY"]))

    from graphiti_core.llms.config import LLMConfig
    from graphiti_core.llms.openai_client import OpenAIClient

    logger.info("GRAPHITI: using OpenAI LLM client")
    return OpenAIClient(LLMConfig(api_key=os.getenv("OPENAI_API_KEY", "")))


def _build_embedder():
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

    return OpenAIEmbedder(
        OpenAIEmbedderConfig(api_key=os.getenv("OPENAI_API_KEY", ""))
    )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_graphiti: Graphiti | None = None


async def get_graphiti() -> Graphiti:
    global _graphiti
    if _graphiti is None:
        Path(GRAPHITI_DB_DIR).mkdir(parents=True, exist_ok=True)

        from graphiti_core.driver.kuzu_driver import KuzuDriver

        # KuzuDriver takes a path string directly — no kuzu.Database construction needed
        driver = KuzuDriver(db=GRAPHITI_DB_DIR)

        _graphiti = Graphiti(
            graph_driver=driver,
            llm_client=_build_llm_client(),
            embedder=_build_embedder(),
        )
        await _graphiti.build_indices_and_constraints()
        logger.info("GRAPHITI: initialised with Kuzu backend at %s", GRAPHITI_DB_DIR)

    return _graphiti


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def add_episode(content: str, group_id: str = "default") -> None:
    """Ingest a message as an episode.

    Graphiti handles entity extraction, relationship detection, embedding, and
    temporal supersession internally — no manual extraction step needed.
    """
    g = await get_graphiti()
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


async def search_memory(
    query: str,
    num_results: int = 10,
    group_id: str = "default",
    include_history: bool = False,
) -> str:
    """Hybrid search (BM25 + cosine + graph) across the knowledge graph.

    Returns a formatted block of facts with temporal metadata, ready to inject
    into an agent's context window.  By default only current (valid) edges are
    returned; pass include_history=True to include superseded facts annotated
    with their invalidation time.
    """
    g = await get_graphiti()
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


async def search_nodes(query: str, group_id: str = "default", limit: int = 10) -> str:
    """Search for entity nodes by name — returns unique entity names found in matching edges."""
    g = await get_graphiti()
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


async def get_entity_context(entity_name: str, group_id: str = "default") -> str:
    """Return all current facts/relationships involving a named entity."""
    g = await get_graphiti()
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
