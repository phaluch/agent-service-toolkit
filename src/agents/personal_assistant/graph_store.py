"""Kuzu-backed graph store for entity-relationship memory.

Schema
------
Node tables : Person, Project, Organization, Topic, Process
Rel tables  : WORKS_ON, WORKS_AT, KNOWS, USES, INTERESTED_IN,
              PART_OF, INVOLVES, RELATED_TO, MENTIONS
"""

import asyncio
import logging
from pathlib import Path
from typing import Literal

import kuzu

logger = logging.getLogger(__name__)

GRAPH_DIR = "./data/personal_assistant_graph"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

EntityTypeLiteral = Literal["person", "project", "organization", "topic", "process"]
RelTypeLiteral = Literal[
    "WORKS_ON",
    "WORKS_AT",
    "KNOWS",
    "USES",
    "INTERESTED_IN",
    "PART_OF",
    "INVOLVES",
    "RELATED_TO",
    "MENTIONS",
]

# Maps our lowercase entity_type → Kuzu table name
_ENTITY_TABLE: dict[str, str] = {
    "person": "Person",
    "project": "Project",
    "organization": "Organization",
    "topic": "Topic",
    "process": "Process",
}

# Maps rel_type → (from_table, to_table) — used for validation and traversal
_REL_ENDPOINTS: dict[str, tuple[str, str]] = {
    "WORKS_ON": ("Person", "Project"),
    "WORKS_AT": ("Person", "Organization"),
    "KNOWS": ("Person", "Person"),
    "USES": ("Project", "Topic"),
    "INTERESTED_IN": ("Person", "Topic"),
    "PART_OF": ("Project", "Project"),
    "INVOLVES": ("Process", "Person"),
    "RELATED_TO": ("Topic", "Topic"),
    "MENTIONS": ("Person", "Topic"),
}

_DDL = [
    # Node tables
    "CREATE NODE TABLE IF NOT EXISTS Person("
    "name STRING, role STRING, context STRING, PRIMARY KEY(name))",
    "CREATE NODE TABLE IF NOT EXISTS Project("
    "name STRING, status STRING, description STRING, PRIMARY KEY(name))",
    "CREATE NODE TABLE IF NOT EXISTS Organization(name STRING, PRIMARY KEY(name))",
    "CREATE NODE TABLE IF NOT EXISTS Topic(name STRING, PRIMARY KEY(name))",
    "CREATE NODE TABLE IF NOT EXISTS Process("
    "name STRING, description STRING, PRIMARY KEY(name))",
    # Relationship tables
    "CREATE REL TABLE IF NOT EXISTS WORKS_ON(FROM Person TO Project, role STRING, since STRING)",
    "CREATE REL TABLE IF NOT EXISTS WORKS_AT(FROM Person TO Organization, role STRING)",
    "CREATE REL TABLE IF NOT EXISTS KNOWS(FROM Person TO Person, context STRING)",
    "CREATE REL TABLE IF NOT EXISTS USES(FROM Project TO Topic)",
    "CREATE REL TABLE IF NOT EXISTS INTERESTED_IN(FROM Person TO Topic)",
    "CREATE REL TABLE IF NOT EXISTS PART_OF(FROM Project TO Project)",
    "CREATE REL TABLE IF NOT EXISTS INVOLVES(FROM Process TO Person)",
    "CREATE REL TABLE IF NOT EXISTS RELATED_TO(FROM Topic TO Topic)",
    "CREATE REL TABLE IF NOT EXISTS MENTIONS(FROM Person TO Topic)",
]

# ---------------------------------------------------------------------------
# DB singleton
# ---------------------------------------------------------------------------

_db: kuzu.Database | None = None


def _get_db() -> kuzu.Database:
    global _db
    if _db is None:
        Path(GRAPH_DIR).parent.mkdir(parents=True, exist_ok=True)
        _db = kuzu.Database(GRAPH_DIR)
    return _db


def _get_conn() -> kuzu.Connection:
    return kuzu.Connection(_get_db())


# ---------------------------------------------------------------------------
# Schema initialisation (idempotent — called once per connection)
# ---------------------------------------------------------------------------


def _init_schema_sync() -> None:
    conn = _get_conn()
    for stmt in _DDL:
        conn.execute(stmt)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rows(result: kuzu.QueryResult) -> list[list]:
    rows = []
    while result.has_next():
        rows.append(result.get_next())
    return rows


def _merge_node_sync(table: str, name: str, extra: dict[str, str]) -> None:
    """Upsert a node: create if absent, update properties if present."""
    conn = _get_conn()
    assignments = ", ".join(f"n.{k} = ${k}" for k in extra)
    query = f"MERGE (n:{table} {{name: $name}})"
    if assignments:
        query += f" ON MATCH SET {assignments} ON CREATE SET {assignments}"
    conn.execute(query, {"name": name, **extra})


def _merge_rel_sync(
    from_table: str,
    from_name: str,
    to_table: str,
    to_name: str,
    rel_type: str,
    props: dict[str, str],
) -> None:
    """Create relationship if it does not already exist."""
    conn = _get_conn()

    # Ensure both endpoints exist (MERGE with just name)
    conn.execute(f"MERGE (n:{from_table} {{name: $name}})", {"name": from_name})
    conn.execute(f"MERGE (n:{to_table} {{name: $name}})", {"name": to_name})

    # Check whether the relationship already exists
    check = conn.execute(
        f"MATCH (s:{from_table} {{name: $src}})-[r:{rel_type}]->(t:{to_table} {{name: $tgt}}) "
        "RETURN count(r) AS cnt",
        {"src": from_name, "tgt": to_name},
    )
    cnt = check.get_next()[0] if check.has_next() else 0
    if cnt > 0:
        return

    prop_str = ", ".join(f"{k}: ${k}" for k in props)
    rel_literal = f"[:{rel_type} {{{prop_str}}}]" if prop_str else f"[:{rel_type}]"
    conn.execute(
        f"MATCH (s:{from_table} {{name: $src}}), (t:{to_table} {{name: $tgt}}) "
        f"CREATE (s)-{rel_literal}->(t)",
        {"src": from_name, "tgt": to_name, **props},
    )


# ---------------------------------------------------------------------------
# Public sync helpers (wrapped async below)
# ---------------------------------------------------------------------------


def _upsert_entities_sync(entities: list[dict]) -> int:
    """entities: list of {name, entity_type, properties}."""
    _init_schema_sync()
    stored = 0
    for e in entities:
        table = _ENTITY_TABLE.get((e.get("entity_type") or "").lower())
        if not table:
            logger.warning("Unknown entity_type %r — skipping", e.get("entity_type"))
            continue
        name = (e.get("name") or "").strip()
        if not name:
            continue
        props = {k: str(v) for k, v in (e.get("properties") or {}).items() if v}
        try:
            _merge_node_sync(table, name, props)
            stored += 1
        except Exception:
            logger.exception("Failed to upsert entity %r", name)
    return stored


def _upsert_relationships_sync(relationships: list[dict]) -> int:
    """relationships: list of {source, source_type, target, target_type, rel_type, properties}."""
    _init_schema_sync()
    stored = 0
    for r in relationships:
        rel_type = (r.get("rel_type") or "").upper()
        if rel_type not in _REL_ENDPOINTS:
            logger.warning("Unknown rel_type %r — skipping", rel_type)
            continue
        src_table = _ENTITY_TABLE.get((r.get("source_type") or "").lower())
        tgt_table = _ENTITY_TABLE.get((r.get("target_type") or "").lower())
        if not src_table or not tgt_table:
            logger.warning("Unknown entity type in relationship — skipping")
            continue
        src = (r.get("source") or "").strip()
        tgt = (r.get("target") or "").strip()
        if not src or not tgt:
            continue
        props = {k: str(v) for k, v in (r.get("properties") or {}).items() if v}
        try:
            _merge_rel_sync(src_table, src, tgt_table, tgt, rel_type, props)
            stored += 1
        except Exception:
            logger.exception("Failed to upsert relationship %r→%r (%s)", src, tgt, rel_type)
    return stored


def _get_entity_neighborhood_sync(entity_name: str) -> str:
    """Return a human-readable summary of an entity and its direct relationships."""
    _init_schema_sync()
    conn = _get_conn()

    lines: list[str] = []

    # Find which table the entity lives in and fetch its properties
    found_table: str | None = None
    for entity_type, table in _ENTITY_TABLE.items():
        result = conn.execute(f"MATCH (n:{table} {{name: $name}}) RETURN n.*", {"name": entity_name})
        if result.has_next():
            row = result.get_next()
            cols = result.get_column_names()
            props = {c.replace("n.", ""): v for c, v in zip(cols, row) if v and c != "n.name"}
            prop_str = ", ".join(f"{k}: {v}" for k, v in props.items()) if props else ""
            lines.append(f"[{table}] {entity_name}" + (f" ({prop_str})" if prop_str else ""))
            found_table = table
            break

    if not found_table:
        return f"No entity named '{entity_name}' found in the knowledge graph."

    lines.append("Relationships:")

    # Outgoing
    for rel_type, (from_table, to_table) in _REL_ENDPOINTS.items():
        if from_table != found_table:
            continue
        result = conn.execute(
            f"MATCH (s:{from_table} {{name: $name}})-[r:{rel_type}]->(t:{to_table}) "
            "RETURN t.name",
            {"name": entity_name},
        )
        for row in _rows(result):
            lines.append(f"  → [{rel_type}] {row[0]} ({to_table})")

    # Incoming
    for rel_type, (from_table, to_table) in _REL_ENDPOINTS.items():
        if to_table != found_table:
            continue
        result = conn.execute(
            f"MATCH (s:{from_table})-[r:{rel_type}]->(t:{to_table} {{name: $name}}) "
            "RETURN s.name",
            {"name": entity_name},
        )
        for row in _rows(result):
            lines.append(f"  ← [{rel_type}] {row[0]} ({from_table})")

    return "\n".join(lines) if len(lines) > 2 else f"{lines[0]}\n  (no relationships yet)"


def _search_entities_sync(query: str, limit: int = 10) -> str:
    """Case-insensitive substring search across all node tables."""
    _init_schema_sync()
    conn = _get_conn()

    results: list[str] = []
    for entity_type, table in _ENTITY_TABLE.items():
        result = conn.execute(
            f"MATCH (n:{table}) WHERE lower(n.name) CONTAINS lower($q) RETURN n.name LIMIT $lim",
            {"q": query, "lim": limit},
        )
        for row in _rows(result):
            results.append(f"[{table}] {row[0]}")

    if not results:
        return f"No entities matching '{query}' found in the knowledge graph."
    return "\n".join(results)


# ---------------------------------------------------------------------------
# Async public API
# ---------------------------------------------------------------------------


async def upsert_entities(entities: list[dict]) -> int:
    """Async wrapper — upsert a list of entity dicts into the graph."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _upsert_entities_sync, entities)


async def upsert_relationships(relationships: list[dict]) -> int:
    """Async wrapper — upsert a list of relationship dicts into the graph."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _upsert_relationships_sync, relationships)


async def get_entity_neighborhood(entity_name: str) -> str:
    """Return entity properties + all direct relationships as formatted text."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_entity_neighborhood_sync, entity_name)


async def search_entities(query: str, limit: int = 10) -> str:
    """Substring search across all node tables; returns formatted list."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _search_entities_sync, query, limit)


def _dump_graph_sync() -> dict:
    """Return all entities and relationships as structured dicts for observability."""
    _init_schema_sync()
    conn = _get_conn()

    entities = []
    for table in _ENTITY_TABLE.values():
        result = conn.execute(f"MATCH (n:{table}) RETURN n.*")
        cols = [c.replace("n.", "") for c in result.get_column_names()]
        while result.has_next():
            row = result.get_next()
            props = {k: v for k, v in zip(cols, row) if v is not None}
            entities.append({"node_type": table, **props})

    relationships = []
    for rel_type, (from_table, to_table) in _REL_ENDPOINTS.items():
        result = conn.execute(
            f"MATCH (s:{from_table})-[:{rel_type}]->(t:{to_table}) RETURN s.name, t.name"
        )
        while result.has_next():
            row = result.get_next()
            relationships.append(
                {
                    "source": row[0],
                    "source_type": from_table,
                    "target": row[1],
                    "target_type": to_table,
                    "rel_type": rel_type,
                }
            )

    return {"entities": entities, "relationships": relationships}


async def dump_graph() -> dict:
    """Return all graph entities and relationships for observability."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _dump_graph_sync)
