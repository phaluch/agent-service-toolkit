"""Kuzu-backed graph store for entity-relationship memory.

Schema
------
Node tables : Person, Project, Organization, Topic, Process
Rel tables  : WORKS_ON, WORKS_AT, KNOWS, USES, INTERESTED_IN,
              PART_OF, INVOLVES, RELATED_TO, MENTIONS

All relationship tables carry `is_valid BOOLEAN` and `invalidated_at STRING`.
By default queries filter to is_valid=true. Pass include_history=True to see
the full temporal record.
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import kuzu

logger = logging.getLogger(__name__)

GRAPH_DIR = "./data/personal_assistant_graph/db"

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
    # Relationship tables — all carry is_valid + invalidated_at for temporal tracking
    "CREATE REL TABLE IF NOT EXISTS WORKS_ON("
    "FROM Person TO Project, role STRING, since STRING, is_valid BOOLEAN, invalidated_at STRING)",
    "CREATE REL TABLE IF NOT EXISTS WORKS_AT("
    "FROM Person TO Organization, role STRING, is_valid BOOLEAN, invalidated_at STRING)",
    "CREATE REL TABLE IF NOT EXISTS KNOWS("
    "FROM Person TO Person, context STRING, is_valid BOOLEAN, invalidated_at STRING)",
    "CREATE REL TABLE IF NOT EXISTS USES("
    "FROM Project TO Topic, is_valid BOOLEAN, invalidated_at STRING)",
    "CREATE REL TABLE IF NOT EXISTS INTERESTED_IN("
    "FROM Person TO Topic, is_valid BOOLEAN, invalidated_at STRING)",
    "CREATE REL TABLE IF NOT EXISTS PART_OF("
    "FROM Project TO Project, is_valid BOOLEAN, invalidated_at STRING)",
    "CREATE REL TABLE IF NOT EXISTS INVOLVES("
    "FROM Process TO Person, is_valid BOOLEAN, invalidated_at STRING)",
    "CREATE REL TABLE IF NOT EXISTS RELATED_TO("
    "FROM Topic TO Topic, is_valid BOOLEAN, invalidated_at STRING)",
    "CREATE REL TABLE IF NOT EXISTS MENTIONS("
    "FROM Person TO Topic, is_valid BOOLEAN, invalidated_at STRING)",
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
    logger.info("KUZU upsert node [%s] %r props=%s", table, name, extra or "{}")


def _merge_rel_sync(
    from_table: str,
    from_name: str,
    to_table: str,
    to_name: str,
    rel_type: str,
    props: dict[str, str],
) -> None:
    """Create relationship if a valid one does not already exist."""
    conn = _get_conn()

    # Ensure both endpoints exist (MERGE with just name)
    conn.execute(f"MERGE (n:{from_table} {{name: $name}})", {"name": from_name})
    conn.execute(f"MERGE (n:{to_table} {{name: $name}})", {"name": to_name})

    # Check whether a valid relationship already exists
    check = conn.execute(
        f"MATCH (s:{from_table} {{name: $src}})-[r:{rel_type}]->(t:{to_table} {{name: $tgt}}) "
        "WHERE r.is_valid = true RETURN count(r) AS cnt",
        {"src": from_name, "tgt": to_name},
    )
    cnt = check.get_next()[0] if check.has_next() else 0
    if cnt > 0:
        logger.info(
            "KUZU rel already valid — skip: %r -[%s]-> %r", from_name, rel_type, to_name
        )
        return

    all_props = {"is_valid": True, "invalidated_at": "", **props}
    prop_str = ", ".join(f"{k}: ${k}" for k in all_props)
    conn.execute(
        f"MATCH (s:{from_table} {{name: $src}}), (t:{to_table} {{name: $tgt}}) "
        f"CREATE (s)-[:{rel_type} {{{prop_str}}}]->(t)",
        {"src": from_name, "tgt": to_name, **all_props},
    )
    logger.info("KUZU create rel: %r -[%s]-> %r props=%s", from_name, rel_type, to_name, props or "{}")


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


def _get_entity_neighborhood_sync(entity_name: str, include_history: bool = False) -> str:
    """Return a human-readable summary of an entity and its direct relationships.

    By default only valid (current) relationships are shown. Pass include_history=True
    to include invalidated relationships annotated with their invalidation timestamp.
    """
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
        logger.info("KUZU neighborhood: entity %r not found", entity_name)
        return f"No entity named '{entity_name}' found in the knowledge graph."

    logger.info("KUZU neighborhood: found %r in table %s", entity_name, found_table)
    validity_filter = "" if include_history else " WHERE r.is_valid = true"
    lines.append("Relationships:" if include_history else "Relationships (current):")

    # Outgoing
    for rel_type, (from_table, to_table) in _REL_ENDPOINTS.items():
        if from_table != found_table:
            continue
        result = conn.execute(
            f"MATCH (s:{from_table} {{name: $name}})-[r:{rel_type}]->(t:{to_table})"
            f"{validity_filter} RETURN t.name, r.is_valid, r.invalidated_at",
            {"name": entity_name},
        )
        for row in _rows(result):
            label = f"  → [{rel_type}] {row[0]} ({to_table})"
            if include_history and not row[1]:
                label += f" [invalidated: {row[2]}]"
            lines.append(label)

    # Incoming
    for rel_type, (from_table, to_table) in _REL_ENDPOINTS.items():
        if to_table != found_table:
            continue
        result = conn.execute(
            f"MATCH (s:{from_table})-[r:{rel_type}]->(t:{to_table} {{name: $name}})"
            f"{validity_filter} RETURN s.name, r.is_valid, r.invalidated_at",
            {"name": entity_name},
        )
        for row in _rows(result):
            label = f"  ← [{rel_type}] {row[0]} ({from_table})"
            if include_history and not row[1]:
                label += f" [invalidated: {row[2]}]"
            lines.append(label)

    result_str = "\n".join(lines) if len(lines) > 2 else f"{lines[0]}\n  (no relationships yet)"
    logger.info("KUZU neighborhood result for %r:\n%s", entity_name, result_str)
    return result_str


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
        logger.info("KUZU search_entities: no matches for %r", query)
        return f"No entities matching '{query}' found in the knowledge graph."
    logger.info("KUZU search_entities: %d match(es) for %r: %s", len(results), query, results)
    return "\n".join(results)


def _invalidate_rel_sync(
    source_table: str,
    source_name: str,
    rel_type: str,
    target_table: str,
    target_name: str | None = None,
) -> int:
    """Mark matching relationships as is_valid=false.

    If target_name is None, invalidates ALL valid relationships of rel_type
    originating from source_name (e.g. all current WORKS_ON edges for a person).
    Returns the number of relationships invalidated.
    """
    _init_schema_sync()
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()

    if target_name:
        result = conn.execute(
            f"MATCH (s:{source_table} {{name: $src}})-[r:{rel_type}]->(t:{target_table} {{name: $tgt}}) "
            "WHERE r.is_valid = true "
            "SET r.is_valid = false, r.invalidated_at = $now "
            "RETURN count(r)",
            {"src": source_name, "tgt": target_name, "now": now},
        )
    else:
        result = conn.execute(
            f"MATCH (s:{source_table} {{name: $src}})-[r:{rel_type}]->(t:{target_table}) "
            "WHERE r.is_valid = true "
            "SET r.is_valid = false, r.invalidated_at = $now "
            "RETURN count(r)",
            {"src": source_name, "now": now},
        )

    row = result.get_next() if result.has_next() else [0]
    count = row[0]
    logger.info(
        "KUZU invalidate_rel: %r -[%s]-> %r: %d relationship(s) marked invalid",
        source_name, rel_type, target_name or "*", count,
    )
    return count


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


async def get_entity_neighborhood(entity_name: str, include_history: bool = False) -> str:
    """Return entity properties + direct relationships as formatted text.

    Pass include_history=True to include invalidated (historical) relationships.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _get_entity_neighborhood_sync, entity_name, include_history
    )


async def invalidate_relationships(relationships: list[dict]) -> int:
    """Async wrapper — mark a list of relationships as invalid.

    Each dict should have: source, source_type, rel_type, target_type,
    and optionally target (if omitted, all valid rels of that type from source are invalidated).
    Returns total count of invalidated relationships.
    """
    loop = asyncio.get_event_loop()
    total = 0
    for r in relationships:
        src_table = _ENTITY_TABLE.get((r.get("source_type") or "").lower())
        tgt_table = _ENTITY_TABLE.get((r.get("target_type") or "").lower())
        rel_type = (r.get("rel_type") or "").upper()
        if not src_table or not tgt_table or rel_type not in _REL_ENDPOINTS:
            logger.warning("invalidate_relationships: skipping invalid entry %r", r)
            continue
        n = await loop.run_in_executor(
            None,
            _invalidate_rel_sync,
            src_table,
            r["source"],
            rel_type,
            tgt_table,
            r.get("target"),
        )
        total += n
        logger.info("Invalidated %d %s relationship(s) from %r", n, rel_type, r["source"])
    return total


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
