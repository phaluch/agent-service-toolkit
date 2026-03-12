"""ChromaDB-backed knowledge store for the personal assistant.

All documents carry `is_valid` (1=current, 0=historical) in their metadata.
By default reads filter to is_valid=1. Pass include_history=True to see the
full temporal record.
"""

import logging
import uuid
from datetime import datetime, timezone

from langchain_chroma import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

PERSIST_DIR = "./data/personal_assistant_kb"
COLLECTION_NAME = "personal_assistant_knowledge"


def _get_embeddings():
    """Return embeddings, raising clearly if no supported key is configured."""
    try:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings()
    except Exception:
        pass
    raise RuntimeError(
        "No supported embeddings provider found. "
        "Set OPENAI_API_KEY to enable the personal assistant knowledge store."
    )


def get_store() -> Chroma:
    embeddings = _get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )


async def store_facts(facts: list[dict]) -> None:
    """Add a list of fact dicts to the knowledge store.

    Each fact dict should have:
      - content (str): full self-contained sentence
      - entity_type (str): person | project | process | general
      - entity_name (str): primary entity label (e.g. "Paulo")

    All facts are stored with is_valid=1 (current).
    """
    if not facts:
        return
    try:
        store = get_store()
        now = datetime.now(timezone.utc).isoformat()
        docs = [
            Document(
                page_content=f["content"],
                metadata={
                    "entity_type": f.get("entity_type", "general"),
                    "entity_name": f.get("entity_name", ""),
                    "insertion_time": now,
                    "is_valid": 1,
                    "invalidated_at": "",
                },
            )
            for f in facts
        ]
        ids = [str(uuid.uuid4()) for _ in docs]
        await store.aadd_documents(docs, ids=ids)
        logger.info("CHROMA store_facts: stored %d fact(s)", len(docs))
        for doc in docs:
            logger.info(
                "  [%s/%s] %s",
                doc.metadata["entity_type"],
                doc.metadata["entity_name"],
                doc.page_content[:100],
            )
    except Exception as e:
        logger.error(f"Failed to store knowledge facts: {e}")


async def invalidate_facts(entity_name: str) -> int:
    """Mark all valid facts for the given entity_name as historical (is_valid=0).

    Returns the number of facts invalidated.
    """
    try:
        store = get_store()
        now = datetime.now(timezone.utc).isoformat()

        # Retrieve all valid docs for this entity (no similarity filter — get by metadata)
        results = store._collection.get(
            where={"$and": [{"entity_name": {"$eq": entity_name}}, {"is_valid": {"$eq": 1}}]}
        )
        ids = results.get("ids", [])
        logger.info("CHROMA invalidate_facts: entity=%r found %d valid fact(s)", entity_name, len(ids))
        if not ids:
            return 0

        existing_metadatas = results.get("metadatas", [{}] * len(ids))
        updated_metadatas = [
            {**m, "is_valid": 0, "invalidated_at": now} for m in existing_metadatas
        ]
        store._collection.update(ids=ids, metadatas=updated_metadatas)
        logger.info("CHROMA invalidate_facts: marked %d fact(s) as historical for %r", len(ids), entity_name)
        return len(ids)
    except Exception as e:
        logger.error(f"Failed to invalidate facts for '{entity_name}': {e}")
        return 0


async def retrieve_facts(
    query: str, k: int = 5, include_history: bool = False
) -> list[Document]:
    """Return the k most relevant documents for the given query.

    By default only valid (current) facts are returned, sorted newest first.
    Pass include_history=True to include invalidated facts (annotated in metadata).
    """
    try:
        store = get_store()
        filter_arg = None if include_history else {"is_valid": {"$eq": 1}}
        logger.info(
            "CHROMA retrieve_facts: query=%r k=%d include_history=%s",
            query[:80], k, include_history,
        )
        docs = await store.asimilarity_search(query, k=k, filter=filter_arg)
        docs.sort(key=lambda d: d.metadata.get("insertion_time", ""), reverse=True)
        logger.info("CHROMA retrieve_facts: %d doc(s) returned", len(docs))
        for doc in docs:
            logger.info(
                "  [%s/%s] valid=%s %s",
                doc.metadata.get("entity_type", "?"),
                doc.metadata.get("entity_name", "?"),
                doc.metadata.get("is_valid", "?"),
                doc.page_content[:100],
            )
        return docs
    except Exception as e:
        logger.error(f"Failed to retrieve knowledge: {e}")
        return []
