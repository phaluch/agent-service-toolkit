"""ChromaDB-backed knowledge store for the personal assistant."""

import logging
import uuid

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
    """
    if not facts:
        return
    try:
        store = get_store()
        docs = [
            Document(
                page_content=f["content"],
                metadata={
                    "entity_type": f.get("entity_type", "general"),
                    "entity_name": f.get("entity_name", ""),
                },
            )
            for f in facts
        ]
        ids = [str(uuid.uuid4()) for _ in docs]
        await store.aadd_documents(docs, ids=ids)
        logger.info(f"Stored {len(docs)} knowledge facts")
    except Exception as e:
        logger.error(f"Failed to store knowledge facts: {e}")


async def retrieve_facts(query: str, k: int = 5) -> list[Document]:
    """Return the k most relevant documents for the given query."""
    try:
        store = get_store()
        return await store.asimilarity_search(query, k=k)
    except Exception as e:
        logger.error(f"Failed to retrieve knowledge: {e}")
        return []
