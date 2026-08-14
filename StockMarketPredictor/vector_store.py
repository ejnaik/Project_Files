"""
Phase 2: Metadata-Filtered Vector Database (ChromaDB)
=======================================================
Stock Market Predictor & Financial Intelligence Agent

Takes news items tagged with a topic label (see Phase 1's
`tag_financial_news`), converts them into LangChain `Document` objects
with the topic stored strictly in `metadata`, indexes them into a local
in-memory Chroma vector store using Google Gemini embeddings, and exposes
a metadata-filtered retriever so the agent (Phase 4) can search *within*
a topic instead of over the whole corpus - this is what prevents
semantic hallucination (e.g. a "Merger" query pulling back irrelevant
"Earnings" chatter that happens to be lexically similar).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# "gemini-embedding-2-preview" is Google's current Gemini embedding model as
# of this writing (no "models/" prefix needed for langchain_google_genai).
# It's a preview model name - check https://ai.google.dev/gemini-api/docs/embeddings
# for the current stable identifier if this starts returning a not-found error.
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-2-preview")
DEFAULT_TOP_K = 4
COLLECTION_NAME = "financial_news"

# Module-level handle to the active in-memory store so `search_financial_news`
# can be called without threading a store object through every call site
# (e.g. from a LangGraph tool in Phase 4). `build_vector_store` sets this;
# tests/callers that want isolation can instead pass `vector_store=` explicitly.
_ACTIVE_STORE: Optional[Chroma] = None


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    # langchain_google_genai reads GOOGLE_API_KEY (or GEMINI_API_KEY) from
    # the environment itself, but we check explicitly here so a missing key
    # fails fast with a clear message instead of a confusing SDK error deep
    # inside the embed call.
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Export it before building the "
            "vector store (GoogleGenerativeAIEmbeddings requires it)."
        )
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


# --------------------------------------------------------------------------- #
# Document creation + indexing
# --------------------------------------------------------------------------- #


def _news_items_to_documents(news_items: list[dict]) -> list[Document]:
    """Convert raw news dicts into LangChain `Document` objects.

    Each `news_items` entry is expected to look like:
        {"text": "Apple revenue grows 12% YoY on iPhone demand", "topic": "Earnings"}

    Any extra keys (e.g. "source", "published_at", "ticker") are passed
    through into metadata as-is, so upstream enrichment isn't lost - but
    "topic" is always guaranteed to be present in metadata since that's
    the field the retriever filters on.
    """
    documents: list[Document] = []

    for i, item in enumerate(news_items):
        text = item.get("text")
        topic = item.get("topic")

        if not text or not text.strip():
            logger.warning("Skipping news_items[%d]: empty/missing 'text'.", i)
            continue
        if not topic:
            logger.warning(
                "news_items[%d] has no 'topic'; tag it with "
                "tag_financial_news() before indexing. Skipping.", i
            )
            continue

        metadata = {k: v for k, v in item.items() if k not in ("text",)}
        metadata["topic"] = topic  # ensure it's present and authoritative

        documents.append(Document(page_content=text.strip(), metadata=metadata))

    return documents


def build_vector_store(
    news_items: list[dict],
    embeddings: Optional[GoogleGenerativeAIEmbeddings] = None,
    persist: bool = False,
) -> Chroma:
    """Build a local, in-memory Chroma vector store from tagged news items.

    Parameters
    ----------
    news_items : list[dict]
        Each dict must have "text" and "topic"; extra keys become
        additional metadata.
    embeddings : GoogleGenerativeAIEmbeddings, optional
        Inject a pre-configured embeddings client (e.g. for testing with
        a stub). Defaults to `gemini-embedding-2-preview`.
    persist : bool
        If False (default), the store is purely in-memory and vanishes
        when the process exits - matches the spec's "local, in-memory"
        requirement. Set True only if you also pass a `persist_directory`
        via a custom Chroma client; left off by default to keep this
        phase's behavior explicit and ephemeral.

    Returns
    -------
    Chroma
        The populated vector store. Also cached module-globally so
        `search_financial_news` can use it without an explicit handle.
    """
    global _ACTIVE_STORE

    documents = _news_items_to_documents(news_items)
    if not documents:
        raise ValueError(
            "No valid documents to index - every news_item was missing "
            "'text' or 'topic'."
        )

    embeddings = embeddings or _get_embeddings()

    logger.info("Embedding and indexing %d documents into Chroma...", len(documents))
    store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        # No `persist_directory` -> Chroma runs purely in-memory (ephemeral).
    )

    _ACTIVE_STORE = store
    logger.info("Vector store ready with %d documents.", len(documents))
    return store


# --------------------------------------------------------------------------- #
# Filtered retrieval
# --------------------------------------------------------------------------- #


@dataclass
class RetrievedNews:
    text: str
    topic: str
    metadata: dict
    score: Optional[float] = None

    def __repr__(self) -> str:  # concise, readable in logs/tool output
        score_str = f" (score={self.score:.4f})" if self.score is not None else ""
        return f"[{self.topic}]{score_str} {self.text}"


def search_financial_news(
    query: str,
    topic: Optional[str] = None,
    k: int = DEFAULT_TOP_K,
    vector_store: Optional[Chroma] = None,
) -> list[RetrievedNews]:
    """Semantic search over indexed financial news, strictly scoped to a
    topic when one is provided.

    Parameters
    ----------
    query : str
        Natural-language search query.
    topic : str, optional
        If given, retrieval is constrained to documents whose
        `metadata["topic"]` exactly matches this value - this is the
        anti-hallucination guardrail: the agent (Phase 4) can't
        accidentally surface "M&A rumor" content when it asked for
        "Earnings" sentiment. If omitted, search runs over the full
        corpus.
    k : int
        Number of results to return.
    vector_store : Chroma, optional
        Store to search against. Defaults to the store most recently
        built by `build_vector_store`.

    Returns
    -------
    list[RetrievedNews]
        Formatted results (empty list if nothing matches the filter).
    """
    store = vector_store or _ACTIVE_STORE
    if store is None:
        raise RuntimeError(
            "No vector store available. Call build_vector_store(news_items) "
            "first, or pass vector_store= explicitly."
        )
    if not query or not query.strip():
        raise ValueError("search_financial_news received an empty query.")

    search_kwargs: dict = {"k": k}
    if topic:
        search_kwargs["filter"] = {"topic": topic}

    retriever = store.as_retriever(search_kwargs=search_kwargs)

    logger.info(
        "Searching (topic filter=%s, k=%d): %r", topic or "<none>", k, query
    )
    results = retriever.invoke(query)

    formatted = [
        RetrievedNews(
            text=doc.page_content,
            topic=doc.metadata.get("topic", "Unknown"),
            metadata=doc.metadata,
        )
        for doc in results
    ]

    if topic and not formatted:
        logger.info(
            "No results for topic=%r - either nothing was indexed under "
            "that label, or it doesn't match any tagged document exactly.",
            topic,
        )

    return formatted


# --------------------------------------------------------------------------- #
# Mock data + smoke test: verifies metadata filtering actually constrains
# results (not just semantic similarity doing the work by coincidence).
# --------------------------------------------------------------------------- #

MOCK_NEWS_ITEMS = [
    {
        "text": "Apple reports Q3 revenue up 12% YoY, beating analyst estimates on strong iPhone demand.",
        "topic": "Earnings",
        "ticker": "AAPL",
    },
    {
        "text": "Microsoft's quarterly earnings top expectations as Azure cloud revenue accelerates.",
        "topic": "Earnings",
        "ticker": "MSFT",
    },
    {
        "text": "Amazon posts disappointing quarterly profit, citing rising logistics costs.",
        "topic": "Earnings",
        "ticker": "AMZN",
    },
    {
        "text": "Federal Reserve signals possible rate cuts amid cooling inflation data.",
        "topic": "Macroeconomics",
        "ticker": None,
    },
    {
        "text": "US 10-year Treasury yield falls as investors price in a dovish Fed path.",
        "topic": "Macroeconomics",
        "ticker": None,
    },
    {
        "text": "Tesla recalls 200,000 vehicles over autopilot software bug, shares slide.",
        "topic": "Regulatory/Recall",
        "ticker": "TSLA",
    },
    {
        "text": "Chevron to acquire Hess Corporation in $53 billion all-stock merger deal.",
        "topic": "M&A",
        "ticker": "CVX",
    },
    {
        "text": "Exxon Mobil finalizes acquisition of Pioneer Natural Resources in major oil sector consolidation.",
        "topic": "M&A",
        "ticker": "XOM",
    },
]


def _run_smoke_test() -> None:
    """Build the store from mock data and confirm the topic filter actually
    excludes off-topic (but sometimes lexically-similar) documents.
    """
    build_vector_store(MOCK_NEWS_ITEMS)

    print("\n--- Unfiltered search: 'company financial results' ---")
    for r in search_financial_news("company financial results"):
        print(r)

    print("\n--- Filtered search: topic='Earnings', query='revenue growth' ---")
    for r in search_financial_news("revenue growth", topic="Earnings"):
        print(r)
        assert r.topic == "Earnings", "Metadata filter leaked a non-Earnings doc!"

    print("\n--- Filtered search: topic='M&A', query='corporate acquisition deal' ---")
    for r in search_financial_news("corporate acquisition deal", topic="M&A"):
        print(r)
        assert r.topic == "M&A", "Metadata filter leaked a non-M&A doc!"

    print("\n--- Filtered search that should return nothing: topic='Crypto' ---")
    empty = search_financial_news("bitcoin price surge", topic="Crypto")
    print(f"Result count: {len(empty)} (expected 0)")
    assert empty == []

    print("\nAll filter assertions passed - metadata filtering is enforced.")


if __name__ == "__main__":
    _run_smoke_test()
