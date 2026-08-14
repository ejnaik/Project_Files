"""
Phase 5: API & Containerization (FastAPI)
=============================================
Stock Market Predictor & Financial Intelligence Agent

Wraps the compiled LangGraph agent (Phase 4) - which in turn calls the
XGBoost quantitative baseline (Phase 3) and the metadata-filtered Chroma
retriever (Phase 2) - behind a single POST /predict endpoint. A lifespan
handler warms up every heavy component (Keras classifier, vector store,
LangGraph agent/LLM client) once at process startup so the first real
request isn't the one paying model-load / client-construction latency.

This revision replaces the static MOCK_NEWS_ITEMS-only bootstrap with a
live ingestion pipeline (Approach 1): an in-process APScheduler background
job that periodically pulls fresh headlines via yfinance, tags each one
with Phase 1's Keras router, and upserts them into the running Chroma
store so Phase 2's retrieval always has reasonably current sentiment to
draw on - without needing a separate ingestion service/process.
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional

import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException
from langchain_core.documents import Document
from pydantic import BaseModel, Field

import langgraph_agent
import nlp_router
import vector_store

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# --------------------------------------------------------------------------- #
# Live news ingestion configuration
# --------------------------------------------------------------------------- #

TARGET_TICKERS = ["AAPL", "TSLA", "MSFT"]
NEWS_PER_TICKER = 8
INGESTION_INTERVAL_HOURS = 1

# Chroma's in-memory client isn't guaranteed safe for a write racing another
# write from a different thread (the scheduler runs fetch_and_index_news in
# its own worker thread, while FastAPI's sync route handlers - like
# /predict, indirectly via search_financial_news - run in Starlette's
# threadpool). This lock only guards the *write* path here; concurrent
# reads during a write are assumed acceptable at this prototype's scale. If
# you need stronger guarantees later, move this lock (or a read/write
# variant of it) into vector_store.py so every caller shares it.
_VECTOR_STORE_WRITE_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Live ingestion: fetch -> tag -> upsert
# --------------------------------------------------------------------------- #


def _extract_article_text_and_metadata(article: dict) -> Optional[dict]:
    """Defensively pull a headline (+ light metadata) out of one yfinance
    news article dict.

    yfinance's `Ticker.news` schema is undocumented and has changed across
    versions (this project could not verify the live response shape
    against a real network call in its dev sandbox - Yahoo Finance was
    unreachable there). This tries several known key paths - both the
    newer nested `content` shape and the older flat shape - and returns
    None if nothing usable is found, rather than raising. Adjust the key
    paths here if your installed yfinance version differs.
    """
    content = article.get("content") if isinstance(article.get("content"), dict) else article

    title = content.get("title") or article.get("title")
    if not title or not str(title).strip():
        return None

    summary = content.get("summary") or content.get("description") or ""
    text = f"{title}. {summary}".strip() if summary else str(title).strip()

    link = (
        (content.get("canonicalUrl") or {}).get("url")
        or (content.get("clickThroughUrl") or {}).get("url")
        or article.get("link")
        or ""
    )
    publisher = (
        (content.get("provider") or {}).get("displayName")
        or article.get("publisher")
        or "Unknown"
    )
    published_at = (
        content.get("pubDate")
        or content.get("displayTime")
        or article.get("providerPublishTime")
        or ""
    )
    article_id = (
        article.get("id") or content.get("id") or article.get("uuid") or link or title
    )

    return {
        "text": text,
        "link": str(link),
        "publisher": str(publisher),
        "published_at": str(published_at),
        "article_id": str(article_id),
    }


def fetch_and_index_news() -> None:
    """Background ingestion job: pull the latest headlines for
    TARGET_TICKERS via yfinance, tag each one with Phase 1's Keras router,
    convert to LangChain Documents, and upsert them into the live Chroma
    store (Phase 2) so search_financial_news always has reasonably fresh
    sentiment to draw on.

    Designed to never raise: a network blip, a malformed article, or an
    embeddings-API hiccup is caught and logged so a single bad run can't
    crash the APScheduler thread or take down future scheduled runs.
    """
    try:
        documents: list[Document] = []
        ids: list[str] = []

        for ticker in TARGET_TICKERS:
            try:
                raw_articles = yf.Ticker(ticker).news or []
            except Exception:
                logger.exception(
                    "fetch_and_index_news: failed to fetch news for %s; "
                    "skipping this ticker for this run.", ticker,
                )
                continue

            for article in raw_articles[:NEWS_PER_TICKER]:
                try:
                    parsed = _extract_article_text_and_metadata(article)
                    if parsed is None or not parsed["text"]:
                        continue

                    topic = nlp_router.tag_financial_news(parsed["text"])

                    documents.append(
                        Document(
                            page_content=parsed["text"],
                            metadata={
                                "topic": topic,
                                "ticker": ticker,
                                "source": parsed["publisher"],
                                "url": parsed["link"],
                                "published_at": parsed["published_at"],
                            },
                        )
                    )
                    # Stable, content-derived id -> re-ingesting the same
                    # article on the next hourly run upserts it in place
                    # instead of duplicating it in the vector store.
                    ids.append(f"{ticker}:{parsed['article_id']}")
                except Exception:
                    logger.exception(
                        "fetch_and_index_news: failed to process one article "
                        "for %s; skipping it.", ticker,
                    )
                    continue

        if not documents:
            logger.warning(
                "fetch_and_index_news: no usable articles parsed this run; "
                "nothing indexed."
            )
            return

        store = vector_store._ACTIVE_STORE  # module-level store from Phase 2
        if store is None:
            logger.warning(
                "fetch_and_index_news: vector store not initialized yet; "
                "skipping this ingestion run."
            )
            return

        with _VECTOR_STORE_WRITE_LOCK:
            store.add_documents(documents, ids=ids)  # upserts (see note below)

        message = f"Indexed {len(documents)} new articles into ChromaDB."
        logger.info(message)
        print(message)

    except Exception:
        # Final safety net: nothing above should escape, but a scheduled
        # job that raises is a scheduled job that (depending on APScheduler
        # config) may stop being rescheduled - so guarantee this function
        # never propagates.
        logger.exception("fetch_and_index_news: unhandled error during ingestion run.")


# NOTE on upsert semantics: langchain_community's Chroma.add_documents()
# calls the underlying chromadb collection's `.upsert(...)` (not `.add`)
# under the hood, so passing explicit `ids=` above already gives true
# upsert-by-id behavior - no separate delete-then-add step is needed.


# --------------------------------------------------------------------------- #
# Lifespan: warm up every heavy component, seed live data, start scheduler
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AlphaPredict API - warming up components...")

    # 1) Phase 1: Keras classifier + vectorizer + label encoder.
    nlp_router.warm_up()

    # 2) Phase 2: Chroma vector store, bootstrapped from the bundled mock
    #    corpus. This is a safety net, not the primary data source anymore:
    #    it guarantees the store is non-empty (Chroma.from_documents
    #    requires at least one document to initialize the collection) even
    #    if the live ingestion pass immediately below fails (e.g. no
    #    network yet at container start).
    vector_store.build_vector_store(vector_store.MOCK_NEWS_ITEMS)

    # 3) Phase 4: compiled LangGraph agent + tool-bound LLM client.
    langgraph_agent.warm_up()

    # 4) Live ingestion (Approach 1): run once synchronously so the store
    #    has real, current data immediately on startup rather than waiting
    #    up to an hour for the first scheduled run.
    logger.info("Running initial live news ingestion...")
    fetch_and_index_news()

    # 5) Schedule recurring ingestion in a background thread. BackgroundScheduler
    #    (as opposed to AsyncIOScheduler) runs jobs in their own worker
    #    thread pool, so hourly ingestion never blocks the FastAPI event
    #    loop from serving requests.
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        fetch_and_index_news,
        trigger="interval",
        hours=INGESTION_INTERVAL_HOURS,
        id="news_ingestion_job",
        replace_existing=True,
        max_instances=1,  # never let overlapping runs stack up
    )
    scheduler.start()
    logger.info(
        "Background news ingestion scheduler started (every %dh).",
        INGESTION_INTERVAL_HOURS,
    )

    logger.info("AlphaPredict API ready.")
    yield

    logger.info("Shutting down AlphaPredict API.")
    scheduler.shutdown()


app = FastAPI(
    title="AlphaPredict - Stock Market Predictor & Financial Intelligence Agent",
    description=(
        "Combines an XGBoost next-day price forecast with metadata-filtered "
        "retrieval over financial news to produce a synthesized investment brief."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #


class PredictRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, examples=["AAPL"])
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        examples=["What's your outlook for the next trading day?"],
    )


class PredictResponse(BaseModel):
    ticker: str
    query: str
    financial_brief: str


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/health")
def health() -> dict:
    """Liveness/readiness probe for orchestrators (Docker/K8s healthchecks)."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Run the AlphaPredict agent for a given ticker + user query and return
    the synthesized financial brief.

    The ticker is folded into the prompt explicitly (rather than relying on
    the free-text `query` to mention it) so the agent's tool calls are
    always anchored to the correct symbol, regardless of how the user
    phrases their question.
    """
    ticker = request.ticker.strip().upper()
    effective_query = (
        f"Provide your outlook for {ticker}. "
        f"Additional context/question from the user: {request.query.strip()}"
    )

    try:
        brief = langgraph_agent.run_financial_agent(effective_query)
    except ValueError as exc:
        # Bad input surfaced by our own validation (e.g. empty query,
        # ticker yfinance can't resolve to usable data).
        logger.warning("Validation error for ticker=%s: %s", ticker, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EnvironmentError as exc:
        # Missing GOOGLE_API_KEY or similar misconfiguration - this is an
        # operator/config problem, not a client error.
        logger.error("Configuration error: %s", exc)
        raise HTTPException(
            status_code=503, detail="Service is misconfigured and cannot process requests."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        # Anything else (yfinance/Gemini connectivity, unexpected tool
        # failures, etc.) - don't leak internals to the client, but log
        # the full detail server-side for debugging.
        logger.exception("Unhandled error running agent for ticker=%s", ticker)
        raise HTTPException(
            status_code=502, detail="Upstream data/model provider error. Please retry shortly."
        ) from exc

    return PredictResponse(ticker=ticker, query=request.query, financial_brief=brief)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
