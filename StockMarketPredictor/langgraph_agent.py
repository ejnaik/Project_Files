"""
Phase 4: Agentic Orchestration (LangGraph)
=============================================
Stock Market Predictor & Financial Intelligence Agent

Wires the Phase 3 quantitative baseline and Phase 2 metadata-filtered
vector search together as tools for a tool-calling LLM, orchestrated as
a LangGraph state machine: agent -> (tools -> agent)* -> END. The agent
is instructed to always ground its final answer in BOTH the numerical
forecast and retrieved qualitative sentiment before responding.
"""

from __future__ import annotations

import logging
import os
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from quant_baseline import predict_stock_baseline
from vector_store import search_financial_news as _vector_search

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# "gemini-3.6-flash" is Google's current fast/cheap chat model as of this
# writing, and supports tool calling via .bind_tools(). Override via env var
# if you want a different Gemini model (e.g. a larger one for higher-stakes
# analysis) without editing code.
LLM_MODEL = os.getenv("GOOGLE_CHAT_MODEL", "gemini-3.6-flash")
LLM_TEMPERATURE = 0
RECURSION_LIMIT = 15  # generous headroom for a 2-tool agent loop


# --------------------------------------------------------------------------- #
# Tools (wrap Phase 3 + Phase 2 functions for the LLM)
# --------------------------------------------------------------------------- #


@tool
def get_quantitative_forecast(ticker: str) -> dict:
    """Run the XGBoost quantitative baseline for a stock ticker and return
    a forward-looking, next-trading-day closing price forecast.

    ALWAYS call this FIRST for any request that asks for a price
    prediction, forecast, or investment view on a specific ticker -
    the numerical target it returns anchors the rest of the analysis.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL", "TSLA", "MSFT".

    Returns:
        A dict with keys:
          - ticker: the symbol, uppercased
          - current_price: today's actual closing price
          - predicted_price: forecast for the next trading day's close
          - predicted_change_pct: implied % move from current to predicted
          - as_of_date: ISO date the forecast is anchored to (today)
    """
    logger.info("[tool] get_quantitative_forecast(ticker=%s)", ticker)
    return predict_stock_baseline(ticker)


@tool
def search_financial_news(query: str, topic: Optional[str] = None) -> list[dict]:
    """Semantic search over the indexed financial news/tweet corpus to find
    qualitative sentiment, catalysts, or shocks relevant to a stock or theme.

    Call this AFTER get_quantitative_forecast to check whether recent news
    supports, contradicts, or explains the numerical forecast (e.g. an
    upcoming earnings beat, a regulatory recall, a macro rate-cut signal).

    Args:
        query: Natural-language search query, e.g. "Apple iPhone demand
            earnings outlook".
        topic: Optional EXACT-match topic filter corresponding to the
            internal news classifier's category labels (e.g. "Earnings",
            "M&A", "Macroeconomics", "Regulatory/Recall"). Supplying a
            topic narrows the search to just that category and prevents
            unrelated stories from leaking in. If you are not confident
            which exact category applies, omit topic and search the
            full corpus instead of guessing.

    Returns:
        A list of {"topic": str, "text": str} dicts, most relevant first.
        Returns a single info dict if nothing matched.
    """
    logger.info("[tool] search_financial_news(query=%r, topic=%r)", query, topic)
    results = _vector_search(query=query, topic=topic)

    if not results:
        scope = f"topic={topic!r}" if topic else "the full corpus"
        return [{"info": f"No indexed news matched query={query!r} within {scope}."}]

    return [{"topic": r.topic, "text": r.text} for r in results]


TOOLS = [get_quantitative_forecast, search_financial_news]


# --------------------------------------------------------------------------- #
# Agent state
# --------------------------------------------------------------------------- #


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# --------------------------------------------------------------------------- #
# System prompt + LLM
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are AlphaPredict, an elite quantitative analyst at a \
systematic trading desk. You produce concise, data-driven investment briefs \
by combining a numerical price forecast with qualitative market context.

For every ticker- or investment-related question, follow this exact process:

1. Call `get_quantitative_forecast` FIRST to obtain the forward-looking, \
next-trading-day price target for the ticker. This number is your anchor - \
never state a price view without it.

2. Call `search_financial_news` to check for recent qualitative sentiment, \
catalysts, or shocks (earnings surprises, M&A, regulatory action, macro \
shifts) that could support, contradict, or explain the quantitative \
forecast. Use a topic filter when you can confidently name the relevant \
category; otherwise search broadly.

3. Synthesize both signals into a single, cohesive brief. Explicitly state:
   - The current price and the forecast next-day price/direction, with the \
implied percentage move.
   - Whether the retrieved news corroborates or conflicts with that \
forecast, and why.
   - A short, clearly-labeled takeaway (e.g. "Near-term bias: cautiously \
bullish") - never a definitive buy/sell instruction.

Rules:
- Do not skip either tool call unless the user's question is unrelated to a \
specific ticker's price outlook.
- If a tool returns an error or no data, say so plainly rather than \
guessing or fabricating numbers.
- Keep the final brief tight: prefer 4-8 sentences of dense, specific \
analysis over generic commentary.
- You are not a licensed financial advisor; frame output as analysis, not \
personalized investment advice, and note that markets carry risk.
"""


def _build_llm_with_tools() -> ChatGoogleGenerativeAI:
    # langchain_google_genai reads GOOGLE_API_KEY (or GEMINI_API_KEY) from
    # the environment itself, but we check explicitly here so a missing key
    # fails fast at warm-up with a clear message instead of a confusing SDK
    # error on the first tool-calling request.
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Export it before running the agent."
        )
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    return llm.bind_tools(TOOLS)


# Lazily built + cached so importing this module doesn't require an API key
# (useful for tests/Phase 5 startup ordering), and so repeated calls to
# run_financial_agent() don't re-instantiate the client every time.
_llm_with_tools: Optional[ChatGoogleGenerativeAI] = None


def _get_llm_with_tools() -> ChatGoogleGenerativeAI:
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = _build_llm_with_tools()
    return _llm_with_tools


# --------------------------------------------------------------------------- #
# Graph nodes
# --------------------------------------------------------------------------- #


def agent_node(state: AgentState) -> dict:
    """Invoke the tool-bound LLM with the system prompt + running message
    history, and append its response (which may be a tool call or a final
    answer) to state.
    """
    llm = _get_llm_with_tools()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #


def build_graph():
    """Construct and compile the AlphaPredict state machine:

        START -> agent -> (tools_condition) -> tools -> agent -> ... -> END

    `tools_condition` (from langgraph.prebuilt) inspects the agent's last
    message: if it contains tool_calls, route to "tools"; otherwise route
    to END. This gives the agent a natural multi-hop loop (forecast tool,
    then news tool, then a final synthesized answer) without hand-rolled
    routing logic.
    """
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # agent -> tools | END
    graph.add_edge("tools", "agent")

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    """Return the cached compiled graph, building it on first use. Exposed
    for Phase 5 (FastAPI) to warm up at startup rather than on first request.
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def warm_up() -> None:
    """Eagerly build the compiled graph and construct the tool-bound LLM
    client. Call this at API startup (see Phase 5's FastAPI lifespan) so
    the first real /predict request doesn't pay graph-compilation or
    client-construction cost, and so a missing GOOGLE_API_KEY surfaces as
    a loud startup failure instead of a confusing first-request 500.
    """
    get_compiled_graph()
    _get_llm_with_tools()
    logger.info("AlphaPredict agent warmed up and ready.")


# --------------------------------------------------------------------------- #
# Execution entry point
# --------------------------------------------------------------------------- #


def run_financial_agent(user_query: str) -> str:
    """Run the AlphaPredict agent end-to-end on a natural-language query.

    Parameters
    ----------
    user_query : str
        e.g. "What's your outlook on AAPL for tomorrow?"

    Returns
    -------
    str
        The final AI message's text content (the synthesized brief).

    Note on provider content shapes: `AIMessage.content` is NOT always a
    plain string. ChatOpenAI typically returns one, but Gemini (via
    langchain_google_genai) - and other providers - can return a list of
    content blocks instead, e.g. `[{"type": "text", "text": "..."}]`, even
    for a single-part response. Returning `.content` directly broke here
    once the LLM was swapped to Gemini: FastAPI's `PredictResponse` model
    requires `financial_brief: str`, and Pydantic rejected the raw list
    with a validation error. `AIMessage.text` (a langchain_core property,
    not `.content`) normalizes both shapes into a plain string, so we use
    that instead - this fix belongs here rather than in main.py, since
    main.py should not need to know about a specific provider's message
    internals.
    """
    if not user_query or not user_query.strip():
        raise ValueError("run_financial_agent received an empty query.")

    graph = get_compiled_graph()
    initial_state: AgentState = {"messages": [HumanMessage(content=user_query)]}

    logger.info("Running AlphaPredict agent on query: %r", user_query)
    final_state = graph.invoke(initial_state, config={"recursion_limit": RECURSION_LIMIT})

    final_message = final_state["messages"][-1]
    # str(...) here because `.text` returns langchain_core's TextAccessor -
    # a str subclass, functionally identical, but wrapping it guarantees a
    # plain builtins.str crosses the function boundary rather than leaking
    # an internal LangChain type to callers (e.g. Pydantic's `str` field).
    return str(final_message.text)


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Requires GOOGLE_API_KEY, a live network path to yfinance, and a
    # vector store already populated via vector_store.build_vector_store(...)
    # (see Phase 2) so search_financial_news has something to retrieve.
    brief = run_financial_agent("What is your outlook on AAPL for the next trading day?")
    print(brief)
