"""
Chart Generation: Price History + Forecast Visualization
============================================================
Stock Market Predictor & Financial Intelligence Agent

Renders a PNG chart of a ticker's recent closing-price history (the same
data Phase 3's XGBoost model trains on) with the next-day forecast plotted
as a distinguished final point, and returns it as a base64-encoded data
URI ready to embed directly in the `/predict` JSON response or an
`<img src="...">` tag.

Runs on Matplotlib's non-interactive "Agg" backend, since this executes
inside a headless server process with no display attached - the backend
MUST be set before `matplotlib.pyplot` is imported anywhere in the
process, which is why it happens at module import time here, before the
`pyplot` import below.
"""

from __future__ import annotations

import base64
import io
import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

FIGURE_SIZE = (8, 4)
FIGURE_DPI = 120
HISTORY_COLOR = "#1f77b4"
FORECAST_COLOR = "#d62728"


def generate_forecast_chart_base64(
    ticker: str,
    price_history: pd.Series,
    predicted_price: float,
) -> str:
    """Render a price-history-plus-forecast line chart.

    Parameters
    ----------
    ticker : str
        Ticker symbol, used only for the chart title.
    price_history : pd.Series
        Historical `Close` prices, indexed by date - as returned by
        `quant_baseline.predict_stock_baseline_with_history`. The series'
        LAST point is treated as "today" (the live inference row) and is
        where the forecast line/marker is anchored from.
    predicted_price : float
        The model's forecast for the next trading day's close.

    Returns
    -------
    str
        A `data:image/png;base64,...` data URI string. Embed this directly
        as the `src` of an `<img>` tag, or return it as-is in a JSON field
        for a frontend to render.

    Notes
    -----
    Every call creates a Matplotlib Figure and explicitly closes it via
    `plt.close(fig)` before returning. Matplotlib figures are NOT garbage
    collected automatically just because a Python reference goes out of
    scope - in a long-running server process (this function is called once
    per `/predict` request), skipping `plt.close()` would leak memory on
    every single request. On a memory-constrained deployment target (see
    the project README's EC2 notes), that leak would eventually degrade or
    crash the service, so this cleanup is not optional.
    """
    if price_history is None or price_history.empty:
        raise ValueError("generate_forecast_chart_base64 received empty price_history.")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    try:
        ax.plot(
            price_history.index,
            price_history.values,
            color=HISTORY_COLOR,
            linewidth=1.5,
            label="Historical Close",
        )

        last_date = price_history.index[-1]
        last_price = float(price_history.iloc[-1])
        forecast_date = last_date + pd.Timedelta(days=1)

        # Dashed connector from the last known close to the forecast point,
        # plus a distinct marker, so the forecast is visually unmistakable
        # from the solid historical line rather than blending into it.
        ax.plot(
            [last_date, forecast_date],
            [last_price, predicted_price],
            color=FORECAST_COLOR,
            linewidth=1.5,
            linestyle="--",
        )
        ax.scatter(
            [forecast_date],
            [predicted_price],
            color=FORECAST_COLOR,
            zorder=5,
            s=40,
            label="Forecast (Next Close)",
        )

        ax.set_title(f"{ticker.upper()} - Close Price History with Next-Day Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
    finally:
        plt.close(fig)  # always release the figure, even if rendering raised

    return f"data:image/png;base64,{encoded}"
