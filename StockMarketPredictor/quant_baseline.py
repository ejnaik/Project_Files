"""
Phase 3: The Quantitative Baseline (XGBoost)
==============================================
Stock Market Predictor & Financial Intelligence Agent

Downloads recent daily OHLCV data for a ticker, engineers a small set of
technical features, and trains a lightweight `XGBRegressor` to produce a
next-day closing-price forecast. This is the "numerical forecast" half
of the agent - Phase 4's LangGraph orchestrator combines this output
with the unstructured sentiment retrieved from Phase 2's vector store.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

LOOKBACK_PERIOD = "6mo"
FEATURE_COLUMNS = ["Close", "MA_7", "MA_14", "Volatility"]
TARGET_COLUMN = "Next_Close"

# Rolling windows need at least this many trading days of history before
# the first non-NaN feature row exists (14-day window is the binding
# constraint here). 6 months of daily data comfortably clears this.
MIN_ROWS_REQUIRED = 30

XGB_PARAMS = dict(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    objective="reg:squarederror",
    random_state=42,
)


# --------------------------------------------------------------------------- #
# Data ingestion + feature engineering
# --------------------------------------------------------------------------- #


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Recent yfinance versions can return MultiIndex columns (e.g. when
    the ticker level is included) even for a single-ticker download.
    Normalize to flat, single-level columns so downstream code can rely
    on `df["Close"]` unconditionally.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def prepare_stock_data(ticker: str) -> pd.DataFrame:
    """Download 6 months of daily OHLCV data and engineer features.

    Features
    --------
    - Close            : daily closing price
    - MA_7              : 7-day simple moving average of Close
    - MA_14             : 14-day simple moving average of Close
    - Volatility        : 14-day rolling std. dev. of daily returns
    - Next_Close (target): Close shifted so that each row's target is the
                            *following* trading day's close
                            (`Close.shift(-1)`)

    Rows with NaNs introduced by the rolling windows (the first ~13 rows)
    are dropped. The final row's Next_Close is deliberately LEFT as NaN
    and retained rather than dropped: that row is today's (the most
    recent trading day in this download), and its "next close" hasn't
    happened yet. Keeping it is what lets `predict_stock_baseline` use
    today's real features to forecast tomorrow's genuinely unknown close,
    instead of only ever being able to "predict" a day already present
    in the historical data.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "AAPL".

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe indexed by date with columns
        [Close, MA_7, MA_14, Volatility, Next_Close]. Every row has a
        valid Next_Close except the last, which is NaN by design (the
        row reserved for live inference).

    Raises
    ------
    ValueError
        If yfinance returns no data for the ticker, or too little history
        survives the rolling-window/dropna steps to train on.
    """
    logger.info("Downloading %s daily data (last %s)...", ticker, LOOKBACK_PERIOD)
    raw = yf.download(
        ticker,
        period=LOOKBACK_PERIOD,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if raw is None or raw.empty:
        raise ValueError(
            f"yfinance returned no data for ticker '{ticker}'. "
            "Check the symbol and your network connection."
        )

    raw = _flatten_yfinance_columns(raw)

    df = raw[["Close"]].copy()

    # -- Technical features -------------------------------------------------
    df["MA_7"] = df["Close"].rolling(window=7).mean()
    df["MA_14"] = df["Close"].rolling(window=14).mean()

    daily_returns = df["Close"].pct_change()
    df["Volatility"] = daily_returns.rolling(window=14).std()

    # -- Target: next trading day's close -----------------------------------
    df[TARGET_COLUMN] = df["Close"].shift(-1)

    df = df.dropna(subset=FEATURE_COLUMNS)  # keep the final NaN-target row on purpose

    if len(df) < MIN_ROWS_REQUIRED:
        raise ValueError(
            f"Only {len(df)} usable rows remain for '{ticker}' after feature "
            f"engineering (need >= {MIN_ROWS_REQUIRED}). Ticker may be too "
            "newly listed or thinly traded for this lookback window."
        )

    logger.info("Prepared %d rows of features for %s.", len(df), ticker)
    return df


# --------------------------------------------------------------------------- #
# Model training + prediction
# --------------------------------------------------------------------------- #


def predict_stock_baseline(ticker: str) -> dict:
    """Train a lightweight XGBoost baseline and forecast the next close.

    Design: this is a genuinely forward-looking forecast, not a backtest.
    `prepare_stock_data` retains today's row (the most recent trading day)
    with its Next_Close intentionally NaN, since tomorrow hasn't happened
    yet. We train on every row that DOES have a known target (i.e. every
    row except today's) and then run inference on today's real feature
    values to predict tomorrow's close. Train and inference rows are
    still strictly disjoint - today's row is never in `y_train` (its
    target is NaN) - it's just no longer a row from the historical past;
    it's the live edge of the dataset.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "AAPL".

    Returns
    -------
    dict
        {
            "ticker": str,
            "current_price": float,   # today's actual Close (last known price)
            "predicted_price": float, # model's forecast for tomorrow's close
            "predicted_change_pct": float,
            "as_of_date": str,        # ISO date of the inference row (today)
        }
    """
    df = prepare_stock_data(ticker)

    # Today's row (Next_Close is NaN by design) -> the live inference input.
    # Every other row has a known target and is fair game for training.
    inference_row = df.iloc[[-1]]  # keep as a 1-row DataFrame for predict()
    train_df = df.iloc[:-1]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    logger.info(
        "Training XGBRegressor on %d rows for %s...", len(X_train), ticker
    )
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)

    X_infer = inference_row[FEATURE_COLUMNS]
    predicted_price = float(model.predict(X_infer)[0])
    current_price = float(inference_row["Close"].iloc[0])

    result = {
        "ticker": ticker.upper(),
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "predicted_change_pct": round(
            (predicted_price - current_price) / current_price * 100, 2
        ),
        "as_of_date": inference_row.index[0].strftime("%Y-%m-%d"),
    }

    logger.info(
        "%s: current=%.2f predicted_next=%.2f (%.2f%%)",
        result["ticker"],
        result["current_price"],
        result["predicted_price"],
        result["predicted_change_pct"],
    )
    return result


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    for symbol in ["AAPL"]:
        try:
            forecast = predict_stock_baseline(symbol)
            print(forecast)
        except Exception as exc:  # noqa: BLE001 - smoke test, surface any failure
            print(f"Failed to forecast {symbol}: {exc}")
