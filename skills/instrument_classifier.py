"""
instrument_classifier.py
-----------------------
Detects what type of financial instrument a ticker represents (stock, ETF,
crypto, index, etc.) by reading yfinance's quoteType field.

Every other skill calls classify() first to route to the right analysis path.
For example, fundamental_analysis uses this to decide whether to look up
P/E ratios (stocks) or expense ratios (ETFs).

Usage:
    from skills.instrument_classifier import classify
    meta = classify("SPY")
    print(meta["type"])         # "etf"
    print(meta["display_name"]) # "SPDR S&P 500 ETF Trust"
"""

import yfinance as yf

_QUOTE_TYPE_MAP = {
    "EQUITY":         "stock",
    "ETF":            "etf",
    "MUTUALFUND":     "mutual_fund",
    "CRYPTOCURRENCY": "crypto",
    "CURRENCY":       "forex",
    "INDEX":          "index",
    "FUTURE":         "future",
    "OPTION":         "option",
}


def classify(ticker: str) -> dict:
    """
    Classify a ticker symbol and return metadata about the instrument.

    Returns a dict with:
        ticker        – uppercased symbol
        type          – one of: stock, etf, mutual_fund, crypto, forex,
                        index, future, option, other, unknown
        display_name  – full readable name (e.g. "Apple Inc.")
        exchange      – exchange code (e.g. "NMS", "PCX")
        currency      – trading currency (e.g. "USD")
        sector        – sector string for stocks (e.g. "Technology")
        industry      – industry string for stocks
        category      – Morningstar category for ETFs/funds
        fund_family   – fund provider (e.g. "Vanguard", "BlackRock")
        description   – first 400 chars of business/fund summary
    """
    ticker = ticker.upper().strip()

    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}

    # Treat as unknown if no price data came back at all
    has_price = (
        info.get("regularMarketPrice") is not None
        or info.get("currentPrice") is not None
        or info.get("navPrice") is not None
    )
    if not info or not has_price:
        return {"ticker": ticker, "type": "unknown", "display_name": ticker}

    raw = info.get("quoteType", "EQUITY")
    instrument_type = _QUOTE_TYPE_MAP.get(raw, "other")

    return {
        "ticker":       ticker,
        "type":         instrument_type,
        "quote_type_raw": raw,
        "display_name": info.get("longName") or info.get("shortName") or ticker,
        "exchange":     info.get("exchange", ""),
        "currency":     info.get("currency", "USD"),
        "sector":       info.get("sector", ""),
        "industry":     info.get("industry", ""),
        "category":     info.get("category", ""),      # Morningstar ETF category
        "fund_family":  info.get("fundFamily", ""),    # e.g. "Vanguard"
        "description":  (info.get("longBusinessSummary") or "")[:400],
    }
