"""
analyst_data.py -- Analyst consensus, price targets, earnings dates,
insider transactions, and institutional ownership via yfinance.
"""

from dotenv import load_dotenv
import os
import sys
from datetime import datetime, timezone, timedelta

import yfinance as yf

load_dotenv()


def _safe_get(d, key, default=None):
    """Return d[key] if it exists and is not None, else default."""
    if d is None:
        return default
    val = d.get(key)
    return val if val is not None else default


def _unix_to_date_str(unix_ts):
    """Convert a Unix timestamp (int) to 'YYYY-MM-DD' string, or None."""
    if unix_ts is None:
        return None
    try:
        return datetime.fromtimestamp(int(unix_ts), tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return None


def _get_institutional_pct(major_holders):
    """
    Extract institutional ownership percentage from major_holders DataFrame.
    Looks for a row whose Breakdown label contains 'institution' (case-insensitive).
    Returns a float 0-1 or None.
    """
    if major_holders is None:
        return None
    try:
        df = major_holders
        for idx in range(len(df)):
            try:
                row = df.iloc[idx]
                # DataFrame columns vary by yfinance version; try both layouts
                label = ""
                value_str = ""
                if len(row) >= 2:
                    label = str(row.iloc[1]).lower()
                    value_str = str(row.iloc[0])
                elif len(row) >= 1:
                    label = str(df.index[idx]).lower()
                    value_str = str(row.iloc[0])

                if "institution" in label:
                    value_str = value_str.replace("%", "").strip()
                    return float(value_str) / 100.0
            except Exception:
                continue
    except Exception:
        pass
    return None


def _get_insider_net_sentiment(insider_transactions):
    """
    Evaluate net insider buy/sell activity over the last 90 days.
    Returns 'buying', 'neutral', or 'selling'.
    """
    if insider_transactions is None:
        return "neutral"
    try:
        df = insider_transactions.copy()
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=90)

        # Normalise the date column
        date_col = None
        for candidate in ["Start Date", "startDate", "Date", "date"]:
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            return "neutral"

        df[date_col] = df[date_col].apply(
            lambda x: x.replace(tzinfo=timezone.utc) if hasattr(x, "replace") and x.tzinfo is None else x
        )
        recent = df[df[date_col] >= cutoff]

        if recent.empty:
            return "neutral"

        # Identify transaction type column
        txn_col = None
        for candidate in ["Transaction", "transaction"]:
            if candidate in recent.columns:
                txn_col = candidate
                break

        if txn_col is None:
            return "neutral"

        buys = recent[recent[txn_col].str.lower().str.contains("buy|purchase", na=False)]
        sells = recent[recent[txn_col].str.lower().str.contains("sell|sale", na=False)]

        net = len(buys) - len(sells)
        if net > 0:
            return "buying"
        elif net < 0:
            return "selling"
        else:
            return "neutral"
    except Exception:
        return "neutral"


def _score_analyst_rating(rec_mean):
    if rec_mean is None:
        return 0
    if rec_mean <= 2.0:
        return 2
    if rec_mean <= 3.0:
        return 1
    return 0


def _score_num_analysts(num):
    if num is None:
        return 0
    if num >= 20:
        return 2
    if num >= 10:
        return 1
    return 0


def _score_upside(upside_pct):
    """upside_pct is a decimal, e.g. 0.169 for 16.9%."""
    if upside_pct is None:
        return 0
    if upside_pct > 0.10:
        return 2
    if upside_pct > 0.0:
        return 1
    return 0


def _score_institutional(inst_pct):
    """inst_pct is 0-1."""
    if inst_pct is None:
        return 0
    if inst_pct > 0.60:
        return 2
    if inst_pct > 0.30:
        return 1
    return 0


def _score_insider(sentiment):
    if sentiment == "buying":
        return 2
    if sentiment == "neutral":
        return 1
    return 0  # selling


def _signal_label(score):
    if score >= 8:
        return "VERY BULLISH. Strong analyst conviction and large upside."
    if score >= 6:
        return "BULLISH. Analysts are constructive with meaningful upside."
    if score >= 4:
        return "MIXED. Some analyst support but limited conviction."
    if score >= 2:
        return "CAUTIOUS. Analysts are lukewarm or see limited upside."
    return "BEARISH. Analysts see downside or coverage is thin."


def analyze(ticker: str) -> dict:
    """
    Fetch analyst consensus data for *ticker* and return a result dict.

    Keys
    ----
    ticker, num_analysts, recommendation_key, recommendation_mean,
    target_mean, target_high, target_low, current_price,
    upside_pct, next_earnings_date, institutional_ownership_pct,
    insider_net_sentiment, analyst_score, summary
    """
    ticker = ticker.upper().strip()
    stock = yf.Ticker(ticker)

    # --- info dict -----------------------------------------------------------
    try:
        info = stock.info or {}
    except Exception:
        info = {}

    num_analysts = _safe_get(info, "numberOfAnalystOpinions")
    rec_mean = _safe_get(info, "recommendationMean")
    rec_key = _safe_get(info, "recommendationKey", "n/a")
    target_mean = _safe_get(info, "targetMeanPrice")
    target_high = _safe_get(info, "targetHighPrice")
    target_low = _safe_get(info, "targetLowPrice")
    current_price = _safe_get(info, "currentPrice") or _safe_get(info, "regularMarketPrice")

    # Upside calculation
    upside_pct = None
    if target_mean is not None and current_price and current_price != 0:
        upside_pct = (target_mean - current_price) / current_price

    # Next earnings date
    next_earnings_date = _unix_to_date_str(_safe_get(info, "earningsTimestamp"))

    # --- institutional holders -----------------------------------------------
    institutional_ownership_pct = None
    try:
        major_holders = stock.major_holders
        institutional_ownership_pct = _get_institutional_pct(major_holders)
    except Exception:
        pass

    # Fallback: use institutionsPercentHeld from info
    if institutional_ownership_pct is None:
        raw = _safe_get(info, "institutionsPercentHeld")
        if raw is not None:
            try:
                institutional_ownership_pct = float(raw)
            except Exception:
                pass

    # --- insider transactions ------------------------------------------------
    insider_net_sentiment = "neutral"
    try:
        insider_tx = stock.insider_transactions
        insider_net_sentiment = _get_insider_net_sentiment(insider_tx)
    except Exception:
        pass

    # --- scoring -------------------------------------------------------------
    s_rating = _score_analyst_rating(rec_mean)
    s_analysts = _score_num_analysts(num_analysts)
    s_upside = _score_upside(upside_pct)
    s_institutional = _score_institutional(institutional_ownership_pct)
    s_insider = _score_insider(insider_net_sentiment)

    analyst_score = s_rating + s_analysts + s_upside + s_institutional + s_insider

    # --- summary string (plain ASCII, no emoji, no box-drawing) --------------
    sep_heavy = "=" * 40
    sep_light = "-" * 40

    def fmt_price(p):
        if p is None:
            return "N/A"
        return f"${p:,.2f}"

    def fmt_pct(p):
        if p is None:
            return "N/A"
        sign = "+" if p >= 0 else ""
        return f"{sign}{p * 100:.1f}%"

    def fmt_inst(p):
        if p is None:
            return "N/A"
        return f"{p * 100:.1f}%"

    rec_display = rec_key.upper().replace("_", " ") if rec_key else "N/A"
    mean_display = f"{rec_mean:.1f}" if rec_mean is not None else "N/A"
    num_display = f"{int(num_analysts)} analysts" if num_analysts is not None else "N/A"

    insider_display = insider_net_sentiment.upper()

    lines = [
        f"Analyst & Market Data -- {ticker}",
        sep_heavy,
        f"  Analyst Coverage  : {num_display}",
        f"  Consensus         : {rec_display} (mean: {mean_display} / 5.0)",
        f"  Price Target      : {fmt_price(target_mean)} mean  "
        f"({fmt_price(target_low)} low / {fmt_price(target_high)} high)",
        f"  Upside to Target  : {fmt_pct(upside_pct)}",
        f"  Next Earnings     : {next_earnings_date or 'N/A'}",
        f"  Institutional Own.: {fmt_inst(institutional_ownership_pct)}",
        f"  Insider Activity  : {insider_display}",
        "",
        f"  Analyst Score     : {analyst_score} / 10",
        "",
        f"  Signal: {_signal_label(analyst_score)}",
        "",
        "This is not financial advice. Always do your own research.",
    ]

    summary = "\n".join(lines)

    return {
        "ticker": ticker,
        "num_analysts": num_analysts,
        "recommendation_key": rec_key,
        "recommendation_mean": rec_mean,
        "target_mean": target_mean,
        "target_high": target_high,
        "target_low": target_low,
        "current_price": current_price,
        "upside_pct": upside_pct,
        "next_earnings_date": next_earnings_date,
        "institutional_ownership_pct": institutional_ownership_pct,
        "insider_net_sentiment": insider_net_sentiment,
        "analyst_score": analyst_score,
        "summary": summary,
    }


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = analyze(symbol)
    print(result["summary"])
