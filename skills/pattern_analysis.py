"""
pattern_analysis.py
-------------------
Compares the current market condition of a ticker to every similar moment
in the past 3 years to answer: "When AAPL looked like this before, what
happened next?"

How it works:
  1. Fetch 3 years of daily OHLCV data.
  2. Compute a "fingerprint" for today:
       - RSI zone     : oversold (<35) | neutral | overbought (>65)
       - MA state     : bull (price > MA50 > MA200) | recovery | caution | bear
       - Trend        : up | sideways | down  (based on 10-day momentum)
       - Vol regime   : low | normal | high   (based on 20-day daily std dev)
  3. Scan every 5 trading days in history (after day 200, to allow MA200).
     For each window that matches on RSI zone + MA state, record the 20-day
     and 60-day forward return.
  4. Report average return, win rate, and number of samples.
  5. Detect named patterns: golden cross, death cross, support test, etc.

Usage:
    from skills.pattern_analysis import analyze
    result = analyze("AAPL")
    print(result["summary"])
"""

import yfinance as yf
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------

def _rsi_zone(rsi):
    if rsi is None:
        return "neutral"
    if rsi < 35:
        return "oversold"
    if rsi > 65:
        return "overbought"
    return "neutral"


def _ma_state(price, ma50, ma200):
    """Four-way classification of price vs moving averages."""
    if price is None or ma50 is None or ma200 is None:
        return "unknown"
    if price > ma50 and ma50 > ma200:
        return "bull"
    if price > ma50 and ma50 <= ma200:
        return "recovery"
    if price <= ma50 and price > ma200:
        return "caution"
    return "bear"


def _trend(closes, idx, window=10):
    """Up/sideways/down based on momentum vs window-period-ago close."""
    if idx < window:
        return "sideways"
    pct = (closes.iloc[idx] - closes.iloc[idx - window]) / closes.iloc[idx - window]
    if pct > 0.03:
        return "up"
    if pct < -0.03:
        return "down"
    return "sideways"


def _vol_regime(returns, idx, window=20):
    """Low/normal/high based on rolling 20-day daily return std dev."""
    if idx < window:
        return "normal"
    vol = returns.iloc[idx - window: idx].std()
    if vol < 0.012:
        return "low"
    if vol > 0.025:
        return "high"
    return "normal"


# ---------------------------------------------------------------------------
# RSI calculation (same as technical_analysis.py)
# ---------------------------------------------------------------------------

def _calc_rsi_series(prices, period=14):
    """Return a Series of RSI values aligned with prices."""
    deltas = prices.diff()
    gain = deltas.clip(lower=0)
    loss = -deltas.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# ---------------------------------------------------------------------------
# Named pattern detection
# ---------------------------------------------------------------------------

def _detect_named_patterns(close, ma50_s, ma200_s, rsi_s):
    """
    Detect well-known technical patterns in the most recent data.
    Returns a list of (name, description) tuples.
    """
    patterns = []

    if len(close) < 25:
        return patterns

    c_now = close.iloc[-1]
    ma50_now = ma50_s.iloc[-1]
    ma200_now = ma200_s.iloc[-1]
    rsi_now = rsi_s.iloc[-1] if not pd.isna(rsi_s.iloc[-1]) else None

    # Golden cross: MA50 crossed above MA200 within last 20 days
    for i in range(2, min(21, len(close))):
        if (ma50_s.iloc[-i] < ma200_s.iloc[-i]
                and ma50_now > ma200_now):
            patterns.append((
                "Golden Cross",
                f"MA50 crossed above MA200 roughly {i} sessions ago — "
                "historically a medium-term bullish signal.",
            ))
            break

    # Death cross: MA50 crossed below MA200 within last 20 days
    for i in range(2, min(21, len(close))):
        if (ma50_s.iloc[-i] > ma200_s.iloc[-i]
                and ma50_now < ma200_now):
            patterns.append((
                "Death Cross",
                f"MA50 crossed below MA200 roughly {i} sessions ago — "
                "historically a medium-term bearish signal.",
            ))
            break

    # Support test: price within 3% of MA200
    if abs(c_now - ma200_now) / ma200_now < 0.03:
        direction = "above" if c_now >= ma200_now else "below"
        patterns.append((
            "200-Day MA Support Test",
            f"Price is within 3% of the 200-day MA (${ma200_now:,.2f}), "
            f"currently {direction} it. Key support/resistance zone.",
        ))

    # Oversold bounce setup: RSI < 35 and price near or above MA200
    if rsi_now and rsi_now < 35:
        if c_now >= ma200_now * 0.98:
            patterns.append((
                "Oversold Bounce Setup",
                f"RSI is {rsi_now:.1f} (oversold) and price is near long-term "
                "support. Historically a higher-probability bounce zone.",
            ))
        else:
            patterns.append((
                "Oversold — Below Key Support",
                f"RSI is {rsi_now:.1f} (oversold) but price is below the 200-day "
                "MA. Oversold can continue in a downtrend — wait for confirmation.",
            ))

    # Overbought / extended: RSI > 70
    if rsi_now and rsi_now > 70:
        patterns.append((
            "Overbought / Extended",
            f"RSI is {rsi_now:.1f} — territory where short-term pullbacks are common. "
            "Strong trends can stay overbought; watch for momentum divergence.",
        ))

    # 52-week high proximity
    high_52 = close.iloc[-252:].max() if len(close) >= 252 else close.max()
    low_52 = close.iloc[-252:].min() if len(close) >= 252 else close.min()
    if abs(c_now - high_52) / high_52 < 0.03:
        patterns.append((
            "Near 52-Week High",
            f"Price (${c_now:,.2f}) is within 3% of the 52-week high "
            f"(${high_52:,.2f}). Breakout potential, but also natural resistance.",
        ))
    elif abs(c_now - low_52) / low_52 < 0.05:
        patterns.append((
            "Near 52-Week Low",
            f"Price (${c_now:,.2f}) is within 5% of the 52-week low "
            f"(${low_52:,.2f}). High risk area — needs strong catalyst to reverse.",
        ))

    return patterns


# ---------------------------------------------------------------------------
# Historical fingerprint matching
# ---------------------------------------------------------------------------

def _scan_history(close, ma50_s, ma200_s, rsi_s, returns, current_fp):
    """
    Scan all historical windows for fingerprint matches.
    Returns list of (20d_return, 60d_return) pairs for matching windows.
    """
    matches_20d = []
    matches_60d = []

    n = len(close)
    # Start from day 210 so MA200 is meaningful; stop 60 days before end
    # so we have room to compute forward returns.
    start = 210
    end = n - 20   # need at least 20d forward

    for i in range(start, end, 5):  # sample every 5 days
        rsi_val = rsi_s.iloc[i]
        if pd.isna(rsi_val):
            continue
        ma50_val = ma50_s.iloc[i]
        ma200_val = ma200_s.iloc[i]
        if pd.isna(ma50_val) or pd.isna(ma200_val):
            continue

        fp = {
            "rsi_zone": _rsi_zone(rsi_val),
            "ma_state": _ma_state(close.iloc[i], ma50_val, ma200_val),
        }

        # Match on both RSI zone and MA state
        if (fp["rsi_zone"] == current_fp["rsi_zone"]
                and fp["ma_state"] == current_fp["ma_state"]):
            fwd_20 = (close.iloc[i + 20] - close.iloc[i]) / close.iloc[i]
            matches_20d.append(fwd_20)

            if i + 60 < n:
                fwd_60 = (close.iloc[i + 60] - close.iloc[i]) / close.iloc[i]
                matches_60d.append(fwd_60)

    return matches_20d, matches_60d


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def analyze(ticker: str) -> dict:
    """
    Run historical pattern analysis for a ticker.

    Returns a dict with:
        ticker, current_fingerprint, named_patterns,
        similar_setups_count, avg_return_20d, win_rate_20d,
        avg_return_60d, win_rate_60d, summary
    """
    ticker = ticker.upper().strip()

    try:
        hist = yf.Ticker(ticker).history(period="3y")
        if hist.empty:
            return {"ticker": ticker, "error": f"No price history for '{ticker}'."}
    except Exception:
        return {"ticker": ticker, "error": f"Could not fetch history for '{ticker}'."}

    close = hist["Close"]
    n = len(close)

    if n < 220:
        return {
            "ticker": ticker,
            "error": "Not enough history for pattern analysis (need 220+ trading days).",
        }

    # --- Compute indicator series ---
    ma50_s  = close.rolling(50).mean()
    ma200_s = close.rolling(200).mean()
    rsi_s   = _calc_rsi_series(close)
    returns = close.pct_change()

    # --- Current fingerprint ---
    rsi_now  = rsi_s.iloc[-1] if not pd.isna(rsi_s.iloc[-1]) else None
    ma50_now = ma50_s.iloc[-1]
    ma200_now = ma200_s.iloc[-1]
    price_now = close.iloc[-1]

    current_fp = {
        "rsi_zone": _rsi_zone(rsi_now),
        "ma_state": _ma_state(price_now, ma50_now, ma200_now),
        "trend":    _trend(close, n - 1),
        "vol_regime": _vol_regime(returns, n - 1),
    }

    # --- Named pattern detection ---
    named_patterns = _detect_named_patterns(close, ma50_s, ma200_s, rsi_s)

    # --- Historical match scanning ---
    matches_20d, matches_60d = _scan_history(
        close, ma50_s, ma200_s, rsi_s, returns, current_fp
    )

    # --- Summarise match stats ---
    count = len(matches_20d)
    if count > 0:
        avg_20d = sum(matches_20d) / count
        win_20d = sum(1 for r in matches_20d if r > 0) / count
    else:
        avg_20d = win_20d = None

    if matches_60d:
        avg_60d = sum(matches_60d) / len(matches_60d)
        win_60d = sum(1 for r in matches_60d if r > 0) / len(matches_60d)
    else:
        avg_60d = win_60d = None

    # --- Build summary ---
    def pct(v):
        return f"{v * 100:+.1f}%" if v is not None else "N/A"

    def wr(v):
        return f"{v * 100:.0f}%" if v is not None else "N/A"

    fp = current_fp
    summary_lines = [
        f"Historical Pattern Analysis -- {ticker}",
        "=" * 40,
        f"  Current Fingerprint:",
        f"    RSI Zone     : {fp['rsi_zone'].upper()}",
        f"    MA State     : {fp['ma_state'].upper()}  "
        f"(price {'above' if fp['ma_state'] in ('bull','recovery') else 'below'} MA50)",
        f"    Trend (10d)  : {fp['trend'].upper()}",
        f"    Volatility   : {fp['vol_regime'].upper()}",
        "",
    ]

    if named_patterns:
        summary_lines.append("  Detected Patterns:")
        for name, desc in named_patterns:
            summary_lines.append(f"    * {name}")
            summary_lines.append(f"      {desc}")
        summary_lines.append("")

    summary_lines.append(f"  Historical Matches  : {count} similar past setups found")
    if count > 0:
        summary_lines += [
            f"  -- 20-Day Outlook --",
            f"    Avg Return   : {pct(avg_20d)}",
            f"    Win Rate     : {wr(win_20d)}",
        ]
        if avg_60d is not None:
            summary_lines += [
                f"  -- 60-Day Outlook --",
                f"    Avg Return   : {pct(avg_60d)}",
                f"    Win Rate     : {wr(win_60d)}",
            ]
    else:
        summary_lines.append(
            "  (No historical matches found for this exact fingerprint.)"
        )

    summary_lines += ["", "This is not financial advice. Always do your own research."]
    summary = "\n".join(summary_lines)

    return {
        "ticker":              ticker,
        "current_fingerprint": current_fp,
        "named_patterns":      named_patterns,
        "similar_setups_count": count,
        "avg_return_20d":      avg_20d,
        "win_rate_20d":        win_20d,
        "avg_return_60d":      avg_60d,
        "win_rate_60d":        win_60d,
        "summary":             summary,
    }


if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    r = analyze(symbol)
    print(r.get("error") or r["summary"])
