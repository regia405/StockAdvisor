"""
technical_analysis.py
---------------------
Fetches historical price data for a stock ticker using yfinance and produces:
  - Relative Strength Index (RSI)
  - MACD (Moving Average Convergence Divergence)
  - Moving averages (50-day, 200-day)
  - Chart pattern detection (support/resistance trends)
  - A composite technical score out of 10

Usage:
    from skills.technical_analysis import analyze
    result = analyze("AAPL")
    print(result["summary"])
"""

import os
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ---------------------------------------------------------------------------
# Scoring thresholds
# Each metric is scored individually on a 0–2 scale, then summed to 0–10.
# ---------------------------------------------------------------------------

def _score_rsi(rsi):
    """
    Relative Strength Index (14-period) measures momentum and overbought/oversold.
    
    Scoring:
      <30      → 2 pts  (oversold, potential bounce)
      30–70    → 1 pt   (neutral momentum)
      >70      → 0 pts  (overbought, potential pullback)
    """
    if rsi is None:
        return 1  # neutral default
    if rsi < 30:
        return 2
    if rsi <= 70:
        return 1
    return 0


def _score_macd(macd, signal):
    """
    MACD (Moving Average Convergence Divergence) compares two exponential 
    moving averages. Positive histogram and MACD above signal line = bullish.
    
    Scoring:
      MACD > Signal & Histogram > 0  → 2 pts  (bullish momentum)
      MACD ≈ Signal                  → 1 pt   (neutral)
      MACD < Signal & Histogram < 0  → 0 pts  (bearish momentum)
    """
    if macd is None or signal is None:
        return 1
    
    histogram = macd - signal
    
    if histogram > 0:
        return 2
    if histogram < -0.01:  # small threshold to avoid false neutrals
        return 0
    return 1


def _score_price_vs_ma(price, ma50, ma200):
    """
    Price position relative to moving averages.
    
    Scoring:
      Price > MA50 > MA200  → 2 pts  (strong uptrend)
      Price > MA50, near MA200 → 1 pt (mild uptrend)
      Price < MA50 or Price < MA200 → 0 pts (downtrend)
    """
    if price is None or ma50 is None or ma200 is None:
        return 1
    
    if price > ma50 and ma50 > ma200:
        return 2
    if price > ma50:
        return 1
    return 0


def _score_trend_strength(recent_closes):
    """
    Measure trend strength by comparing recent prices to their average.
    Strong uptrend: most recent closes > 10-period average
    Strong downtrend: most recent closes < 10-period average
    
    Scoring:
      Clear uptrend     → 2 pts
      Weak trend        → 1 pt
      Clear downtrend   → 0 pts
    """
    if recent_closes is None or len(recent_closes) < 10:
        return 1
    
    try:
        avg = recent_closes[-10:].mean()
        recent = recent_closes.iloc[-1]
        
        # Check last 3 days for trend confirmation
        last_3_avg = recent_closes[-3:].mean()
        
        if last_3_avg > avg and recent > avg:
            return 2
        if last_3_avg < avg and recent < avg:
            return 0
        return 1
    except Exception:
        return 1


def _score_volatility(returns):
    """
    Lower volatility (calmer price action) is often better for technical trading.
    
    Scoring:
      <2% daily volatility  → 2 pts  (stable)
      2–3% daily volatility → 1 pt   (moderate)
      >3% daily volatility  → 0 pts  (very volatile)
    """
    if returns is None or len(returns) < 2:
        return 1
    
    try:
        vol = returns.std()
        if vol <= 0.02:
            return 2
        if vol <= 0.03:
            return 1
        return 0
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# Technical indicator calculations
# ---------------------------------------------------------------------------

def _calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index (RSI)."""
    try:
        if len(prices) < period:
            return None
        
        deltas = prices.diff()
        seed = deltas[:period + 1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        rs = up / down if down != 0 else 0
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        # Continue RSI calculation for remaining periods
        for i in range(period + 1, len(prices)):
            delta = deltas.iloc[i]
            if delta > 0:
                up = (up * (period - 1) + delta) / period
            else:
                down = (down * (period - 1) + (-delta)) / period
            
            rs = up / down if down != 0 else 0
            rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    except Exception:
        return None


def _calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD and signal line."""
    try:
        if len(prices) < slow:
            return None, None
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        
        return macd.iloc[-1], signal_line.iloc[-1]
    except Exception:
        return None, None


def _calculate_moving_averages(prices):
    """Calculate 50-day and 200-day moving averages."""
    try:
        ma50 = prices.rolling(window=50).mean()
        ma200 = prices.rolling(window=200).mean()
        
        return ma50.iloc[-1], ma200.iloc[-1]
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def _generate_chart(ticker: str, hist: pd.DataFrame, ma50, ma200, rsi, output_dir="data") -> str:
    """
    Generate a 3-panel chart: price + MAs, RSI, and volume.
    Saves to {output_dir}/{ticker}_chart.png and returns the file path.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{ticker}_chart.png")

        close = hist["Close"]
        volume = hist["Volume"]
        dates = hist.index

        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(12, 9),
            gridspec_kw={"height_ratios": [3, 1.5, 1]},
            sharex=True,
        )
        fig.patch.set_facecolor("#0f0f0f")
        for ax in (ax1, ax2, ax3):
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="#cccccc")
            ax.yaxis.label.set_color("#cccccc")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")

        # --- Panel 1: Price + MAs ---
        ax1.plot(dates, close, color="#4fc3f7", linewidth=1.5, label="Price")
        ma50_series = close.rolling(50).mean()
        ma200_series = close.rolling(200).mean()
        ax1.plot(dates, ma50_series, color="#ffb300", linewidth=1.2, linestyle="--", label="MA50")
        ax1.plot(dates, ma200_series, color="#ef5350", linewidth=1.2, linestyle="--", label="MA200")
        ax1.set_ylabel("Price", color="#cccccc")
        ax1.set_title(f"{ticker} — Technical Analysis Chart (1 Year)", color="#ffffff", fontsize=13, pad=10)
        ax1.legend(loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc", framealpha=0.8)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        # --- Panel 2: RSI ---
        deltas = close.diff()
        gain = deltas.clip(lower=0)
        loss = -deltas.clip(upper=0)
        avg_gain = gain.ewm(com=13, min_periods=14).mean()
        avg_loss = loss.ewm(com=13, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        rsi_series = 100.0 - (100.0 / (1.0 + rs))

        ax2.plot(dates, rsi_series, color="#ce93d8", linewidth=1.2, label="RSI (14)")
        ax2.axhline(70, color="#ef5350", linewidth=0.8, linestyle="--", alpha=0.5)
        ax2.axhline(30, color="#26a69a", linewidth=0.8, linestyle="--", alpha=0.5)
        ax2.fill_between(dates, 70, rsi_series.clip(lower=70), color="#ef5350", alpha=0.15)
        ax2.fill_between(dates, rsi_series.clip(upper=30), 30, color="#26a69a", alpha=0.15)
        ax2.set_ylabel("RSI", color="#cccccc")
        ax2.set_ylim(0, 100)

        # --- Panel 3: Volume ---
        bar_colors = ["#26a69a" if c >= o else "#ef5350"
                      for c, o in zip(hist["Close"], hist["Open"])]
        ax3.bar(dates, volume, color=bar_colors, alpha=0.7, width=1)
        ax3.set_ylabel("Volume", color="#cccccc")
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))

        # --- Shared x-axis formatting (bottom panel only) ---
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right", color="#cccccc")

        plt.tight_layout()
        plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return path
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def analyze(ticker: str, output_dir: str = "data") -> dict:
    """
    Perform technical analysis on a stock ticker.

    Args:
        ticker: Stock symbol string, e.g. "AAPL", "MSFT", "TSLA"

    Returns:
        A dict with keys:
          ticker, current_price, rsi, macd, signal, ma50, ma200,
          technical_score, summary
        All numeric fields are None if data could not be fetched.
    """
    ticker = ticker.upper().strip()

    # --- Fetch historical data from yfinance --------------------------------
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")  # 1 year of data
        
        if hist.empty:
            return {
                "ticker": ticker,
                "error": f"Could not fetch price history for '{ticker}'. "
                         "Please check that the ticker symbol is valid.",
            }
    except Exception:
        return {
            "ticker": ticker,
            "error": f"Could not fetch price history for '{ticker}'. "
                     "Please check that the ticker symbol is valid.",
        }

    # --- Extract prices and calculate metrics -------------------------------
    close_prices = hist['Close']
    current_price = close_prices.iloc[-1]
    
    # RSI (14-period)
    rsi = _calculate_rsi(close_prices)
    
    # MACD
    macd, signal = _calculate_macd(close_prices)
    
    # Moving averages
    ma50, ma200 = _calculate_moving_averages(close_prices)
    
    # Trend strength (last 10 closes)
    trend_score_input = close_prices
    
    # Daily returns for volatility
    returns = close_prices.pct_change()

    # --- Calculate composite score ------------------------------------------
    score = (
        _score_rsi(rsi)
        + _score_macd(macd, signal)
        + _score_price_vs_ma(current_price, ma50, ma200)
        + _score_trend_strength(close_prices)
        + _score_volatility(returns)
    )
    # Score is now 0–10 (5 metrics × max 2 pts each)

    # --- Generate chart -----------------------------------------------------
    chart_path = _generate_chart(ticker, hist, ma50, ma200, rsi, output_dir=output_dir)

    # --- Build a plain-English summary --------------------------------------
    def fmt(val, prefix="", suffix="", decimals=2, pct=False):
        """Format a metric value, returning 'N/A' if None."""
        if val is None:
            return "N/A"
        if pct:
            return f"{prefix}{val * 100:.{decimals}f}%{suffix}"
        return f"{prefix}{val:,.{decimals}f}{suffix}"

    summary_lines = [
        f"Technical Analysis — {ticker}",
        "=" * 40,
        f"  Current Price    : {fmt(current_price, prefix='$')}",
        f"  RSI (14)         : {fmt(rsi, decimals=1)}",
        f"  MACD             : {fmt(macd, decimals=3)}",
        f"  Signal Line      : {fmt(signal, decimals=3)}",
        f"  MA50             : {fmt(ma50, prefix='$', decimals=2)}",
        f"  MA200            : {fmt(ma200, prefix='$', decimals=2)}",
        f"  Volatility       : {fmt(returns.std(), pct=True)}",
        "",
        f"  Technical Score  : {score} / 10",
        "",
        _score_label(score),
        "",
        f"  Chart saved to   : {chart_path}" if chart_path else "  Chart            : could not generate",
        "",
        "This is not financial advice. Always do your own research.",
    ]
    summary = "\n".join(summary_lines)

    return {
        "ticker":           ticker,
        "current_price":    current_price,
        "rsi":              rsi,
        "macd":             macd,
        "signal":           signal,
        "ma50":             ma50,
        "ma200":            ma200,
        "volatility":       returns.std(),
        "technical_score":  score,
        "chart_path":       chart_path,
        "summary":          summary,
    }


def _score_label(score: int) -> str:
    """Return a plain-English interpretation of the technical score."""
    if score >= 8:
        return "  Signal: STRONG technical setup. Bullish momentum and uptrend."
    if score >= 6:
        return "  Signal: GOOD technical setup. Positive indicators align."
    if score >= 4:
        return "  Signal: MIXED technical picture. Conflicting signals."
    if score >= 2:
        return "  Signal: WEAK technical setup. Bearish momentum."
    return   "  Signal: POOR technical setup. Strong bearish signals."


# ---------------------------------------------------------------------------
# Quick self-test — run this file directly to try it out
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = analyze(symbol)
    if "error" in result:
        print(result["error"])
    else:
        print(result["summary"])
