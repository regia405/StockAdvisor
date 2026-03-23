"""
fundamental_analysis.py
-----------------------
Fetches real financial data for a stock ticker using yfinance and produces:
  - Current price
  - P/E ratio (price-to-earnings)
  - Revenue growth (year-over-year)
  - Earnings per share (EPS)
  - Debt-to-equity ratio
  - A composite fundamental score out of 10

Usage:
    from skills.fundamental_analysis import analyze
    result = analyze("AAPL")
    print(result["summary"])
"""

import yfinance as yf


# ---------------------------------------------------------------------------
# ETF analysis
# ---------------------------------------------------------------------------

def _analyze_etf(ticker: str, stock, info: dict) -> dict:
    """
    Fundamental analysis for ETFs. Uses expense ratio, AUM, historical
    returns, yield, and beta instead of earnings-based metrics.

    Scoring (0-10, five metrics x 2 pts):
      Expense ratio : <0.10% → 2, <0.50% → 1, else 0
      Total assets  : >$10B  → 2, >$1B   → 1, else 0
      3-yr ann return: >12%  → 2, >6%    → 1, else 0
      Yield         : >2.5%  → 2, >0.5%  → 1, else 0
      Beta          : <0.8   → 2, <1.2   → 1, else 0
    """
    current_price = info.get("navPrice") or info.get("currentPrice") or info.get("regularMarketPrice")

    expense_ratio  = info.get("annualReportExpenseRatio") or info.get("expenseRatio")
    total_assets   = info.get("totalAssets")
    three_yr       = info.get("threeYearAverageReturn")
    five_yr        = info.get("fiveYearAverageReturn")
    ytd            = info.get("ytdReturn")
    etf_yield      = info.get("yield") or info.get("dividendYield")
    beta           = info.get("beta3Year") or info.get("beta")
    category       = info.get("category", "")
    fund_family    = info.get("fundFamily", "")

    # --- Scores ---
    def score_expense(er):
        if er is None: return 0
        if er < 0.001:  return 2   # <0.10%
        if er < 0.005:  return 1   # <0.50%
        return 0

    def score_assets(ta):
        if ta is None: return 0
        if ta > 10e9:  return 2
        if ta > 1e9:   return 1
        return 0

    def score_3yr(r):
        if r is None: return 0
        if r > 0.12:   return 2   # >12% annualised
        if r > 0.06:   return 1   # >6%
        return 0

    def score_yield(y):
        if y is None: return 0
        if y > 0.025:  return 2
        if y > 0.005:  return 1
        return 0

    def score_beta(b):
        if b is None: return 1   # neutral default
        if b < 0.8:    return 2
        if b < 1.2:    return 1
        return 0

    score = (
        score_expense(expense_ratio)
        + score_assets(total_assets)
        + score_3yr(three_yr)
        + score_yield(etf_yield)
        + score_beta(beta)
    )

    def fmt(val, prefix="", suffix="", decimals=2, pct=False):
        if val is None: return "N/A"
        if pct: return f"{prefix}{val * 100:.{decimals}f}%{suffix}"
        return f"{prefix}{val:,.{decimals}f}{suffix}"

    def fmt_assets(val):
        if val is None: return "N/A"
        if val >= 1e12: return f"${val/1e12:.1f}T"
        if val >= 1e9:  return f"${val/1e9:.1f}B"
        return f"${val/1e6:.0f}M"

    label = _etf_score_label(score)

    summary_lines = [
        f"Fundamental Analysis (ETF) -- {ticker}",
        "=" * 40,
        f"  Fund Family      : {fund_family or 'N/A'}",
        f"  Category         : {category or 'N/A'}",
        f"  Current Price    : {fmt(current_price, prefix='$')}",
        f"  Expense Ratio    : {fmt(expense_ratio, pct=True, decimals=3)}",
        f"  Total Assets     : {fmt_assets(total_assets)}",
        f"  YTD Return       : {fmt(ytd, pct=True)}",
        f"  3-Yr Ann. Return : {fmt(three_yr, pct=True)}",
        f"  5-Yr Ann. Return : {fmt(five_yr, pct=True)}",
        f"  Dividend Yield   : {fmt(etf_yield, pct=True)}",
        f"  Beta             : {fmt(beta)}",
        "",
        f"  Fundamental Score: {score} / 10",
        "",
        label,
        "",
        "This is not financial advice. Always do your own research.",
    ]

    return {
        "ticker":            ticker,
        "instrument_type":   "etf",
        "current_price":     current_price,
        "expense_ratio":     expense_ratio,
        "total_assets":      total_assets,
        "ytd_return":        ytd,
        "three_yr_return":   three_yr,
        "five_yr_return":    five_yr,
        "etf_yield":         etf_yield,
        "beta":              beta,
        "category":          category,
        "fund_family":       fund_family,
        "fundamental_score": score,
        "summary":           "\n".join(summary_lines),
    }


def _etf_score_label(score: int) -> str:
    if score >= 8: return "  Signal: EXCELLENT ETF. Low cost, large AUM, strong track record."
    if score >= 6: return "  Signal: GOOD ETF. Solid across most metrics."
    if score >= 4: return "  Signal: AVERAGE ETF. Some concerns — check alternatives."
    if score >= 2: return "  Signal: BELOW-AVERAGE ETF. High fees or weak returns."
    return           "  Signal: POOR ETF. Consider lower-cost alternatives."


# ---------------------------------------------------------------------------
# Scoring thresholds
# Each metric is scored individually on a 0–2 scale, then summed to 0–10.
# ---------------------------------------------------------------------------

def _score_pe_ratio(pe):
    """
    Price-to-Earnings ratio measures how much investors pay per dollar of
    earnings. Lower is generally cheaper/safer; very high can signal
    overvaluation.

    Scoring:
      0–15   → 2 pts  (undervalued or reasonably priced)
      15–25  → 1 pt   (fair value range)
      >25    → 0 pts  (expensive / speculative)
    """
    if pe is None:
        return 0
    if pe <= 15:
        return 2
    if pe <= 25:
        return 1
    return 0


def _score_revenue_growth(growth):
    """
    Year-over-year revenue growth shows whether the business is expanding.

    Scoring:
      >15%   → 2 pts  (strong growth)
      >5%    → 1 pt   (moderate growth)
      ≤5%    → 0 pts  (stagnant or shrinking)
    """
    if growth is None:
        return 0
    if growth > 0.15:
        return 2
    if growth > 0.05:
        return 1
    return 0


def _score_eps(eps):
    """
    Earnings per share — positive EPS means the company is profitable.

    Scoring:
      >5     → 2 pts  (high profitability)
      >0     → 1 pt   (profitable)
      ≤0     → 0 pts  (unprofitable)
    """
    if eps is None:
        return 0
    if eps > 5:
        return 2
    if eps > 0:
        return 1
    return 0


def _score_debt_to_equity(de):
    """
    Debt-to-equity ratio measures financial leverage. Lower means the company
    relies less on borrowed money, which reduces risk.

    Scoring:
      <0.5   → 2 pts  (low debt, strong balance sheet)
      <1.5   → 1 pt   (manageable debt)
      ≥1.5   → 0 pts  (high leverage / elevated risk)
    """
    if de is None:
        return 0
    if de < 0.5:
        return 2
    if de < 1.5:
        return 1
    return 0


def _score_profit_margin(margin):
    """
    Net profit margin shows how much of each revenue dollar becomes profit.

    Scoring:
      >20%   → 2 pts  (very profitable)
      >10%   → 1 pt   (healthy margins)
      ≤10%   → 0 pts  (thin or negative margins)
    """
    if margin is None:
        return 0
    if margin > 0.20:
        return 2
    if margin > 0.10:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Data fetching helpers
# ---------------------------------------------------------------------------

def _get_revenue_growth(financials):
    """
    Calculate year-over-year revenue growth from the income statement.
    yfinance returns columns in descending date order (newest first).
    Returns a float (e.g. 0.12 for 12%) or None if data is unavailable.
    """
    try:
        # 'Total Revenue' row, two most recent annual periods
        revenue = financials.loc["Total Revenue"]
        if len(revenue) < 2:
            return None
        current = revenue.iloc[0]   # most recent year
        prior   = revenue.iloc[1]   # year before
        if prior == 0:
            return None
        return (current - prior) / abs(prior)
    except (KeyError, IndexError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def analyze(ticker: str) -> dict:
    """
    Perform fundamental analysis on a stock ticker.

    Args:
        ticker: Stock symbol string, e.g. "AAPL", "MSFT", "TSLA"

    Returns:
        A dict with keys:
          ticker, current_price, pe_ratio, revenue_growth, eps,
          debt_to_equity, profit_margin, fundamental_score, summary
        All numeric fields are None if data could not be fetched.
    """
    ticker = ticker.upper().strip()

    # --- Fetch data from yfinance -------------------------------------------
    stock = yf.Ticker(ticker)

    try:
        info = stock.info
    except Exception:
        info = {}

    # Validate that the ticker is real — yfinance returns an empty or minimal
    # dict for unknown symbols.
    has_price = (
        info.get("regularMarketPrice") is not None
        or info.get("currentPrice") is not None
        or info.get("navPrice") is not None
    )
    if not info or not has_price:
        return {
            "ticker": ticker,
            "error": f"Could not fetch data for '{ticker}'. "
                     "Please check that the ticker symbol is valid.",
        }

    # --- Route ETFs to their own analysis branch ----------------------------
    if info.get("quoteType") == "ETF":
        return _analyze_etf(ticker, stock, info)

    # --- Extract individual metrics -----------------------------------------

    # Current price: prefer regularMarketPrice, fall back to currentPrice
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")

    # P/E ratio (trailing twelve months)
    pe_ratio = info.get("trailingPE")

    # Earnings per share (trailing twelve months)
    eps = info.get("trailingEps")

    # Debt-to-equity ratio (reported as a percentage by yfinance, e.g. 150
    # means 1.5×; we normalise to a decimal for our scoring functions)
    raw_de = info.get("debtToEquity")
    debt_to_equity = raw_de / 100 if raw_de is not None else None

    # Net profit margin
    profit_margin = info.get("profitMargins")

    # Revenue growth requires the annual income statement
    try:
        financials = stock.financials   # annual income statement
        revenue_growth = _get_revenue_growth(financials)
    except Exception:
        revenue_growth = None

    # --- Calculate composite score ------------------------------------------
    score = (
        _score_pe_ratio(pe_ratio)
        + _score_revenue_growth(revenue_growth)
        + _score_eps(eps)
        + _score_debt_to_equity(debt_to_equity)
        + _score_profit_margin(profit_margin)
    )
    # Score is now 0–10 (5 metrics × max 2 pts each)

    # --- Build a plain-English summary --------------------------------------
    def fmt(val, prefix="", suffix="", decimals=2, pct=False):
        """Format a metric value, returning 'N/A' if None."""
        if val is None:
            return "N/A"
        if pct:
            return f"{prefix}{val * 100:.{decimals}f}%{suffix}"
        return f"{prefix}{val:,.{decimals}f}{suffix}"

    summary_lines = [
        f"Fundamental Analysis — {ticker}",
        "=" * 40,
        f"  Current Price    : {fmt(current_price, prefix='$')}",
        f"  P/E Ratio        : {fmt(pe_ratio, decimals=1)}",
        f"  EPS (TTM)        : {fmt(eps, prefix='$')}",
        f"  Revenue Growth   : {fmt(revenue_growth, pct=True)}",
        f"  Debt / Equity    : {fmt(debt_to_equity, decimals=2)}",
        f"  Profit Margin    : {fmt(profit_margin, pct=True)}",
        "",
        f"  Fundamental Score: {score} / 10",
        "",
        _score_label(score),
        "",
        "This is not financial advice. Always do your own research.",
    ]
    summary = "\n".join(summary_lines)

    return {
        "ticker":            ticker,
        "instrument_type":   "stock",
        "current_price":     current_price,
        "pe_ratio":          pe_ratio,
        "revenue_growth":    revenue_growth,
        "eps":               eps,
        "debt_to_equity":    debt_to_equity,
        "profit_margin":     profit_margin,
        "fundamental_score": score,
        "summary":           summary,
    }


def _score_label(score: int) -> str:
    """Return a plain-English interpretation of the fundamental score."""
    if score >= 8:
        return "  Signal: STRONG fundamentals. The company looks financially healthy."
    if score >= 6:
        return "  Signal: GOOD fundamentals. Solid business with minor concerns."
    if score >= 4:
        return "  Signal: MIXED fundamentals. Some strengths, some red flags."
    if score >= 2:
        return "  Signal: WEAK fundamentals. Significant financial concerns."
    return   "  Signal: POOR fundamentals. High risk — proceed with caution."


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
