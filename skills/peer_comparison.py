"""
peer_comparison.py -- Compare a ticker against its sector peers on key metrics.

Metrics compared:
  - P/E Ratio (trailingPE)
  - Revenue Growth (revenueGrowth)
  - Profit Margin (profitMargins)
  - 52-Week Return (52WeekChange)
  - EPS (trailingEps)

Peer detection:
  1. Use PEER_MAP for well-known tickers.
  2. Fall back to SECTOR_FALLBACK using the ticker's reported sector.
  3. If neither works, returns an error.
"""

import yfinance as yf

# ---------------------------------------------------------------------------
# Peer maps
# ---------------------------------------------------------------------------

PEER_MAP = {
    # Tech
    "AAPL":  ["MSFT", "GOOGL", "META", "AMZN", "NVDA"],
    "MSFT":  ["AAPL", "GOOGL", "AMZN", "META", "NVDA"],
    "GOOGL": ["META", "MSFT", "AAPL", "AMZN", "SNAP"],
    "META":  ["GOOGL", "SNAP", "PINS", "TWTR", "MSFT"],
    "NVDA":  ["AMD", "INTC", "QCOM", "AVGO", "TSM"],
    "AMD":   ["NVDA", "INTC", "QCOM", "AVGO", "MU"],
    "TSLA":  ["F", "GM", "RIVN", "NIO", "LCID"],
    # Finance
    "JPM":   ["BAC", "WFC", "GS", "MS", "C"],
    "BAC":   ["JPM", "WFC", "C", "GS", "USB"],
    "GS":    ["MS", "JPM", "BAC", "C", "BLK"],
    # Healthcare
    "JNJ":   ["PFE", "MRK", "ABBV", "BMY", "LLY"],
    "PFE":   ["JNJ", "MRK", "ABBV", "BMY", "AZN"],
    # Retail/Consumer
    "AMZN":  ["WMT", "TGT", "COST", "EBAY", "SHOP"],
    "WMT":   ["AMZN", "TGT", "COST", "KR", "DG"],
    # ETFs
    "SPY":   ["IVV", "VOO", "QQQ", "DIA", "VTI"],
    "QQQ":   ["SPY", "IVV", "VOO", "XLK", "VGT"],
}

SECTOR_FALLBACK = {
    "Technology":             ["AAPL", "MSFT", "GOOGL", "META", "NVDA"],
    "Financial Services":     ["JPM", "BAC", "WFC", "GS", "MS"],
    "Healthcare":             ["JNJ", "PFE", "MRK", "ABBV", "LLY"],
    "Consumer Cyclical":      ["AMZN", "TSLA", "HD", "NKE", "MCD"],
    "Consumer Defensive":     ["WMT", "PG", "KO", "PEP", "COST"],
    "Energy":                 ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Industrials":            ["CAT", "BA", "HON", "GE", "UPS"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_pct(value):
    """Format a fractional value as a percentage string, or 'N/A'."""
    if value is None:
        return "N/A"
    return "{:.1f}%".format(value * 100)


def _fmt_float(value, decimals=1):
    """Format a float to a fixed number of decimal places, or 'N/A'."""
    if value is None:
        return "N/A"
    return "{:.{}f}".format(value, decimals)


def _fetch_metrics(ticker_symbol):
    """
    Fetch key metrics for a single ticker from yfinance.

    Returns a dict with keys:
      ticker, pe_ratio, revenue_growth, profit_margin, week52_return, eps
    All numeric values may be None if not available.
    """
    try:
        info = yf.Ticker(ticker_symbol).info
    except Exception:
        info = {}

    pe_ratio       = info.get("trailingPE")
    revenue_growth = info.get("revenueGrowth")
    profit_margin  = info.get("profitMargins")
    week52_return  = info.get("52WeekChange")
    eps            = info.get("trailingEps")

    # Coerce non-finite floats to None
    import math
    def _clean(v):
        try:
            if v is None:
                return None
            f = float(v)
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    return {
        "ticker":         ticker_symbol.upper(),
        "pe_ratio":       _clean(pe_ratio),
        "revenue_growth": _clean(revenue_growth),
        "profit_margin":  _clean(profit_margin),
        "week52_return":  _clean(week52_return),
        "eps":            _clean(eps),
    }


def _rank_metric(value, all_values, higher_is_better=True):
    """
    Rank `value` among `all_values` (which includes `value`).
    Returns 1-based rank (1 = best). None values rank last.

    For P/E ratio, lower is better (higher_is_better=False).
    """
    valid = [v for v in all_values if v is not None]
    if value is None or not valid:
        return len(all_values)  # worst rank

    if higher_is_better:
        sorted_vals = sorted(valid, reverse=True)
    else:
        sorted_vals = sorted(valid)

    try:
        return sorted_vals.index(value) + 1
    except ValueError:
        return len(all_values)


def _compute_peer_score(ranks, n_total):
    """
    Compute a peer score 0-10 from a list of 1-based ranks.

    Uses: score = max(0, 10 - int(avg_rank * 2))
    n_total is the total number of tickers ranked (target + peers).
    """
    valid_ranks = [r for r in ranks if r is not None]
    if not valid_ranks:
        return 0
    avg_rank = sum(valid_ranks) / len(valid_ranks)
    score = max(0, 10 - int(avg_rank * 2))
    return min(score, 10)


def _signal_label(peer_score):
    if peer_score >= 8:
        return "LEADER. Ranks near the top of its peer group."
    if peer_score >= 6:
        return "ABOVE AVERAGE vs peers. Stronger than most competitors."
    if peer_score >= 4:
        return "AVERAGE vs peers. Some strengths, some weaknesses."
    if peer_score >= 2:
        return "BELOW AVERAGE vs peers. Underperforming most metrics."
    return "LAGGARD. Ranks near the bottom of its peer group."


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze(ticker):
    """
    Compare `ticker` against its sector peers.

    Returns a dict with keys:
      ticker, peers_analyzed, peer_data, ticker_rank_pe,
      ticker_rank_growth, ticker_rank_margin, ticker_rank_return,
      ticker_rank_eps, peer_score, summary, error (only on failure)
    """
    ticker = ticker.upper().strip()

    # --- Determine peers ---------------------------------------------------
    peers = None
    sector_used = None

    if ticker in PEER_MAP:
        peers = PEER_MAP[ticker]
    else:
        # Try sector fallback
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector")
        except Exception:
            sector = None

        if sector and sector in SECTOR_FALLBACK:
            sector_used = sector
            candidates = SECTOR_FALLBACK[sector]
            # Exclude the ticker itself from peers
            peers = [p for p in candidates if p != ticker][:5]

    if not peers:
        return {
            "ticker": ticker,
            "error": (
                "Could not determine peers for '{}'. "
                "Not in PEER_MAP and sector not recognised.".format(ticker)
            ),
        }

    # --- Fetch metrics for ticker + all peers ------------------------------
    all_tickers = [ticker] + peers
    peer_data = []
    for sym in all_tickers:
        peer_data.append(_fetch_metrics(sym))

    ticker_metrics = peer_data[0]

    # Collect per-metric value lists (all tickers, including subject)
    pe_vals      = [d["pe_ratio"]       for d in peer_data]
    growth_vals  = [d["revenue_growth"] for d in peer_data]
    margin_vals  = [d["profit_margin"]  for d in peer_data]
    return_vals  = [d["week52_return"]  for d in peer_data]
    eps_vals     = [d["eps"]            for d in peer_data]

    n_total = len(all_tickers)

    # --- Rank the subject ticker on each metric ----------------------------
    # Lower P/E is better; higher is better for all others
    rank_pe     = _rank_metric(ticker_metrics["pe_ratio"],       pe_vals,     higher_is_better=False)
    rank_growth = _rank_metric(ticker_metrics["revenue_growth"], growth_vals, higher_is_better=True)
    rank_margin = _rank_metric(ticker_metrics["profit_margin"],  margin_vals, higher_is_better=True)
    rank_return = _rank_metric(ticker_metrics["week52_return"],  return_vals, higher_is_better=True)
    rank_eps    = _rank_metric(ticker_metrics["eps"],            eps_vals,    higher_is_better=True)

    # --- Peer score --------------------------------------------------------
    ranks = [rank_pe, rank_growth, rank_margin, rank_return, rank_eps]
    peer_score = _compute_peer_score(ranks, n_total)

    # --- Sector averages (peers only, excluding subject) -------------------
    def _avg(vals_list):
        valid = [v for v in vals_list[1:] if v is not None]  # skip index 0 (subject)
        return sum(valid) / len(valid) if valid else None

    avg_pe      = _avg(pe_vals)
    avg_growth  = _avg(growth_vals)
    avg_margin  = _avg(margin_vals)
    avg_return  = _avg(return_vals)

    # --- Build summary string ----------------------------------------------
    peer_names = ", ".join(peers)
    label = sector_used if sector_used else "Sector"

    col_w  = 16  # metric label column width
    val_w  = 10  # ticker value column width
    avg_w  = 12  # sector avg column width

    header      = "Peer Comparison -- {} vs {}".format(ticker, label)
    sep_double  = "=" * 42
    sep_single  = "-" * 42

    def _rank_str(r):
        return "{}/{}".format(r, n_total)

    rows = [
        ("P/E Ratio",
         _fmt_float(ticker_metrics["pe_ratio"]),
         _fmt_float(avg_pe),
         _rank_str(rank_pe)),
        ("Revenue Growth",
         _fmt_pct(ticker_metrics["revenue_growth"]),
         _fmt_pct(avg_growth),
         _rank_str(rank_growth)),
        ("Profit Margin",
         _fmt_pct(ticker_metrics["profit_margin"]),
         _fmt_pct(avg_margin),
         _rank_str(rank_margin)),
        ("52-Week Return",
         _fmt_pct(ticker_metrics["week52_return"]),
         _fmt_pct(avg_return),
         _rank_str(rank_return)),
    ]

    lines = []
    lines.append(header)
    lines.append(sep_double)
    lines.append("  Peers: {}".format(peer_names))
    lines.append("")
    lines.append("  {:<{cw}}{:<{vw}}{:<{aw}}{}".format(
        "Metric", ticker, "Sector Avg", "Rank",
        cw=col_w, vw=val_w, aw=avg_w))
    lines.append("  " + sep_single)
    for metric_name, t_val, a_val, r_str in rows:
        lines.append("  {:<{cw}}{:<{vw}}{:<{aw}}{}".format(
            metric_name, t_val, a_val, r_str,
            cw=col_w, vw=val_w, aw=avg_w))
    lines.append("")
    lines.append("  Peer Score       : {} / 10".format(peer_score))
    lines.append("")
    lines.append("  Signal: {}".format(_signal_label(peer_score)))
    lines.append("")
    lines.append("This is not financial advice. Always do your own research.")

    summary = "\n".join(lines)

    return {
        "ticker":              ticker,
        "peers_analyzed":      peers,
        "peer_data":           peer_data,
        "ticker_rank_pe":      rank_pe,
        "ticker_rank_growth":  rank_growth,
        "ticker_rank_margin":  rank_margin,
        "ticker_rank_return":  rank_return,
        "ticker_rank_eps":     rank_eps,
        "peer_score":          peer_score,
        "summary":             summary,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    test_tickers = ["AAPL", "JPM"]
    if len(sys.argv) > 1:
        test_tickers = sys.argv[1:]

    for t in test_tickers:
        print("")
        result = analyze(t)
        if "error" in result:
            print("ERROR: {}".format(result["error"]))
        else:
            print(result["summary"])
            print("")
            print("Returned keys: {}".format(", ".join(str(k) for k in result.keys())))
        print("")
