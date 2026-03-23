"""
portfolio_manager.py
--------------------
Manages a personal stock portfolio by:
  - Reading holdings from data/portfolio.json
  - Fetching current prices using yfinance
  - Calculating total value, gains/losses, and risk metrics
  - Flagging significant price movements (>5% daily)
  - Optionally running full analysis on each holding

Usage:
    from skills.portfolio_manager import get_portfolio_status
    from skills.portfolio_manager import analyze_portfolio
    
    # Simple status check
    status = get_portfolio_status()
    print(status["summary"])
    
    # Full analysis with signals
    analysis = analyze_portfolio()
    print(analysis["summary"])
"""

import json
import os
from pathlib import Path
import yfinance as yf

# Try relative imports first, fall back to absolute
try:
    from .fundamental_analysis import analyze as fundamental_analyze
    from .technical_analysis import analyze as technical_analyze
except ImportError:
    from skills.fundamental_analysis import analyze as fundamental_analyze
    from skills.technical_analysis import analyze as technical_analyze


# ---------------------------------------------------------------------------
# Portfolio file management
# ---------------------------------------------------------------------------

PORTFOLIO_FILE = "data/portfolio.json"


def _get_portfolio_path():
    """Return absolute path to portfolio.json."""
    return os.path.join(os.path.dirname(__file__), "..", PORTFOLIO_FILE)


def _load_portfolio():
    """Load portfolio from data/portfolio.json. Returns None if file doesn't exist."""
    path = _get_portfolio_path()
    if not os.path.exists(path):
        return None
    
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _save_portfolio(portfolio):
    """Save portfolio to data/portfolio.json."""
    path = _get_portfolio_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(portfolio, f, indent=2)


def _create_empty_portfolio():
    """Create and save an empty portfolio template."""
    template = {
        "holdings": [
            {
                "ticker": "AAPL",
                "shares": 10,
                "buy_price": 150.00,
                "buy_date": "2025-01-15"
            },
            {
                "ticker": "MSFT",
                "shares": 5,
                "buy_price": 350.00,
                "buy_date": "2025-02-20"
            }
        ]
    }
    _save_portfolio(template)
    return template


# ---------------------------------------------------------------------------
# Portfolio calculations
# ---------------------------------------------------------------------------

def _get_current_prices(holdings):
    """Fetch current prices for all holdings using yfinance."""
    prices = {}
    
    for holding in holdings:
        ticker = holding.get("ticker", "").upper()
        if not ticker:
            continue
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            
            if price:
                prices[ticker] = price
        except Exception:
            prices[ticker] = None
    
    return prices


def _calculate_position_metrics(holding, current_price):
    """Calculate gain/loss and return % for a single position."""
    shares = holding.get("shares", 0)
    buy_price = holding.get("buy_price", 0)
    
    cost_basis = shares * buy_price
    current_value = shares * current_price if current_price else 0
    gain_loss = current_value - cost_basis
    return_pct = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0
    
    return {
        "cost_basis": cost_basis,
        "current_value": current_value,
        "gain_loss": gain_loss,
        "return_pct": return_pct,
    }


def _check_daily_movement(ticker, threshold=0.05):
    """
    Check if a stock has moved more than threshold (default 5%) today.
    Returns percentage change or None if data unavailable.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get today's price and previous close
        current = info.get("currentPrice") or info.get("regularMarketPrice")
        previous_close = info.get("previousClose")
        
        if current and previous_close:
            pct_change = ((current - previous_close) / previous_close) * 100
            return pct_change
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def get_portfolio_status():
    """
    Get a quick status of the portfolio without detailed analysis.
    Returns dict with summary and position details.
    """
    portfolio = _load_portfolio()
    
    if not portfolio:
        return {
            "error": "Portfolio not found. Run create_empty_portfolio() to initialize.",
        }
    
    holdings = portfolio.get("holdings", [])
    
    if not holdings:
        return {
            "error": "Portfolio is empty.",
        }
    
    # Fetch current prices
    prices = _get_current_prices(holdings)
    
    # Calculate metrics
    total_cost_basis = 0
    total_current_value = 0
    positions = []
    flagged_positions = []
    
    for holding in holdings:
        ticker = holding.get("ticker", "").upper()
        shares = holding.get("shares", 0)
        current_price = prices.get(ticker)
        
        if not current_price:
            continue
        
        metrics = _calculate_position_metrics(holding, current_price)
        total_cost_basis += metrics["cost_basis"]
        total_current_value += metrics["current_value"]
        
        position_info = {
            "ticker": ticker,
            "shares": shares,
            "buy_price": holding.get("buy_price"),
            "current_price": current_price,
            "current_value": metrics["current_value"],
            "gain_loss": metrics["gain_loss"],
            "return_pct": metrics["return_pct"],
        }
        
        positions.append(position_info)
        
        # Check for significant daily movement
        daily_pct = _check_daily_movement(ticker)
        if daily_pct and abs(daily_pct) > 5:
            flagged_positions.append({
                "ticker": ticker,
                "daily_change": daily_pct,
            })
    
    # Build summary
    total_gain_loss = total_current_value - total_cost_basis
    total_return_pct = (total_gain_loss / total_cost_basis * 100) if total_cost_basis > 0 else 0
    
    summary_lines = [
        "Portfolio Status",
        "=" * 40,
        f"  Total Cost Basis : ${total_cost_basis:,.2f}",
        f"  Total Value      : ${total_current_value:,.2f}",
        f"  Total Gain/Loss  : ${total_gain_loss:,.2f} ({total_return_pct:+.2f}%)",
        "",
        "  Positions:",
    ]
    
    for pos in positions:
        emoji = "📈" if pos["gain_loss"] >= 0 else "📉"
        summary_lines.append(
            f"    {emoji} {pos['ticker']}: {pos['shares']} @ "
            f"${pos['current_price']:.2f} = ${pos['current_value']:,.2f} "
            f"({pos['return_pct']:+.1f}%)"
        )
    
    if flagged_positions:
        summary_lines.append("")
        summary_lines.append("  ⚠️  ALERT: Significant Daily Movements:")
        for flag in flagged_positions:
            emoji = "🔴" if flag["daily_change"] < 0 else "🟢"
            summary_lines.append(
                f"    {emoji} {flag['ticker']}: {flag['daily_change']:+.2f}% today"
            )
    
    summary_lines.append("")
    summary_lines.append("This is not financial advice. Always do your own research.")
    
    summary = "\n".join(summary_lines)
    
    return {
        "total_cost_basis": total_cost_basis,
        "total_current_value": total_current_value,
        "total_gain_loss": total_gain_loss,
        "total_return_pct": total_return_pct,
        "positions": positions,
        "flagged": flagged_positions,
        "summary": summary,
    }


def analyze_portfolio():
    """
    Perform full analysis on all portfolio holdings using fundamental,
    technical, and news analysis.
    
    Returns dict with analysis scores and signals for each holding.
    """
    portfolio = _load_portfolio()
    
    if not portfolio:
        return {
            "error": "Portfolio not found. Run create_empty_portfolio() to initialize.",
        }
    
    holdings = portfolio.get("holdings", [])
    
    if not holdings:
        return {
            "error": "Portfolio is empty.",
        }
    
    # Start with portfolio status
    status = get_portfolio_status()
    if "error" in status:
        return status
    
    # Analyze each holding
    analysis_results = []
    
    for holding in holdings:
        ticker = holding.get("ticker", "").upper()
        
        # Run all three analyses
        fund = fundamental_analyze(ticker)
        tech = technical_analyze(ticker)
        
        # Skip news for now if it has errors
        # news = news_fetcher_analyze(ticker)
        
        if "error" not in fund and "error" not in tech:
            analysis_results.append({
                "ticker": ticker,
                "fundamental_score": fund.get("fundamental_score"),
                "technical_score": tech.get("technical_score"),
                "combined_score": (
                    fund.get("fundamental_score", 0) + 
                    tech.get("technical_score", 0)
                ) / 2,
            })
    
    # Build summary
    summary_lines = [
        "Portfolio Analysis Summary",
        "=" * 40,
        status.get("summary", ""),
        "",
        "Analysis Scores (per holding):",
    ]
    
    for result in analysis_results:
        summary_lines.append(
            f"  {result['ticker']}: Fund={result['fundamental_score']}/10, "
            f"Tech={result['technical_score']}/10, "
            f"Combined={result['combined_score']:.1f}/10"
        )
    
    summary_lines.append("")
    summary_lines.append("This is not financial advice. Always do your own research.")
    
    summary = "\n".join(summary_lines)
    
    return {
        "status": status,
        "analysis": analysis_results,
        "summary": summary,
    }


def create_empty_portfolio():
    """Create and save an empty portfolio template for the user to edit."""
    _create_empty_portfolio()
    return {
        "message": f"Portfolio template created at {_get_portfolio_path()}",
        "template": _load_portfolio(),
    }


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

def get_risk_metrics():
    """
    Compute portfolio-level risk metrics vs the S&P 500 (SPY):
      - Portfolio beta
      - Annualised Sharpe ratio  (risk-free rate assumed 5%)
      - Maximum drawdown (worst peak-to-trough over 1 year)

    Returns a dict with: beta, sharpe_ratio, max_drawdown, summary
    """
    portfolio = _load_portfolio()
    if not portfolio:
        return {"error": "Portfolio not found."}

    holdings = portfolio.get("holdings", [])
    if not holdings:
        return {"error": "Portfolio is empty."}

    tickers = [h["ticker"].upper() for h in holdings]
    shares  = {h["ticker"].upper(): h["shares"] for h in holdings}

    try:
        import pandas as pd
        import yfinance as yf

        # Fetch 1 year of daily closes for all holdings + SPY benchmark
        all_tickers = tickers + ["SPY"]
        raw = yf.download(all_tickers, period="1y", auto_adjust=True, progress=False)
        closes = raw["Close"] if "Close" in raw else raw

        if closes.empty:
            return {"error": "Could not fetch price history."}

        # Build daily portfolio value series (shares * price)
        port_value = sum(
            closes[t] * shares[t]
            for t in tickers
            if t in closes.columns
        )
        port_value = port_value.dropna()

        if len(port_value) < 20:
            return {"error": "Not enough history to compute risk metrics."}

        port_returns = port_value.pct_change().dropna()
        spy_returns  = closes["SPY"].pct_change().dropna() if "SPY" in closes.columns else None

        # Beta: cov(portfolio, SPY) / var(SPY)
        beta = None
        if spy_returns is not None:
            aligned = pd.concat([port_returns, spy_returns], axis=1).dropna()
            aligned.columns = ["port", "spy"]
            cov = aligned.cov().loc["port", "spy"]
            var = aligned["spy"].var()
            beta = round(cov / var, 2) if var != 0 else None

        # Sharpe ratio (annualised, risk-free = 5% / 252)
        rf_daily = 0.05 / 252
        excess   = port_returns - rf_daily
        sharpe   = round((excess.mean() / excess.std()) * (252 ** 0.5), 2) if excess.std() != 0 else None

        # Max drawdown
        cumulative = (1 + port_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = round(drawdown.min(), 4)  # negative number

        def fmt(v, decimals=2):
            return f"{v:.{decimals}f}" if v is not None else "N/A"

        summary_lines = [
            "Portfolio Risk Metrics",
            "=" * 40,
            f"  Beta vs S&P 500 : {fmt(beta)}  {'(market-like)' if beta and 0.9 < beta < 1.1 else '(aggressive)' if beta and beta >= 1.1 else '(defensive)' if beta else ''}",
            f"  Sharpe Ratio    : {fmt(sharpe)}  {'(strong)' if sharpe and sharpe > 1 else '(acceptable)' if sharpe and sharpe > 0.5 else '(weak)' if sharpe else ''}",
            f"  Max Drawdown    : {fmt(max_dd * 100, 1)}%  (worst peak-to-trough, 1yr)",
            "",
            "This is not financial advice. Always do your own research.",
        ]

        return {
            "beta":         beta,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "summary":      "\n".join(summary_lines),
        }

    except Exception as e:
        return {"error": f"Risk metrics failed: {e}"}


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

def get_correlation_matrix():
    """
    Compute the pairwise daily-return correlation matrix for all holdings.
    Returns a dict with: matrix (dict of dicts), summary (formatted table)
    """
    portfolio = _load_portfolio()
    if not portfolio:
        return {"error": "Portfolio not found."}

    holdings = portfolio.get("holdings", [])
    tickers  = [h["ticker"].upper() for h in holdings]

    if len(tickers) < 2:
        return {"error": "Need at least 2 holdings to compute correlations."}

    try:
        import yfinance as yf
        import pandas as pd

        raw    = yf.download(tickers, period="1y", auto_adjust=True, progress=False)
        closes = raw["Close"] if "Close" in raw else raw
        rets   = closes.pct_change().dropna()

        if rets.empty:
            return {"error": "Could not fetch price history."}

        corr = rets.corr().round(2)

        # Build ASCII table
        cols = [t for t in tickers if t in corr.columns]
        col_w = 8
        header = "         " + "".join(f"{c:>{col_w}}" for c in cols)
        rows   = [header, "-" * len(header)]
        for r in cols:
            row = f"{r:<8} " + "".join(
                f"{corr.loc[r, c]:>{col_w}.2f}" if c in corr.columns else f"{'N/A':>{col_w}}"
                for c in cols
            )
            rows.append(row)

        summary_lines = [
            "Portfolio Correlation Matrix (1-Year Daily Returns)",
            "=" * 40,
            *rows,
            "",
            "  Note: Values near 1.0 = highly correlated (concentrated risk).",
            "  Values near 0 = diversified. Values near -1 = hedged.",
            "",
            "This is not financial advice. Always do your own research.",
        ]

        return {
            "matrix":  corr.to_dict(),
            "tickers": cols,
            "summary": "\n".join(summary_lines),
        }

    except Exception as e:
        return {"error": f"Correlation failed: {e}"}


# ---------------------------------------------------------------------------
# Portfolio dashboard chart
# ---------------------------------------------------------------------------

def generate_portfolio_chart(output_dir="data"):
    """
    Generate a 2-panel portfolio dashboard chart:
      Left  — Pie chart of holdings by current value
      Right — Horizontal bar chart of P&L per position

    Saves to data/portfolio_chart.png. Returns the file path.
    """
    portfolio = _load_portfolio()
    if not portfolio:
        return ""

    holdings = portfolio.get("holdings", [])
    if not holdings:
        return ""

    status = get_portfolio_status()
    if "error" in status:
        return ""

    positions = status["positions"]
    if not positions:
        return ""

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import os

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "portfolio_chart.png")

        labels = [p["ticker"] for p in positions]
        values = [p["current_value"] for p in positions]
        gains  = [p["gain_loss"] for p in positions]
        colors_bar = ["#26a69a" if g >= 0 else "#ef5350" for g in gains]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
        fig.patch.set_facecolor("#0f0f0f")

        for ax in (ax1, ax2):
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="#cccccc")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")

        # Pie chart
        pie_colors = ["#4fc3f7", "#ffb300", "#ef5350", "#26a69a", "#ce93d8",
                      "#ff8a65", "#a5d6a7", "#90caf9"]
        wedges, texts, autotexts = ax1.pie(
            values, labels=labels, autopct="%1.1f%%",
            colors=pie_colors[:len(labels)], startangle=140,
            textprops={"color": "#cccccc"},
        )
        for at in autotexts:
            at.set_color("#ffffff")
        ax1.set_title("Holdings by Value", color="#ffffff", pad=12)

        # P&L bar chart (horizontal)
        bars = ax2.barh(labels, gains, color=colors_bar, alpha=0.85)
        ax2.axvline(0, color="#555555", linewidth=0.8)
        ax2.set_xlabel("Gain / Loss ($)", color="#cccccc")
        ax2.set_title("Unrealised P&L per Position", color="#ffffff", pad=12)
        ax2.xaxis.label.set_color("#cccccc")

        # Annotate bars
        for bar, g in zip(bars, gains):
            ax2.text(
                bar.get_width() + (max(abs(v) for v in gains) * 0.02),
                bar.get_y() + bar.get_height() / 2,
                f"${g:+,.0f}",
                va="center", color="#cccccc", fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    except Exception:
        return ""


# ---------------------------------------------------------------------------
# What-if simulator
# ---------------------------------------------------------------------------

def what_if_analysis(new_ticker: str, new_shares: int, new_buy_price: float = None):
    """
    Show how adding a new position would change portfolio composition.

    Args:
        new_ticker    – ticker to add
        new_shares    – number of shares to add
        new_buy_price – optional buy price; fetches current price if None

    Returns dict with: before, after, delta, summary
    """
    portfolio = _load_portfolio()
    if not portfolio:
        return {"error": "Portfolio not found."}

    holdings = portfolio.get("holdings", [])
    status   = get_portfolio_status()
    if "error" in status:
        return status

    # Fetch price for new ticker
    try:
        info = yf.Ticker(new_ticker.upper()).info
        new_price = new_buy_price or info.get("currentPrice") or info.get("regularMarketPrice")
        new_name  = info.get("shortName") or new_ticker.upper()
        sector    = info.get("sector", "Unknown")
    except Exception:
        return {"error": f"Could not fetch data for {new_ticker}."}

    if not new_price:
        return {"error": f"Could not determine price for {new_ticker}."}

    new_value   = new_shares * new_price
    current_val = status["total_current_value"]
    new_total   = current_val + new_value
    new_weight  = new_value / new_total * 100 if new_total > 0 else 0

    # Sector breakdown before
    sectors_before = {}
    for pos in status["positions"]:
        try:
            s = yf.Ticker(pos["ticker"]).info.get("sector", "Unknown")
        except Exception:
            s = "Unknown"
        sectors_before[s] = sectors_before.get(s, 0) + pos["current_value"]

    # Sector breakdown after
    sectors_after = dict(sectors_before)
    sectors_after[sector] = sectors_after.get(sector, 0) + new_value

    def pct_of(val, total):
        return f"{val / total * 100:.1f}%" if total > 0 else "N/A"

    summary_lines = [
        f"What-If Analysis -- Adding {new_shares} x {new_ticker.upper()} @ ${new_price:.2f}",
        "=" * 40,
        f"  New Position Value : ${new_value:,.2f}",
        f"  Portfolio Before   : ${current_val:,.2f}",
        f"  Portfolio After    : ${new_total:,.2f}",
        f"  New Ticker Weight  : {new_weight:.1f}%",
        "",
        "  Sector Exposure After:",
    ]
    for s, v in sorted(sectors_after.items(), key=lambda x: -x[1]):
        bar_len = int(v / new_total * 20)
        summary_lines.append(f"    {s:<25} {pct_of(v, new_total):>6}  {'|' * bar_len}")

    summary_lines += [
        "",
        f"  Note: Adding {new_ticker.upper()} in sector '{sector}'.",
        "  Review sector concentration before proceeding.",
        "",
        "This is not financial advice. Always do your own research.",
    ]

    return {
        "new_ticker":     new_ticker.upper(),
        "new_value":      new_value,
        "portfolio_before": current_val,
        "portfolio_after":  new_total,
        "new_weight_pct": new_weight,
        "sectors_after":  sectors_after,
        "summary":        "\n".join(summary_lines),
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    
    if command == "create":
        result = create_empty_portfolio()
        print(result["message"])
        print(json.dumps(result["template"], indent=2))
    elif command == "analyze":
        result = analyze_portfolio()
        if "error" in result:
            print(result["error"])
        else:
            print(result["summary"])
    else:  # status
        result = get_portfolio_status()
        if "error" in result:
            print(result["error"])
        else:
            print(result["summary"])
