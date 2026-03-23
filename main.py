"""
main.py
-------
Stock Advisor — full CLI.

COMMANDS:
  python main.py analyze   <TICKER>              Full analysis (all modules + Claude AI)
  python main.py watchlist <T1> <T2> ...         Quick score table for multiple tickers
  python main.py fundamental <TICKER>            Fundamentals only
  python main.py technical  <TICKER>             Technicals only
  python main.py news       <TICKER>             News & sentiment only
  python main.py patterns   <TICKER>             Historical patterns only
  python main.py analyst    <TICKER>             Analyst consensus & earnings only
  python main.py peers      <TICKER>             Peer comparison only

  python main.py portfolio                       Portfolio status
  python main.py portfolio analyze               Full portfolio analysis + Claude narrative
  python main.py portfolio risk                  Beta, Sharpe, max drawdown
  python main.py portfolio correlation           Correlation matrix
  python main.py portfolio chart                 Generate dashboard chart
  python main.py portfolio create                Create empty portfolio template
  python main.py whatif <TICKER> <SHARES>        What-if: add a position

  python main.py alerts                          List all price alerts
  python main.py alerts add <TICKER> above|below <PRICE>
  python main.py alerts check                    Check all alerts vs live prices
  python main.py alerts remove <TICKER>

  python main.py digest                          Weekly portfolio digest (Claude summary)
"""

import sys
import os
import json
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

from skills.instrument_classifier import classify
from skills.fundamental_analysis  import analyze as fundamental_analyze
from skills.technical_analysis    import analyze as technical_analyze
from skills.news_fetcher           import analyze as news_analyze
from skills.pattern_analysis       import analyze as pattern_analyze
from skills.analyst_data           import analyze as analyst_analyze
from skills.peer_comparison        import analyze as peer_analyze
from skills.social_sentiment       import analyze as social_analyze
from skills.smart_money            import analyze as smart_money_analyze
from skills.earnings_analysis      import analyze as earnings_analyze
from skills.claude_advisor         import generate_recommendation, generate_portfolio_narrative
from skills.report_builder         import generate_radar_chart, export_pdf
from skills.portfolio_manager      import (
    get_portfolio_status, analyze_portfolio, create_empty_portfolio,
    get_risk_metrics, get_correlation_matrix, generate_portfolio_chart,
    what_if_analysis,
)

W = 62   # separator width
ALERTS_FILE = "data/alerts.json"


# ---------------------------------------------------------------------------
# Full single-ticker analysis
# ---------------------------------------------------------------------------

def print_analysis_report(ticker: str, save_pdf: bool = False):
    ticker = ticker.upper().strip()

    print(f"\n{'=' * W}")
    print(f"  STOCK ADVISOR  --  {ticker}")
    print(f"{'=' * W}")

    # 1. Classify
    print("\n[1/10] Identifying instrument...")
    meta = classify(ticker)
    if meta["type"] == "unknown":
        print(f"\nERROR: Could not identify '{ticker}'. Check the ticker symbol.")
        return
    itype = meta["type"]
    print(f"      {meta['display_name']}  ({itype.upper()})")
    if meta.get("sector"):
        print(f"      Sector: {meta['sector']}  /  {meta['industry']}")
    if meta.get("category"):
        print(f"      Category: {meta['category']}")

    # 2. Run 9 analyses in parallel
    print("\n[2/10] Running 9 analyses in parallel...")
    tasks = {
        "fundamental": lambda: fundamental_analyze(ticker),
        "technical":   lambda: technical_analyze(ticker),
        "news":        lambda: news_analyze(ticker),
        "patterns":    lambda: pattern_analyze(ticker),
        "analyst":     lambda: analyst_analyze(ticker),
        "peers":       lambda: peer_analyze(ticker),
        "social":      lambda: social_analyze(ticker),
        "smart_money": lambda: smart_money_analyze(ticker),
        "earnings":    lambda: earnings_analyze(ticker),
    }
    results = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = {"error": str(e)}

    fund     = results["fundamental"]
    tech     = results["technical"]
    news     = results["news"]
    pats     = results["patterns"]
    anl      = results["analyst"]
    peers    = results["peers"]
    social   = results["social"]
    smart    = results["smart_money"]
    earnings = results["earnings"]

    # 3–9. Print each module
    _print_section("[3/10] FUNDAMENTAL ANALYSIS",          fund)
    _print_section("[4/10] TECHNICAL ANALYSIS",            tech)
    _print_section("[5/10] HISTORICAL PATTERNS",           pats)
    _print_section("[6/10] EARNINGS ANALYSIS",             earnings)
    _print_section("[7/10] NEWS MEDIA SENTIMENT",          news)
    _print_section("[8/10] SOCIAL MEDIA SENTIMENT",        social)
    _print_section("[9/10] SMART MONEY & MARKET ACTIVITY", smart)
    _print_section("       ANALYST DATA",                  anl)
    _print_section("       PEER COMPARISON",               peers)

    # Combined score (9 signals)
    fund_score     = fund.get("fundamental_score",  0)
    tech_score     = tech.get("technical_score",    0)
    news_score     = news.get("news_score",         5)
    anl_score      = anl.get("analyst_score",       5)
    peer_score     = peers.get("peer_score",        5)
    pat_score      = _pattern_to_score(pats)
    social_score   = social.get("social_score",     5)
    smart_score    = smart.get("smart_score",       5)
    earn_score     = earnings.get("earnings_score", 5)
    combined       = (fund_score + tech_score + news_score + anl_score +
                      peer_score + pat_score + social_score + smart_score +
                      earn_score) / 9

    print(f"\n{'=' * W}")
    print(f"  COMBINED SCORE SUMMARY")
    print(f"{'=' * W}")
    print(f"  Fundamental  : {fund_score}/10")
    print(f"  Technical    : {tech_score}/10")
    print(f"  Earnings     : {earn_score}/10")
    print(f"  News Media   : {news_score}/10")
    print(f"  Social Media : {social_score}/10")
    print(f"  Smart Money  : {smart_score}/10")
    print(f"  Analyst      : {anl_score}/10")
    print(f"  Peers        : {peer_score}/10")
    print(f"  Patterns     : {pat_score}/10")
    print(f"  {'-' * 30}")
    print(f"  OVERALL     : {combined:.1f}/10   {_signal_bar(combined)}")
    print()

    # 10. Claude narrative
    print(f"{'-' * W}")
    print(f"[10/10] CLAUDE AI RECOMMENDATION")
    print(f"{'-' * W}")
    print("  Generating detailed analysis...")

    advice = generate_recommendation(
        ticker=ticker, instrument_type=itype, instrument_meta=meta,
        fundamental=fund, technical=tech, news=news, patterns=pats,
        analyst=anl, peers=peers, social=social, smart_money=smart,
        earnings=earnings,
    )
    if "error" in advice:
        print(f"  ERROR: {advice['error']}")
    else:
        print()
        print(advice["narrative"])

    # Radar chart + optional PDF
    scores = {
        "Fundamental": fund_score, "Technical": tech_score,
        "News": news_score,        "Analyst":   anl_score,
        "Peers": peer_score,       "Patterns":  pat_score,
    }
    radar_path = generate_radar_chart(ticker, scores)
    if radar_path:
        print(f"\n  Radar chart saved to : {radar_path}")

    chart_path = tech.get("chart_path", "")
    if save_pdf:
        sections = advice.get("sections", {})
        report_sections = {
            "Fundamental Analysis": fund.get("summary", ""),
            "Technical Analysis":   tech.get("summary", ""),
            "News & Sentiment":     news.get("summary", ""),
            "Historical Patterns":  pats.get("summary", ""),
            "Analyst Data":         anl.get("summary", ""),
            "Peer Comparison":      peers.get("summary", ""),
            "Claude Recommendation": advice.get("narrative", ""),
        }
        charts = [p for p in [chart_path, radar_path] if p]
        pdf_path = export_pdf(ticker, report_sections, charts)
        if pdf_path:
            print(f"  PDF report saved to  : {pdf_path}")

    print(f"\n{'=' * W}")
    print("  This is not financial advice. Always do your own research.")
    print(f"{'=' * W}\n")


def _print_section(title, result):
    print(f"\n{'-' * W}")
    print(f"{title}")
    print(f"{'-' * W}")
    if "error" in result:
        print(f"  ERROR: {result['error']}")
    else:
        print(result.get("summary", "  (no summary)"))


def _pattern_to_score(pats):
    """Convert pattern win-rate data to a 0-10 score."""
    if "error" in pats or pats.get("similar_setups_count", 0) == 0:
        return 5
    wr20 = pats.get("win_rate_20d") or 0.5
    avg20 = pats.get("avg_return_20d") or 0
    score = int(wr20 * 6 + min(max(avg20 * 100, -3), 3) + 2)
    return max(0, min(10, score))


def _signal_bar(score: float) -> str:
    if score >= 8:  return "[**********]  STRONG BUY"
    if score >= 7:  return "[********* ]  BUY"
    if score >= 6:  return "[*******   ]  CAUTIOUS BUY"
    if score >= 5:  return "[*****     ]  HOLD / NEUTRAL"
    if score >= 4:  return "[***       ]  CAUTIOUS REDUCE"
    if score >= 3:  return "[**        ]  REDUCE"
    return                  "[*         ]  AVOID"


# ---------------------------------------------------------------------------
# Watchlist: quick score table for multiple tickers
# ---------------------------------------------------------------------------

def print_watchlist(tickers: list):
    print(f"\n{'=' * W}")
    print(f"  WATCHLIST SCAN  --  {', '.join(t.upper() for t in tickers)}")
    print(f"{'=' * W}")
    print("  Fetching scores in parallel...\n")

    def _score_ticker(ticker):
        ticker = ticker.upper()
        try:
            f = fundamental_analyze(ticker)
            t = technical_analyze(ticker)
            a = analyst_analyze(ticker)
            fs = f.get("fundamental_score", 0)
            ts = t.get("technical_score", 0)
            as_ = a.get("analyst_score", 5)
            combined = (fs + ts + as_) / 3
            price = f.get("current_price") or t.get("current_price")
            upside = a.get("upside_pct")
            return {
                "ticker": ticker,
                "fund": fs, "tech": ts, "analyst": as_,
                "combined": round(combined, 1),
                "price": price,
                "upside_pct": upside,
                "error": None,
            }
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}

    results = []
    with ThreadPoolExecutor(max_workers=min(len(tickers), 8)) as pool:
        futures = {pool.submit(_score_ticker, t): t for t in tickers}
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda x: x.get("combined", 0), reverse=True)

    # Header
    print(f"  {'Ticker':<8} {'Price':>8} {'Fund':>5} {'Tech':>5} {'Analyst':>7} {'Overall':>8} {'Upside':>8}  Signal")
    print(f"  {'-'*8} {'-'*8} {'-'*5} {'-'*5} {'-'*7} {'-'*8} {'-'*8}  {'-'*14}")

    for r in results:
        if r.get("error"):
            print(f"  {r['ticker']:<8}  ERROR: {r['error']}")
            continue
        price_str  = f"${r['price']:.2f}" if r.get("price") else "N/A"
        upside_str = f"{r['upside_pct']*100:+.1f}%" if r.get("upside_pct") is not None else "N/A"
        signal = _signal_bar(r["combined"]).split("]")[-1].strip()
        print(f"  {r['ticker']:<8} {price_str:>8} {r['fund']:>5} {r['tech']:>5} {r['analyst']:>7} "
              f"{r['combined']:>8.1f} {upside_str:>8}  {signal}")

    print(f"\n  This is not financial advice. Always do your own research.\n")


# ---------------------------------------------------------------------------
# Price alerts
# ---------------------------------------------------------------------------

def _load_alerts():
    path = os.path.join(os.path.dirname(__file__), ALERTS_FILE)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_alerts(alerts):
    path = os.path.join(os.path.dirname(__file__), ALERTS_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(alerts, f, indent=2)


def cmd_alerts(args):
    sub = args[0].lower() if args else "list"

    if sub == "list":
        alerts = _load_alerts()
        if not alerts:
            print("  No alerts set. Use: python main.py alerts add <TICKER> above|below <PRICE>")
            return
        print(f"\n{'=' * W}")
        print(f"  PRICE ALERTS")
        print(f"{'=' * W}")
        for ticker, conditions in alerts.items():
            for direction, price in conditions.items():
                print(f"  {ticker:<8} {direction.upper():>6}  ${price:.2f}")
        print()

    elif sub == "add":
        # alerts add AAPL above 280
        if len(args) < 4:
            print("Usage: python main.py alerts add <TICKER> above|below <PRICE>")
            return
        ticker, direction, price = args[1].upper(), args[2].lower(), float(args[3])
        if direction not in ("above", "below"):
            print("Direction must be 'above' or 'below'.")
            return
        alerts = _load_alerts()
        alerts.setdefault(ticker, {})[direction] = price
        _save_alerts(alerts)
        print(f"  Alert set: {ticker} {direction} ${price:.2f}")

    elif sub == "remove":
        if len(args) < 2:
            print("Usage: python main.py alerts remove <TICKER>")
            return
        ticker = args[1].upper()
        alerts = _load_alerts()
        if ticker in alerts:
            del alerts[ticker]
            _save_alerts(alerts)
            print(f"  Alerts removed for {ticker}.")
        else:
            print(f"  No alerts found for {ticker}.")

    elif sub == "check":
        import yfinance as yf
        alerts = _load_alerts()
        if not alerts:
            print("  No alerts configured.")
            return
        print(f"\n{'=' * W}")
        print(f"  ALERT CHECK  --  {date.today()}")
        print(f"{'=' * W}")
        triggered = []
        for ticker, conditions in alerts.items():
            try:
                info = yf.Ticker(ticker).info
                price = info.get("currentPrice") or info.get("regularMarketPrice")
                if price is None:
                    print(f"  {ticker}: could not fetch price")
                    continue
                for direction, target in conditions.items():
                    hit = (direction == "above" and price >= target) or \
                          (direction == "below" and price <= target)
                    status = "TRIGGERED" if hit else "watching"
                    print(f"  {ticker:<8} current=${price:.2f}  {direction} ${target:.2f}  [{status}]")
                    if hit:
                        triggered.append(f"{ticker} {direction} ${target:.2f} (now ${price:.2f})")
            except Exception as e:
                print(f"  {ticker}: error — {e}")
        if triggered:
            print(f"\n  {len(triggered)} alert(s) triggered:")
            for t in triggered:
                print(f"    *** {t}")
        print()

    else:
        print(f"Unknown alerts sub-command '{sub}'.")


# ---------------------------------------------------------------------------
# Weekly digest
# ---------------------------------------------------------------------------

def cmd_digest():
    print(f"\n{'=' * W}")
    print(f"  WEEKLY DIGEST  --  {date.today()}")
    print(f"{'=' * W}")

    status = get_portfolio_status()
    if "error" in status:
        print(f"  ERROR: {status['error']}")
        return

    holdings = status.get("positions", [])
    if not holdings:
        print("  Portfolio is empty.")
        return

    tickers = [p["ticker"] for p in holdings]
    print(f"  Analysing {len(tickers)} holdings: {', '.join(tickers)}...\n")

    analyses = []
    def _run(pos):
        t = pos["ticker"]
        f = fundamental_analyze(t)
        tech = technical_analyze(t)
        return {
            "ticker":            t,
            "fundamental_score": f.get("fundamental_score", 0),
            "technical_score":   tech.get("technical_score", 0),
            "combined_score":    (f.get("fundamental_score", 0) + tech.get("technical_score", 0)) / 2,
            "return_pct":        pos.get("return_pct", 0),
        }

    with ThreadPoolExecutor(max_workers=min(len(holdings), 6)) as pool:
        futures = [pool.submit(_run, p) for p in holdings]
        for future in as_completed(futures):
            try:
                analyses.append(future.result())
            except Exception:
                pass

    # Print scores table
    print(f"  {'Ticker':<8} {'Fund':>5} {'Tech':>5} {'Combined':>9} {'Portfolio Return':>16}")
    print(f"  {'-'*8} {'-'*5} {'-'*5} {'-'*9} {'-'*16}")
    for a in sorted(analyses, key=lambda x: -x["combined_score"]):
        print(f"  {a['ticker']:<8} {a['fundamental_score']:>5} {a['technical_score']:>5} "
              f"{a['combined_score']:>9.1f} {a['return_pct']:>+15.1f}%")

    # Claude portfolio narrative
    print(f"\n  Generating Claude portfolio assessment...")
    narrative = generate_portfolio_narrative(status.get("summary", ""), analyses)
    if "error" in narrative:
        print(f"  ERROR: {narrative['error']}")
    else:
        print()
        print(narrative["narrative"])

    print(f"\n{'=' * W}")
    print("  This is not financial advice. Always do your own research.")
    print(f"{'=' * W}\n")


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

def print_usage():
    print("""
Stock Advisor -- Personal Investment Analysis Tool

SINGLE TICKER:
  python main.py analyze   <TICKER>          Full analysis (all modules + Claude AI)
  python main.py analyze   <TICKER> --pdf    Same + export PDF report
  python main.py watchlist <T1> <T2> ...     Quick score table for multiple tickers
  python main.py fundamental|technical|news|patterns|analyst|peers <TICKER>

PORTFOLIO:
  python main.py portfolio                   Portfolio status
  python main.py portfolio analyze           Full analysis + Claude narrative
  python main.py portfolio risk              Beta, Sharpe, max drawdown
  python main.py portfolio correlation       Correlation matrix
  python main.py portfolio chart             Generate dashboard chart
  python main.py portfolio create            Create empty portfolio template
  python main.py whatif <TICKER> <SHARES>    What-if: add a new position

ALERTS:
  python main.py alerts                      List alerts
  python main.py alerts add AAPL above 280   Add alert
  python main.py alerts check                Check all alerts vs live prices
  python main.py alerts remove AAPL          Remove alerts for a ticker

OTHER:
  python main.py digest                      Weekly portfolio digest (Claude summary)

EXAMPLES:
  python main.py analyze AAPL
  python main.py analyze AAPL --pdf
  python main.py watchlist AAPL MSFT TSLA NVDA
  python main.py whatif TSLA 10
  python main.py alerts add AAPL below 230
  python main.py digest
""")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1].lower()
    rest    = sys.argv[2:]

    # --- Single ticker analyses ---
    if command == "analyze":
        if not rest:
            print("ERROR: Provide a ticker."); return
        save_pdf = "--pdf" in rest
        ticker   = next(a for a in rest if not a.startswith("-"))
        print_analysis_report(ticker, save_pdf=save_pdf)

    elif command == "watchlist":
        if not rest:
            print("ERROR: Provide at least one ticker."); return
        print_watchlist(rest)

    elif command in ("fundamental", "technical", "news", "patterns", "analyst", "peers", "social", "smartmoney", "earnings"):
        if not rest:
            print(f"ERROR: Provide a ticker."); return
        fn_map = {
            "fundamental": fundamental_analyze,
            "technical":   technical_analyze,
            "news":        news_analyze,
            "patterns":    pattern_analyze,
            "analyst":     analyst_analyze,
            "peers":       peer_analyze,
            "social":      social_analyze,
            "smartmoney":  smart_money_analyze,
            "earnings":    earnings_analyze,
        }
        result = fn_map[command](rest[0].upper())
        print(result.get("error") or result.get("summary", ""))

    # --- Portfolio ---
    elif command == "portfolio":
        sub = rest[0].lower() if rest else "status"

        if sub == "analyze":
            status = get_portfolio_status()
            holdings = status.get("positions", [])
            analyses = []
            for pos in holdings:
                t = pos["ticker"]
                f = fundamental_analyze(t)
                tc = technical_analyze(t)
                analyses.append({
                    "ticker": t,
                    "fundamental_score": f.get("fundamental_score", 0),
                    "technical_score":   tc.get("technical_score", 0),
                    "combined_score":    (f.get("fundamental_score", 0) + tc.get("technical_score", 0)) / 2,
                    "return_pct":        pos.get("return_pct", 0),
                })
            print(status.get("summary", ""))
            narrative = generate_portfolio_narrative(status.get("summary", ""), analyses)
            print("\n" + narrative.get("summary", ""))

        elif sub == "risk":
            r = get_risk_metrics()
            print(r.get("error") or r.get("summary", ""))

        elif sub == "correlation":
            r = get_correlation_matrix()
            print(r.get("error") or r.get("summary", ""))

        elif sub == "chart":
            path = generate_portfolio_chart()
            print(f"  Portfolio chart saved to: {path}" if path else "  ERROR: Could not generate chart.")

        elif sub == "create":
            r = create_empty_portfolio()
            print(r["message"])

        else:
            r = get_portfolio_status()
            print(r.get("error") or r.get("summary", ""))

    elif command == "whatif":
        if len(rest) < 2:
            print("Usage: python main.py whatif <TICKER> <SHARES> [buy_price]"); return
        ticker = rest[0].upper()
        shares = int(rest[1])
        price  = float(rest[2]) if len(rest) >= 3 else None
        r = what_if_analysis(ticker, shares, price)
        print(r.get("error") or r.get("summary", ""))

    # --- Alerts ---
    elif command == "alerts":
        cmd_alerts(rest)

    # --- Digest ---
    elif command == "digest":
        cmd_digest()

    else:
        print(f"ERROR: Unknown command '{command}'")
        print_usage()


if __name__ == "__main__":
    main()
