"""
earnings_analysis.py
--------------------
Comprehensive earnings analysis for a stock, covering:

  Quarterly Results     — Last 4 quarters: revenue, margins, EPS,
                          QoQ and YoY growth rates.
  FCF Trend             — Free cash flow vs net income (quality check).
  Earnings Surprise     — EPS beats/misses vs estimates, last 4 quarters,
                          plus stock price reaction on each earnings day.
  Forward Guidance      — Next quarter EPS estimate, forward PE,
                          expected EPS growth.
  Next Earnings Date    — When to expect the next release.
  SEC 8-K Press Release — The actual earnings press release filed with the
                          SEC (EX-99.1 exhibit). Contains CEO/CFO quotes,
                          business highlights, and official guidance.
                          Fetched from SEC EDGAR free API with local CIK
                          caching to avoid repeated large downloads.

Usage:
    from skills.earnings_analysis import analyze
    result = analyze("AAPL")
    print(result["summary"])
"""

import os
import json
import re
import time
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

SEC_HEADERS = {
    "User-Agent": "StockAdvisor/1.0 personal-research@example.com",
    "Accept-Encoding": "gzip, deflate",
}
CIK_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sec_cik_cache.json")
_CIK_MEM_CACHE: dict = {}


# ---------------------------------------------------------------------------
# SEC EDGAR helpers
# ---------------------------------------------------------------------------

def _load_cik_cache() -> dict:
    try:
        with open(CIK_CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cik_cache(cache: dict):
    try:
        os.makedirs(os.path.dirname(CIK_CACHE_PATH), exist_ok=True)
        with open(CIK_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


def _get_cik(ticker: str) -> str | None:
    """Look up the SEC CIK for a ticker, using a local cache to avoid
    repeatedly downloading the full company_tickers.json (~4 MB)."""
    ticker = ticker.upper()

    if ticker in _CIK_MEM_CACHE:
        return _CIK_MEM_CACHE[ticker]

    disk = _load_cik_cache()
    if ticker in disk:
        _CIK_MEM_CACHE[ticker] = str(disk[ticker])
        return str(disk[ticker])

    # Download the full ticker->CIK map from SEC
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=SEC_HEADERS, timeout=15,
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        mapping = {v["ticker"].upper(): str(v["cik_str"]) for v in data.values()}
        _save_cik_cache(mapping)
        for t, c in mapping.items():
            _CIK_MEM_CACHE[t] = c

        return mapping.get(ticker)
    except Exception:
        return None


def _fetch_earnings_press_release(ticker: str) -> str | None:
    """
    Fetch the most recent earnings press release text from SEC EDGAR.

    Strategy:
      1. Look up company CIK.
      2. Fetch the recent filings index (submissions JSON).
      3. Find the most recent 8-K filing.
      4. Fetch its filing index page and locate the EX-99.1 exhibit
         (always the earnings press release — contains CEO quotes, highlights,
         and guidance).
      5. Strip HTML and return the first 4 000 characters.

    Returns None if anything fails — the rest of the analysis still works.
    """
    cik = _get_cik(ticker)
    if not cik:
        return None

    cik_padded = cik.zfill(10)

    try:
        # Fetch the company's recent filing list
        sub_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        resp = requests.get(sub_url, headers=SEC_HEADERS, timeout=12)
        if resp.status_code != 200:
            return None

        data      = resp.json()
        filings   = data.get("filings", {}).get("recent", {})
        forms     = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])

        # Find the most recent 8-K
        for i, form in enumerate(forms[:40]):
            if form != "8-K":
                continue

            acc_clean   = accessions[i].replace("-", "")
            acc_dashed  = accessions[i]
            index_url   = (
                f"https://www.sec.gov/Archives/edgar/data/{cik}"
                f"/{acc_clean}/{acc_dashed}-index.htm"
            )

            idx_resp = requests.get(index_url, headers=SEC_HEADERS, timeout=10)
            if idx_resp.status_code != 200:
                continue

            # Find the EX-99.1 document link (earnings press release)
            idx_text = idx_resp.text
            # Match links to .htm files in the filing (usually EX-99.1)
            links = re.findall(
                r'href="(/Archives/edgar/data/\d+/\d+/[^"]+\.htm)"',
                idx_text, re.IGNORECASE,
            )
            # Prefer links whose anchor text mentions EX-99 or exhibit
            ex99_links = [
                l for l in links
                if re.search(r'ex.?99|exhibit.?99|earnings|results', idx_text[
                    max(0, idx_text.lower().find(l.lower()) - 50):
                    idx_text.lower().find(l.lower()) + 100
                ], re.IGNORECASE)
            ]
            candidates = ex99_links if ex99_links else links

            for link in candidates[:3]:
                doc_url  = f"https://www.sec.gov{link}"
                doc_resp = requests.get(doc_url, headers=SEC_HEADERS, timeout=10)
                if doc_resp.status_code != 200:
                    continue

                # Strip HTML tags and normalise whitespace
                text = re.sub(r'<[^>]+>', ' ', doc_resp.text)
                text = re.sub(r'[ \t]{2,}', ' ', text)
                text = re.sub(r'\n{3,}', '\n\n', text).strip()

                # Skip documents that look like XBRL data (very short readable text)
                if len(text) < 500:
                    continue

                return text[:4000]

            break  # Only try the most recent 8-K

    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# yfinance helpers
# ---------------------------------------------------------------------------

def _get_row(df, candidates: list):
    """Return the first matching row from a DataFrame by name candidates."""
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    return None


def _safe_float(series, idx) -> float | None:
    try:
        v = series.iloc[idx]
        return float(v) if v is not None and not pd.isna(v) else None
    except Exception:
        return None


def _extract_quarterly_results(stock) -> list:
    """
    Pull last 4 quarters of income statement + cash flow.
    Returns list of dicts, most recent first.
    """
    try:
        # yfinance uses different attribute names across versions
        qf = None
        for attr in ("quarterly_income_stmt", "quarterly_financials"):
            try:
                qf = getattr(stock, attr, None)
                if qf is not None and not qf.empty:
                    break
            except Exception:
                pass
        if qf is None or qf.empty:
            return []

        qcf = None
        for attr in ("quarterly_cash_flow", "quarterly_cashflow"):
            try:
                qcf = getattr(stock, attr, None)
                if qcf is not None and not qcf.empty:
                    break
            except Exception:
                pass

        n = min(4, len(qf.columns))
        rev_row = _get_row(qf, ["Total Revenue", "Revenue"])
        gp_row  = _get_row(qf, ["Gross Profit"])
        oi_row  = _get_row(qf, ["Operating Income", "EBIT", "Operating Income Loss"])
        ni_row  = _get_row(qf, ["Net Income", "Net Income Common Stockholders",
                                 "Net Income Loss"])

        fcf_row = ocf_row = capex_row = None
        if qcf is not None:
            fcf_row   = _get_row(qcf, ["Free Cash Flow"])
            ocf_row   = _get_row(qcf, ["Operating Cash Flow",
                                        "Cash Flow From Continuing Operating Activities"])
            capex_row = _get_row(qcf, ["Capital Expenditure", "Capital Expenditures"])

        quarters = []
        for i in range(n):
            date = qf.columns[i]
            qlabel = f"Q{(date.month - 1) // 3 + 1} {date.year}" if hasattr(date, "month") else str(date)[:10]

            def v(row):
                return _safe_float(row, i) if row is not None else None

            rev  = v(rev_row)
            gp   = v(gp_row)
            oi   = v(oi_row)
            ni   = v(ni_row)
            fcf  = v(fcf_row)
            ocf  = v(ocf_row)
            capex = v(capex_row)

            # Derive FCF from OCF - CapEx if not directly available
            if fcf is None and ocf is not None and capex is not None:
                fcf = ocf + capex  # capex is usually negative in yfinance

            q = {
                "date":             str(date)[:10],
                "quarter_label":    qlabel,
                "revenue":          rev,
                "gross_profit":     gp,
                "operating_income": oi,
                "net_income":       ni,
                "free_cash_flow":   fcf,
            }

            if rev and rev > 0:
                q["gross_margin"]     = gp / rev  if gp  is not None else None
                q["operating_margin"] = oi / rev  if oi  is not None else None
                q["net_margin"]       = ni / rev  if ni  is not None else None
                q["fcf_margin"]       = fcf / rev if fcf is not None else None

            quarters.append(q)

        # YoY growth: most recent (index 0) vs 1 year ago (index 3)
        if len(quarters) >= 4:
            q0, q3 = quarters[0], quarters[3]
            for key in ("revenue", "net_income"):
                v0, v3 = q0.get(key), q3.get(key)
                if v0 is not None and v3 and v3 != 0:
                    q0[f"{key}_yoy"] = (v0 - v3) / abs(v3)

        # QoQ growth
        for i in range(len(quarters) - 1):
            r0, r1 = quarters[i].get("revenue"), quarters[i + 1].get("revenue")
            if r0 is not None and r1 and r1 != 0:
                quarters[i]["revenue_qoq"] = (r0 - r1) / abs(r1)

        return quarters

    except Exception:
        return []


def _get_earnings_surprises_with_reaction(stock) -> list:
    """
    EPS estimate vs actual for last 4 quarters, plus stock price reaction
    (2-day return starting from the earnings date).
    """
    try:
        hist_data = getattr(stock, "earnings_history", None)
        if hist_data is None or hist_data.empty:
            return []

        price_hist = stock.history(period="2y")

        results = []
        for _, row in hist_data.tail(4).iterrows():
            est  = row.get("epsEstimate") or row.get("EPS Estimate")
            act  = row.get("epsActual")   or row.get("Reported EPS")
            surp = row.get("surprisePercent")
            date = row.get("quarter") or row.get("Quarter")

            entry = {
                "date":            str(date)[:10] if date else None,
                "eps_estimate":    float(est)  if est  is not None else None,
                "eps_actual":      float(act)  if act  is not None else None,
                "surprise_pct":    float(surp) if surp is not None else None,
            }

            # Stock reaction: find price 1 day before vs 2 days after
            if date is not None and not price_hist.empty:
                try:
                    ts = pd.Timestamp(date).tz_localize(None)
                    prices = price_hist["Close"]
                    prices.index = prices.index.tz_localize(None)
                    # Find nearest trading day on or after the date
                    after  = prices[prices.index >= ts].head(3)
                    before = prices[prices.index <  ts].tail(1)
                    if len(before) > 0 and len(after) >= 1:
                        reaction = (after.iloc[0] - before.iloc[0]) / before.iloc[0]
                        entry["stock_reaction"] = round(float(reaction), 4)
                except Exception:
                    pass

            results.append(entry)

        return results

    except Exception:
        return []


def _get_forward_data(info: dict) -> dict:
    """Extract forward-looking metrics from yfinance info dict."""
    trailing_eps = info.get("trailingEps")
    forward_eps  = info.get("forwardEps")
    forward_pe   = info.get("forwardPE")
    cq_estimate  = info.get("currentQuarterEstimate")
    cq_date      = info.get("currentQuarterEstimateDate")
    cq_year      = info.get("currentQuarterEstimateYear")
    revenue_est  = info.get("revenueEstimate") or info.get("revenueForecasts")
    earnings_growth = info.get("earningsGrowth")
    revenue_growth  = info.get("revenueGrowth")

    eps_growth = None
    if trailing_eps and forward_eps and trailing_eps != 0:
        eps_growth = (forward_eps - trailing_eps) / abs(trailing_eps)

    return {
        "trailing_eps":     trailing_eps,
        "forward_eps":      forward_eps,
        "forward_pe":       forward_pe,
        "eps_growth_fwd":   eps_growth,
        "cq_estimate":      cq_estimate,
        "cq_label":         f"{cq_date} {cq_year}" if cq_date and cq_year else None,
        "earnings_growth":  earnings_growth,
        "revenue_growth":   revenue_growth,
    }


def _get_next_earnings_date(stock, info: dict) -> str | None:
    """Return next earnings date as a string, trying multiple yfinance sources."""
    # calendar attribute (most reliable)
    try:
        cal = stock.calendar
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed:
                if hasattr(ed, "__iter__") and not isinstance(ed, str):
                    return str(list(ed)[0])[:10]
                return str(ed)[:10]
    except Exception:
        pass

    # earnings timestamp from info
    ts = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
    if ts:
        try:
            return datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _compute_score(surprises: list, quarters: list, forward: dict) -> int:
    """0–10 from 5 metrics x 2 pts each."""

    # 1. Beat consistency
    beats = sum(1 for s in surprises if (s.get("surprise_pct") or 0) > 0)
    beat_score = 2 if beats >= 4 else 1 if beats >= 2 else 0

    # 2. Revenue growth momentum
    rev_yoy = quarters[0].get("revenue_yoy") if quarters else None
    rev_score = 2 if rev_yoy and rev_yoy > 0.10 else 1 if rev_yoy and rev_yoy > 0 else 0

    # 3. Gross margin trend (most recent vs quarter prior)
    gm_now  = quarters[0].get("gross_margin")  if len(quarters) > 0 else None
    gm_prev = quarters[1].get("gross_margin")  if len(quarters) > 1 else None
    margin_score = (
        2 if gm_now and gm_prev and gm_now > gm_prev + 0.005 else
        1 if gm_now and gm_prev and gm_now >= gm_prev - 0.005 else
        0
    )

    # 4. FCF quality (positive FCF and > net income = high quality earnings)
    fcf = quarters[0].get("free_cash_flow") if quarters else None
    ni  = quarters[0].get("net_income")     if quarters else None
    if fcf and fcf > 0:
        fcf_score = 2 if ni and fcf > ni * 0.8 else 1
    else:
        fcf_score = 0

    # 5. Forward EPS growth
    eps_growth = forward.get("eps_growth_fwd")
    fwd_score = 2 if eps_growth and eps_growth > 0.05 else 1 if eps_growth and eps_growth >= 0 else 0

    return beat_score + rev_score + margin_score + fcf_score + fwd_score


def _score_label(score: int) -> str:
    if score >= 8: return "  Signal: EXCELLENT earnings quality. Strong growth, consistent beats, healthy FCF."
    if score >= 6: return "  Signal: GOOD earnings profile. Solid execution with minor concerns."
    if score >= 4: return "  Signal: MIXED earnings picture. Some strengths offset by concerns."
    if score >= 2: return "  Signal: WEAK earnings trend. Execution and/or quality concerns."
    return           "  Signal: POOR earnings quality. Significant financial concerns."


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_bn(v):
    """Format a large number in billions/millions."""
    if v is None:
        return "N/A"
    if abs(v) >= 1e9:
        return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.0f}M"
    return f"${v:,.0f}"


def _fmt_pct(v, decimals=1):
    if v is None:
        return "N/A"
    return f"{v * 100:+.{decimals}f}%"


def _fmt_plain(v, decimals=2):
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def analyze(ticker: str) -> dict:
    """
    Run comprehensive earnings analysis for a ticker.

    Returns a dict with keys:
        ticker, next_earnings_date, quarterly_results, earnings_surprises,
        forward_data, press_release_excerpt, earnings_score, summary
    """
    ticker = ticker.upper().strip()

    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
    except Exception:
        return {"ticker": ticker, "error": f"Could not fetch data for '{ticker}'."}

    # Fetch all components
    quarterly   = _extract_quarterly_results(stock)
    surprises   = _get_earnings_surprises_with_reaction(stock)
    forward     = _get_forward_data(info)
    next_date   = _get_next_earnings_date(stock, info)
    press_rel   = _fetch_earnings_press_release(ticker)
    score       = _compute_score(surprises, quarterly, forward)

    # --- Build summary ---
    lines = [
        f"Earnings Analysis -- {ticker}",
        "=" * 40,
        f"  Next Earnings Date : {next_date or 'N/A'}",
        f"  Current Q Estimate : EPS {_fmt_plain(forward.get('cq_estimate'))} "
        f"({forward.get('cq_label') or 'N/A'})",
        f"  Forward EPS        : {_fmt_plain(forward.get('forward_eps'))}  "
        f"(forward PE: {_fmt_plain(forward.get('forward_pe'))})",
        f"  EPS Growth (fwd)   : {_fmt_pct(forward.get('eps_growth_fwd'))}",
        "",
    ]

    if quarterly:
        q = quarterly[0]
        lines += [
            f"  LATEST QUARTER ({q.get('quarter_label', 'N/A')}):",
            f"    Revenue           : {_fmt_bn(q.get('revenue'))}  "
            f"(YoY: {_fmt_pct(q.get('revenue_yoy'))}  QoQ: {_fmt_pct(q.get('revenue_qoq'))})",
            f"    Gross Margin      : {_fmt_pct(q.get('gross_margin'), 1)}",
            f"    Operating Margin  : {_fmt_pct(q.get('operating_margin'), 1)}",
            f"    Net Income        : {_fmt_bn(q.get('net_income'))}",
            f"    Free Cash Flow    : {_fmt_bn(q.get('free_cash_flow'))}",
            f"    FCF Margin        : {_fmt_pct(q.get('fcf_margin'), 1)}",
            "",
        ]

        if len(quarterly) >= 2:
            lines.append("  QUARTERLY TREND (most recent first):")
            lines.append(f"  {'Quarter':<10} {'Revenue':>10} {'GrossMargin':>12} "
                         f"{'OpMargin':>10} {'FCF':>10}")
            lines.append(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")
            for q in quarterly:
                lines.append(
                    f"  {q.get('quarter_label','?'):<10} "
                    f"{_fmt_bn(q.get('revenue')):>10} "
                    f"{_fmt_pct(q.get('gross_margin'),1):>12} "
                    f"{_fmt_pct(q.get('operating_margin'),1):>10} "
                    f"{_fmt_bn(q.get('free_cash_flow')):>10}"
                )
            lines.append("")

    if surprises:
        beats = sum(1 for s in surprises if (s.get("surprise_pct") or 0) > 0)
        lines.append(f"  EPS SURPRISE HISTORY ({beats}/{len(surprises)} beats):")
        lines.append(f"  {'Date':<12} {'Est EPS':>8} {'Act EPS':>8} "
                     f"{'Surprise':>10} {'Stock Rxn':>10}")
        lines.append(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
        for s in surprises:
            surp = s.get("surprise_pct")
            tag  = "BEAT" if (surp or 0) > 0 else "MISS"
            rxn  = s.get("stock_reaction")
            rxn_str = f"{rxn*100:+.1f}%" if rxn is not None else "N/A"
            date_str = s.get('date') or '?'
            lines.append(
                f"  {date_str:<12} "
                f"{_fmt_plain(s.get('eps_estimate')):>8} "
                f"{_fmt_plain(s.get('eps_actual')):>8} "
                f"{f'{surp:+.1f}% {tag}' if surp is not None else 'N/A':>10} "
                f"{rxn_str:>10}"
            )
        lines.append("")

    if press_rel:
        lines += [
            "  SEC 8-K PRESS RELEASE EXCERPT:",
            "  (CEO/CFO quotes, business highlights, and guidance are in this filing.)",
            "  Claude will extract key management themes in the AI recommendation section.",
            f"  [First 300 chars: {press_rel[:300].strip()}...]",
            "",
        ]
    else:
        lines += [
            "  SEC 8-K Press Release: Not fetched or not available.",
            "  (Check company investor relations page for management commentary.)",
            "",
        ]

    lines += [
        f"  Earnings Score     : {score} / 10",
        "",
        _score_label(score),
        "",
        "This is not financial advice. Always do your own research.",
    ]

    return {
        "ticker":                ticker,
        "next_earnings_date":    next_date,
        "quarterly_results":     quarterly,
        "earnings_surprises":    surprises,
        "forward_data":          forward,
        "press_release_excerpt": press_rel,
        "earnings_score":        score,
        "summary":               "\n".join(lines),
    }


if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    r = analyze(symbol)
    print(r.get("error") or r["summary"])
