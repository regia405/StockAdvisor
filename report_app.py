"""
report_app.py
-------------
Simple public-facing app: enter a ticker → generate a full PDF report → download it.

Run locally:
    streamlit run report_app.py

Deploy:
    Push to GitHub → connect on share.streamlit.io → add secrets
    (ANTHROPIC_API_KEY, NEWSAPI_KEY) in the Streamlit Cloud dashboard.
"""

import os
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

# ── Load env (local dev uses .env; Streamlit Cloud uses st.secrets) ───────────
load_dotenv()
for key in ("ANTHROPIC_API_KEY", "NEWSAPI_KEY"):
    if key not in os.environ:
        try:
            if key in st.secrets:
                os.environ[key] = st.secrets[key]
        except Exception:
            pass

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Report Generator",
    page_icon="📄",
    layout="centered",
)

# ── Skill imports ─────────────────────────────────────────────────────────────
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
from skills.claude_advisor         import generate_recommendation
from skills.report_builder         import generate_radar_chart, export_pdf


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pattern_to_score(pats: dict) -> int:
    if "error" in pats or pats.get("similar_setups_count", 0) == 0:
        return 5
    wr20  = pats.get("win_rate_20d")  or 0.5
    avg20 = pats.get("avg_return_20d") or 0
    score = int(wr20 * 6 + min(max(avg20 * 100, -3), 3) + 2)
    return max(0, min(10, score))


def _signal_label(score: float) -> str:
    if score >= 8:  return "STRONG BUY"
    if score >= 7:  return "BUY"
    if score >= 6:  return "CAUTIOUS BUY"
    if score >= 5:  return "HOLD / NEUTRAL"
    if score >= 4:  return "CAUTIOUS REDUCE"
    if score >= 3:  return "REDUCE"
    return                  "AVOID"


def generate_report(ticker: str, tmp_dir: str) -> bytes:
    """Run all analyses and return PDF bytes."""

    # 9 analyses in parallel
    tasks = {
        "meta":        lambda: classify(ticker),
        "fundamental": lambda: fundamental_analyze(ticker),
        "technical":   lambda: technical_analyze(ticker, output_dir=tmp_dir),
        "news":        lambda: news_analyze(ticker),
        "patterns":    lambda: pattern_analyze(ticker),
        "analyst":     lambda: analyst_analyze(ticker),
        "peers":       lambda: peer_analyze(ticker),
        "social":      lambda: social_analyze(ticker),
        "smart_money": lambda: smart_money_analyze(ticker),
        "earnings":    lambda: earnings_analyze(ticker),
    }
    results = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = {"error": str(e)}

    meta     = results.get("meta", {})
    fund     = results.get("fundamental", {})
    tech     = results.get("technical", {})
    news     = results.get("news", {})
    pats     = results.get("patterns", {})
    anl      = results.get("analyst", {})
    peers    = results.get("peers", {})
    social   = results.get("social", {})
    smart    = results.get("smart_money", {})
    earnings = results.get("earnings", {})
    itype    = meta.get("type", "stock")

    # Scores
    fund_score   = fund.get("fundamental_score", 0)
    tech_score   = tech.get("technical_score",   0)
    news_score   = news.get("news_score",         5)
    anl_score    = anl.get("analyst_score",       5)
    peer_score   = peers.get("peer_score",        5)
    pat_score    = _pattern_to_score(pats)
    social_score = social.get("social_score",     5)
    smart_score  = smart.get("smart_score",       5)
    earn_score   = earnings.get("earnings_score", 5)
    combined     = (fund_score + tech_score + news_score + anl_score +
                    peer_score + pat_score + social_score + smart_score +
                    earn_score) / 9

    # Claude AI narrative
    advice = generate_recommendation(
        ticker=ticker, instrument_type=itype, instrument_meta=meta,
        fundamental=fund, technical=tech, news=news, patterns=pats,
        analyst=anl, peers=peers, social=social, smart_money=smart,
        earnings=earnings,
    )

    # Score summary block prepended to the Claude section
    score_summary = (
        f"OVERALL SCORE: {combined:.1f}/10  —  {_signal_label(combined)}\n\n"
        f"  Fundamental  : {fund_score}/10\n"
        f"  Technical    : {tech_score}/10\n"
        f"  Earnings     : {earn_score}/10\n"
        f"  News Media   : {news_score}/10\n"
        f"  Social Media : {social_score}/10\n"
        f"  Smart Money  : {smart_score}/10\n"
        f"  Analyst      : {anl_score}/10\n"
        f"  Peers        : {peer_score}/10\n"
        f"  Patterns     : {pat_score}/10\n"
    )

    # Radar chart
    radar_path = generate_radar_chart(ticker, {
        "Fundamental": fund_score, "Technical": tech_score,
        "News":        news_score, "Analyst":   anl_score,
        "Peers":       peer_score, "Patterns":  pat_score,
    }, output_dir=tmp_dir)

    chart_path = tech.get("chart_path", "")

    # PDF sections
    sections = {
        "Score Summary":          score_summary,
        "Fundamental Analysis":   fund.get("summary", ""),
        "Technical Analysis":     tech.get("summary", ""),
        "Earnings Analysis":      earnings.get("summary", ""),
        "News & Sentiment":       news.get("summary", ""),
        "Social Media Sentiment": social.get("summary", ""),
        "Smart Money Activity":   smart.get("summary", ""),
        "Analyst Consensus":      anl.get("summary", ""),
        "Peer Comparison":        peers.get("summary", ""),
        "Historical Patterns":    pats.get("summary", ""),
        "Claude AI Recommendation": advice.get("narrative") or advice.get("error", "Claude analysis unavailable."),
    }
    charts = [p for p in [chart_path, radar_path] if p and os.path.exists(p)]
    pdf_path = export_pdf(ticker, sections, charts, output_dir=tmp_dir)

    with open(pdf_path, "rb") as f:
        return f.read()


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("Stock Report Generator")
st.markdown(
    "Enter any stock ticker to generate a **comprehensive PDF analysis report** "
    "powered by real market data and Claude AI."
)
st.divider()

col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input(
        "Ticker symbol",
        placeholder="e.g. AAPL, MSFT, TSLA, SPY",
        max_chars=10,
        label_visibility="collapsed",
    ).upper().strip()
with col2:
    generate_btn = st.button("Generate Report", type="primary", use_container_width=True)

st.caption(
    "Includes: Fundamental · Technical · Earnings · News · Social Sentiment · "
    "Smart Money · Analyst Consensus · Peer Comparison · Historical Patterns · Claude AI"
)

if generate_btn:
    if not ticker_input:
        st.warning("Please enter a ticker symbol.")
        st.stop()

    # Quick validity check
    with st.spinner("Verifying ticker…"):
        meta = classify(ticker_input)

    if meta.get("type") == "unknown":
        st.error(f"Could not find ticker **{ticker_input}**. Please check the symbol and try again.")
        st.stop()

    display_name = meta.get("display_name", ticker_input)
    st.success(f"Generating report for **{display_name}** ({ticker_input})…")

    progress = st.progress(0, text="Running 10 analyses in parallel…")
    status   = st.empty()

    steps = [
        (10,  "Fetching fundamental data…"),
        (20,  "Running technical analysis…"),
        (30,  "Analysing earnings…"),
        (40,  "Scanning news & sentiment…"),
        (50,  "Checking social media…"),
        (60,  "Analysing smart money…"),
        (70,  "Gathering analyst consensus…"),
        (80,  "Comparing peers…"),
        (85,  "Finding historical patterns…"),
        (90,  "Generating Claude AI narrative…"),
        (95,  "Building PDF…"),
    ]

    # We run the full pipeline; update the progress bar in parallel using a thread
    import threading

    pdf_result   = {}
    error_result = {}

    def _run():
        tmp_dir = tempfile.mkdtemp(prefix=f"stockrpt_{uuid.uuid4().hex[:8]}_")
        try:
            pdf_result["bytes"] = generate_report(ticker_input, tmp_dir)
        except Exception as e:
            error_result["msg"] = str(e)

    thread = threading.Thread(target=_run)
    thread.start()

    # Animate progress while analysis runs
    import time
    step_idx = 0
    while thread.is_alive():
        if step_idx < len(steps):
            pct, msg = steps[step_idx]
            progress.progress(pct, text=msg)
            status.caption(msg)
            step_idx += 1
        time.sleep(2.5)

    thread.join()
    progress.progress(100, text="Done!")
    status.empty()

    if error_result:
        st.error(f"Analysis failed: {error_result['msg']}")
        st.stop()

    pdf_bytes = pdf_result.get("bytes")
    if not pdf_bytes:
        st.error("PDF generation failed. Please try again.")
        st.stop()

    st.success(f"Report ready for **{ticker_input}**!")
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name=f"{ticker_input}_stock_report.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )

st.divider()
st.caption("Not financial advice. Always do your own research.")
