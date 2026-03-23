"""
report_app.py
-------------
Public-facing Stock Advisor dashboard.
Enter a ticker → full analysis displayed on screen + downloadable PDF.

Run locally:   streamlit run report_app.py
Deploy:        share.streamlit.io → secrets: ANTHROPIC_API_KEY, NEWSAPI_KEY
"""

import os
import sys
import tempfile
import uuid
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

load_dotenv()
for key in ("ANTHROPIC_API_KEY", "NEWSAPI_KEY"):
    if key not in os.environ:
        try:
            if key in st.secrets:
                os.environ[key] = st.secrets[key]
        except Exception:
            pass

st.set_page_config(
    page_title="Stock Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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


# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Page background */
  html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #0f1117 !important;
  }
  [data-testid="stHeader"] { background: #0f1117 !important; }

  /* Hide default Streamlit chrome */
  #MainMenu, header, footer { visibility: hidden; }

  /* Search bar container */
  .search-wrap {
    max-width: 640px;
    margin: 0 auto 2rem auto;
  }

  /* Score cards */
  .score-card {
    background: #1e2130;
    border: 1px solid #2d3250;
    border-radius: 12px;
    padding: 16px 12px;
    text-align: center;
  }
  .score-card .label {
    font-size: 0.72rem;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
  }
  .score-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e6edf3;
  }
  .score-card .sub {
    font-size: 0.75rem;
    color: #8892b0;
    margin-top: 2px;
  }

  /* Signal badge */
  .badge-buy    { background:#0d3b2e; color:#2ecc71; border:1px solid #2ecc71; border-radius:8px; padding:6px 16px; font-weight:700; font-size:0.9rem; display:inline-block; }
  .badge-hold   { background:#3b3000; color:#f1c40f; border:1px solid #f1c40f; border-radius:8px; padding:6px 16px; font-weight:700; font-size:0.9rem; display:inline-block; }
  .badge-reduce { background:#3b0a0a; color:#e74c3c; border:1px solid #e74c3c; border-radius:8px; padding:6px 16px; font-weight:700; font-size:0.9rem; display:inline-block; }

  /* Section cards */
  .section-card {
    background: #1e2130;
    border: 1px solid #2d3250;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }
  .section-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #64b5f6;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 10px;
  }
  .section-body {
    font-size: 0.9rem;
    color: #c9d1d9;
    line-height: 1.7;
    white-space: pre-wrap;
  }

  /* Metric row inside sections */
  .kv-row { display:flex; gap:24px; flex-wrap:wrap; margin-bottom:10px; }
  .kv { display:flex; flex-direction:column; }
  .kv .k { font-size:0.7rem; color:#8892b0; text-transform:uppercase; letter-spacing:0.05em; }
  .kv .v { font-size:1rem; font-weight:600; color:#e6edf3; }

  /* Headline rows */
  .headline { padding: 7px 0; border-bottom: 1px solid #2d3250; font-size:0.88rem; color:#c9d1d9; }
  .headline:last-child { border-bottom: none; }
  .hl-pos { color:#2ecc71; }
  .hl-neg { color:#e74c3c; }
  .hl-neu { color:#8892b0; }

  /* Divider */
  .divider { border:none; border-top:1px solid #2d3250; margin:24px 0; }

  /* Company header */
  .co-name { font-size:1.8rem; font-weight:800; color:#e6edf3; margin:0; }
  .co-meta { font-size:0.85rem; color:#8892b0; margin-top:4px; }
  .co-price { font-size:2rem; font-weight:700; color:#e6edf3; }
  .co-price-label { font-size:0.75rem; color:#8892b0; text-transform:uppercase; }

  /* Bull / bear */
  .bull-box { background:#0d3b2e; border:1px solid #2ecc71; border-radius:10px; padding:14px 18px; }
  .bear-box { background:#3b0a0a; border:1px solid #e74c3c; border-radius:10px; padding:14px 18px; }
  .bull-box p, .bear-box p { font-size:0.88rem; line-height:1.65; margin:0; color:#c9d1d9; }
  .box-title { font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; }
  .green { color:#2ecc71; } .red { color:#e74c3c; }

  /* Stacked kpi */
  .kpi-stack { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .kpi { background:#141724; border-radius:8px; padding:10px 14px; }
  .kpi .kl { font-size:0.68rem; color:#8892b0; text-transform:uppercase; letter-spacing:0.05em; }
  .kpi .kv2 { font-size:1rem; font-weight:700; color:#e6edf3; margin-top:2px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(val, decimals=1):
    return f"{val*100:+.{decimals}f}%" if val is not None else "N/A"

def _dollar(val, decimals=2):
    return f"${val:,.{decimals}f}" if val is not None else "N/A"

def _fmt(val, suffix="", decimals=2):
    return f"{val:.{decimals}f}{suffix}" if val is not None else "N/A"

def _escape_dollars(text):
    """Escape $ so Streamlit doesn't interpret them as LaTeX math delimiters."""
    return text.replace("$", "&#36;") if text else ""

def _pattern_to_score(pats):
    if "error" in pats or pats.get("similar_setups_count", 0) == 0:
        return 5
    wr20  = pats.get("win_rate_20d")  or 0.5
    avg20 = pats.get("avg_return_20d") or 0
    score = int(wr20 * 6 + min(max(avg20 * 100, -3), 3) + 2)
    return max(0, min(10, score))

def _signal(score):
    if score >= 8: return "STRONG BUY",   "badge-buy"
    if score >= 7: return "BUY",           "badge-buy"
    if score >= 6: return "CAUTIOUS BUY", "badge-buy"
    if score >= 5: return "HOLD",         "badge-hold"
    if score >= 4: return "CAUTIOUS REDUCE","badge-reduce"
    if score >= 3: return "REDUCE",        "badge-reduce"
    return "AVOID",                        "badge-reduce"

def _score_color(s):
    if s >= 7: return "#2ecc71"
    if s >= 5: return "#f1c40f"
    return "#e74c3c"

def _kpi(label, value):
    return f'<div class="kpi"><div class="kl">{label}</div><div class="kv2">{value}</div></div>'

def _section(title, body_html):
    return f'<div class="section-card"><div class="section-title">{title}</div><div class="section-body">{body_html}</div></div>'

def _run_all_analyses(ticker):
    """Run all analyses + Claude. Called in a background thread."""
    tasks = {
        "meta":        lambda: classify(ticker),
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
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = {"error": str(e)}

    meta = results.get("meta", {})
    try:
        results["advice"] = generate_recommendation(
            ticker=ticker,
            instrument_type=meta.get("type", "stock"),
            instrument_meta=meta,
            fundamental=results.get("fundamental", {}),
            technical=results.get("technical", {}),
            news=results.get("news", {}),
            patterns=results.get("patterns", {}),
            analyst=results.get("analyst"),
            peers=results.get("peers"),
            social=results.get("social"),
            smart_money=results.get("smart_money"),
            earnings=results.get("earnings"),
        )
    except Exception as e:
        results["advice"] = {"error": str(e)}

    return results


def run_with_progress(ticker):
    """Run analyses in background thread, show smooth progress bar, cache in session_state."""
    cache_key = f"data_{ticker}"

    # Already done — return instantly, page renders normally
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    result_box = {}
    error_box  = {}

    def _worker():
        try:
            result_box["data"] = _run_all_analyses(ticker)
        except Exception as e:
            error_box["err"] = str(e)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    step_msgs = [
        "Fetching live price & market data…",
        "Analysing fundamentals — P/E, EPS, revenue growth…",
        "Running technical indicators — RSI, MACD, moving averages…",
        "Deep-diving earnings history & SEC 8-K filings…",
        "Scanning news headlines & media sentiment…",
        "Reading social media — StockTwits & Reddit…",
        "Tracking smart money — short interest & institutions…",
        "Gathering Wall Street analyst consensus…",
        "Comparing performance against sector peers…",
        "Matching historical price patterns…",
        "Generating Claude AI investment recommendation…",
        "Compiling your personalised report…",
    ]

    n_steps       = len(step_msgs)
    max_pct       = 96          # leave room for the final "done" jump
    tick          = 0.12        # seconds between updates — smooth motion
    step_secs     = 4.5         # seconds each step is shown before switching
    per_step_pct  = max_pct / n_steps
    per_tick_inc  = per_step_pct / (step_secs / tick)

    bar     = st.progress(0, text=f"**{step_msgs[0]}**")
    current = 0.0
    step_idx = 0

    while thread.is_alive():
        current = min(current + per_tick_inc, max_pct)
        new_idx = min(int(current / per_step_pct), n_steps - 1)
        if new_idx != step_idx:
            step_idx = new_idx
        bar.progress(int(current), text=f"**{step_msgs[step_idx]}**")
        time.sleep(tick)

    thread.join()

    if error_box:
        bar.empty()
        st.error(f"Analysis failed: {error_box['err']}")
        st.stop()

    bar.progress(100, text="**Report ready!**")
    time.sleep(0.6)
    bar.empty()

    # Store result then rerun so Streamlit renders the full dashboard cleanly
    st.session_state[cache_key] = result_box.get("data", {})
    st.rerun()

def build_pdf(ticker, data, scores, advice, tmp_dir):
    fund, tech, news, pats, anl, peers, social, smart, earnings = (
        data["fundamental"], data["technical"], data["news"],
        data["patterns"], data["analyst"], data["peers"],
        data["social"], data["smart_money"], data["earnings"],
    )
    fund_score, tech_score, news_score, anl_score, peer_score, \
    pat_score, social_score, smart_score, earn_score, combined = scores

    score_summary = (
        f"OVERALL SCORE: {combined:.1f}/10  —  {_signal(combined)[0]}\n\n"
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
    radar_path = generate_radar_chart(ticker, {
        "Fundamental": fund_score, "Technical": tech_score,
        "News": news_score, "Analyst": anl_score,
        "Peers": peer_score, "Patterns": pat_score,
    }, output_dir=tmp_dir)
    chart_path = tech.get("chart_path", "")
    sections = {
        "Score Summary":            score_summary,
        "Fundamental Analysis":     fund.get("summary", ""),
        "Technical Analysis":       tech.get("summary", ""),
        "Earnings Analysis":        earnings.get("summary", ""),
        "News & Sentiment":         news.get("summary", ""),
        "Social Media Sentiment":   social.get("summary", ""),
        "Smart Money Activity":     smart.get("summary", ""),
        "Analyst Consensus":        anl.get("summary", ""),
        "Peer Comparison":          peers.get("summary", ""),
        "Historical Patterns":      pats.get("summary", ""),
        "Claude AI Recommendation": advice.get("narrative") or advice.get("error", ""),
    }
    charts = [p for p in [chart_path, radar_path] if p and os.path.exists(p)]
    pdf_path = export_pdf(ticker, sections, charts, output_dir=tmp_dir)
    with open(pdf_path, "rb") as f:
        return f.read()


# ── Search bar ────────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p style="text-align:center;font-size:2rem;font-weight:800;color:#e6edf3;margin-bottom:4px;">📈 Stock Advisor</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;font-size:0.95rem;color:#8892b0;margin-bottom:28px;">Enter any stock ticker for a full AI-powered analysis report</p>', unsafe_allow_html=True)

col_l, col_mid, col_r = st.columns([1, 3, 1])
with col_mid:
    ticker_input = st.text_input(
        "", placeholder="Enter ticker  e.g. AAPL · MSFT · TSLA · SPY",
        max_chars=10, key="ticker",
        label_visibility="collapsed",
    ).upper().strip()
    go = st.button("Analyze", type="primary", use_container_width=True)

# Persist the ticker across reruns (button resets to False on st.rerun())
if go and ticker_input:
    st.session_state["active_ticker"] = ticker_input

active_ticker = st.session_state.get("active_ticker", "")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

if not active_ticker:
    st.markdown("""
    <div style="text-align:center;padding:40px 0;color:#8892b0;">
      <p style="font-size:1rem;">Enter a ticker above and click <strong style="color:#e6edf3;">Analyze</strong> to get started.</p>
      <p style="font-size:0.85rem;margin-top:8px;">
        Covers: Fundamentals · Technicals · Earnings · News · Social Sentiment ·
        Smart Money · Analyst Consensus · Peer Comparison · Historical Patterns · Claude AI
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Run analyses ──────────────────────────────────────────────────────────────

data = run_with_progress(active_ticker)

if not data:
    st.error("No data returned. Please try again.")
    st.stop()

meta     = data.get("meta", {})
fund     = data.get("fundamental", {})
tech     = data.get("technical", {})
news     = data.get("news", {})
pats     = data.get("patterns", {})
anl      = data.get("analyst", {})
peers    = data.get("peers", {})
social   = data.get("social", {})
smart    = data.get("smart_money", {})
earnings = data.get("earnings", {})

if meta.get("type") == "unknown":
    st.error(f"Could not find **{active_ticker}**. Please check the symbol.")
    st.stop()

itype = meta.get("type", "stock")

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

scores_tuple = (fund_score, tech_score, news_score, anl_score, peer_score,
                pat_score, social_score, smart_score, earn_score, combined)

signal_label, signal_class = _signal(combined)
price = fund.get("current_price") or tech.get("current_price")


advice = data.get("advice", {})


# ── Company header + PDF button ───────────────────────────────────────────────

hdr_left, hdr_right = st.columns([3, 1])

with hdr_left:
    st.markdown(f'<p class="co-name">{meta.get("display_name", active_ticker)}</p>', unsafe_allow_html=True)
    meta_parts = []
    if meta.get("sector"):
        meta_parts.append(meta["sector"])
    if meta.get("industry"):
        meta_parts.append(meta["industry"])
    if meta.get("category"):
        meta_parts.append(meta["category"])
    meta_parts.append(active_ticker)
    st.markdown(f'<p class="co-meta">{" · ".join(meta_parts)}</p>', unsafe_allow_html=True)

    if price:
        st.markdown(f'<p class="co-price-label">Current Price</p><p class="co-price">{_dollar(price)}</p>', unsafe_allow_html=True)

    st.markdown(f'<span class="{signal_class}">{signal_label} &nbsp; {combined:.1f} / 10</span>', unsafe_allow_html=True)

with hdr_right:
    st.markdown("<br><br>", unsafe_allow_html=True)
    pdf_key = f"pdf_{active_ticker}"

    if pdf_key in st.session_state:
        # Already built — show download button directly
        st.download_button(
            label="⬇ Download PDF Report",
            data=st.session_state[pdf_key],
            file_name=f"{active_ticker}_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        if st.button("📄 Generate PDF Report", use_container_width=True):
            with st.spinner("Building PDF…"):
                try:
                    tmp_dir = tempfile.mkdtemp(prefix=f"rpt_{uuid.uuid4().hex[:8]}_")
                    pdf_bytes = build_pdf(active_ticker, data, scores_tuple, advice, tmp_dir)
                    st.session_state[pdf_key] = pdf_bytes
                    st.rerun()
                except Exception as e:
                    st.warning(f"PDF error: {e}")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ── Score cards ───────────────────────────────────────────────────────────────

score_items = [
    ("Overall",     combined,     "avg of all"),
    ("Fundamental", fund_score,   "valuation · quality"),
    ("Technical",   tech_score,   "price · momentum"),
    ("Earnings",    earn_score,   "growth · FCF"),
    ("News",        news_score,   "media sentiment"),
    ("Social",      social_score, "retail mood"),
    ("Smart Money", smart_score,  "institutions"),
    ("Analyst",     anl_score,    "wall street"),
    ("Peers",       peer_score,   "vs sector"),
    ("Patterns",    pat_score,    "historical"),
]
cols = st.columns(10)
for col, (label, score, sub) in zip(cols, score_items):
    color = _score_color(score)
    col.markdown(
        f'<div class="score-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value" style="color:{color}">{score:.1f}</div>'
        f'<div class="sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ── Claude AI Recommendation ──────────────────────────────────────────────────

sections = advice.get("sections", {})

if "error" in advice:
    st.error(f"Claude API error: {advice['error']}")
else:
    st.markdown("### Claude AI Recommendation")
    rec_col, bc_col, bear_col = st.columns([2, 1, 1])

    with rec_col:
        rec_text = _escape_dollars(sections.get("recommendation", advice.get("narrative", "")))
        st.markdown(
            f'<div class="section-card"><div class="section-title">Recommendation</div>'
            f'<div class="section-body">{rec_text}</div></div>',
            unsafe_allow_html=True,
        )
        risks = _escape_dollars(sections.get("risks", ""))
        if risks:
            st.markdown(
                f'<div class="section-card"><div class="section-title">⚠ Key Risks</div>'
                f'<div class="section-body">{risks}</div></div>',
                unsafe_allow_html=True,
            )

    with bc_col:
        bull = _escape_dollars(sections.get("bull_case", ""))
        if bull:
            st.markdown(
                f'<div class="bull-box"><div class="box-title green">🐂 Bull Case</div><p>{bull}</p></div>',
                unsafe_allow_html=True,
            )

    with bear_col:
        bear = _escape_dollars(sections.get("bear_case", ""))
        if bear:
            st.markdown(
                f'<div class="bear-box"><div class="box-title red">🐻 Bear Case</div><p>{bear}</p></div>',
                unsafe_allow_html=True,
            )

    # Remaining Claude sections
    extra_sections = [
        ("fundamental",  "Fundamental Analysis"),
        ("technical",    "Technical Analysis"),
        ("earnings",     "Earnings Analysis"),
        ("management",   "Management Highlights"),
        ("news_media",   "News Media Sentiment"),
        ("social",       "Social Media Sentiment"),
        ("smart_money",  "Smart Money Activity"),
        ("analyst",      "Analyst View"),
        ("peers",        "Peer Standing"),
        ("patterns",     "Historical Patterns"),
        ("confidence",   "Confidence"),
    ]
    ex_left, ex_right = st.columns(2)
    for i, (key, title) in enumerate(extra_sections):
        body = _escape_dollars(sections.get(key, ""))
        if not body:
            continue
        target = ex_left if i % 2 == 0 else ex_right
        with target:
            with st.expander(title):
                st.markdown(f'<div class="section-body">{body}</div>', unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ── Fundamentals + Technical side by side ────────────────────────────────────

st.markdown("### Market Data")
fd_col, tc_col = st.columns(2)

with fd_col:
    st.markdown("**Fundamentals**")
    if "error" not in fund:
        if itype == "etf":
            metrics_html = (
                f'<div class="kpi-stack">'
                f'{_kpi("Expense Ratio", fund.get("expense_ratio_pct","N/A"))}'
                f'{_kpi("Total Assets",  fund.get("total_assets_fmt","N/A"))}'
                f'{_kpi("Dividend Yield",fund.get("yield_pct","N/A"))}'
                f'{_kpi("YTD Return",    fund.get("ytd_return_pct","N/A"))}'
                f'{_kpi("3-Yr Return",   fund.get("three_yr_return_pct","N/A"))}'
                f'{_kpi("Beta",          str(fund.get("beta","N/A")))}'
                f'</div>'
            )
        else:
            metrics_html = (
                f'<div class="kpi-stack">'
                f'{_kpi("P/E Ratio",      _fmt(fund.get("pe_ratio"), decimals=1))}'
                f'{_kpi("EPS (TTM)",      _dollar(fund.get("eps")))}'
                f'{_kpi("Revenue Growth", _pct(fund.get("revenue_growth")))}'
                f'{_kpi("Profit Margin",  _pct(fund.get("profit_margin")))}'
                f'{_kpi("Debt / Equity",  str(fund.get("debt_to_equity","N/A")))}'
                f'{_kpi("Score",          f"{fund_score}/10")}'
                f'</div>'
            )
        st.markdown(metrics_html, unsafe_allow_html=True)
        with st.expander("Full fundamental report"):
            st.text(fund.get("summary", ""))
    else:
        st.warning(fund["error"])

with tc_col:
    st.markdown("**Technicals**")
    if "error" not in tech:
        chart_path = tech.get("chart_path")
        if chart_path and os.path.exists(chart_path):
            st.image(chart_path, use_container_width=True)
        metrics_html = (
            f'<div class="kpi-stack">'
            f'{_kpi("RSI (14)",  _fmt(tech.get("rsi"), decimals=1))}'
            f'{_kpi("MACD",      _fmt(tech.get("macd"), decimals=3))}'
            f'{_kpi("MA 50",     _dollar(tech.get("ma50")))}'
            f'{_kpi("MA 200",    _dollar(tech.get("ma200")))}'
            f'{_kpi("Volatility",_pct(tech.get("volatility")))}'
            f'{_kpi("Score",     f"{tech_score}/10")}'
            f'</div>'
        )
        st.markdown(metrics_html, unsafe_allow_html=True)
        with st.expander("Full technical report"):
            st.text(tech.get("summary", ""))
    else:
        st.warning(tech["error"])

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ── Earnings ──────────────────────────────────────────────────────────────────

st.markdown("### Earnings")
if "error" not in earnings:
    import pandas as pd
    fwd = earnings.get("forward_data", {}) or {}
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Next Earnings",  earnings.get("next_earnings_date", "N/A"))
    e2.metric("Forward EPS",    _dollar(fwd.get("forward_eps")))
    e3.metric("Forward P/E",    _fmt(fwd.get("forward_pe"), decimals=1))
    e4.metric("EPS Growth Fwd", _pct(fwd.get("eps_growth_fwd")))

    quarters = earnings.get("quarterly_results", [])
    if quarters:
        rows = []
        for q in quarters:
            rows.append({
                "Quarter":       q.get("quarter_label", "?"),
                "Revenue":       f"${q['revenue']/1e9:.2f}B" if q.get("revenue") else "N/A",
                "Gross Margin":  _pct(q.get("gross_margin")),
                "Op Margin":     _pct(q.get("operating_margin")),
                "FCF":           f"${q['free_cash_flow']/1e9:.2f}B" if q.get("free_cash_flow") else "N/A",
                "YoY Revenue":   _pct(q.get("revenue_yoy")),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    surprises = earnings.get("earnings_surprises", [])
    if surprises:
        rows2 = []
        for s in surprises:
            surp = s.get("surprise_pct")
            rxn  = s.get("stock_reaction")
            rows2.append({
                "Date":      s.get("date", "?"),
                "Est EPS":   _dollar(s.get("eps_estimate")),
                "Act EPS":   _dollar(s.get("eps_actual")),
                "Surprise":  _pct(surp/100) if surp is not None else "N/A",
                "Beat/Miss": "✅ BEAT" if (surp or 0) > 0 else "❌ MISS",
                "Stock Rxn": _pct(rxn),
            })
        st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)

    press = earnings.get("press_release_excerpt", "")
    if press:
        with st.expander("SEC 8-K Press Release Excerpt"):
            st.text(press[:3000])
else:
    st.warning(earnings.get("error", "Earnings data unavailable."))

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ── News + Social side by side ────────────────────────────────────────────────

st.markdown("### Sentiment")
ns_col, ss_col = st.columns(2)

with ns_col:
    st.markdown("**News & Media**")
    if "error" not in news:
        n1, n2 = st.columns(2)
        n1.metric("Sentiment", news.get("overall_sentiment", "N/A").upper())
        n2.metric("News Score", f"{news_score}/10")
        for h in news.get("headlines", [])[:8]:
            sent = h.get("sentiment", "neutral")
            icon_class = "hl-pos" if sent == "positive" else "hl-neg" if sent == "negative" else "hl-neu"
            icon = "▲" if sent == "positive" else "▼" if sent == "negative" else "●"
            url   = h.get("url", "")
            title = h.get("title", "")
            link  = f'<a href="{url}" target="_blank" style="color:inherit;text-decoration:none;">{title}</a>' if url else title
            st.markdown(
                f'<div class="headline"><span class="{icon_class}">{icon}</span> {link}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning(news.get("error", "News unavailable."))

with ss_col:
    st.markdown("**Social Media**")
    if "error" not in social:
        sc = social.get("sentiment_counts", {})
        total = sum(sc.values()) or 1
        s1, s2, s3 = st.columns(3)
        s1.metric("Mood",    social.get("overall_sentiment", "N/A").upper())
        s2.metric("Bullish", f"{sc.get('bullish',0)} ({sc.get('bullish',0)*100//total}%)")
        s3.metric("Score",   f"{social_score}/10")
        st.caption(f"StockTwits: {social.get('stocktwits_count',0)}  ·  Reddit: {social.get('reddit_count',0)}")
        for p in social.get("top_posts", [])[:5]:
            sent = p.get("sentiment","neutral")
            icon_class = "hl-pos" if sent == "bullish" else "hl-neg" if sent == "bearish" else "hl-neu"
            icon = "▲" if sent == "bullish" else "▼" if sent == "bearish" else "●"
            src  = p.get("source","")
            age  = f" · {p['age_days']}d ago" if p.get("age_days") is not None else ""
            text = p.get("text","")[:120]
            st.markdown(
                f'<div class="headline"><span class="{icon_class}">{icon}</span> <strong>[{src}]</strong>{age} {text}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning(social.get("error", "Social data unavailable."))

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ── Analyst + Smart Money side by side ───────────────────────────────────────

st.markdown("### Analyst & Institutional Activity")
al_col, sm_col = st.columns(2)

with al_col:
    st.markdown("**Analyst Consensus**")
    if "error" not in anl:
        import pandas as pd
        a1, a2, a3 = st.columns(3)
        a1.metric("Consensus",    str(anl.get("recommendation_key","N/A")).upper())
        a2.metric("Price Target", _dollar(anl.get("target_mean")),
                  delta=f"{anl['upside_pct']*100:+.1f}% upside" if anl.get("upside_pct") is not None else None)
        a3.metric("# Analysts",   str(anl.get("num_analysts","N/A")))
        with st.expander("Full analyst report"):
            st.text(anl.get("summary",""))
    else:
        st.warning(anl.get("error","Analyst data unavailable."))

with sm_col:
    st.markdown("**Smart Money**")
    if "error" not in smart:
        sd   = smart.get("short_data", {}) or {}
        opts = smart.get("options_flow", {}) or {}
        sc_pct = sd.get("short_pct_float")
        sc_chg = sd.get("short_change")
        pcr    = opts.get("put_call_ratio")
        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("Short % Float", f"{sc_pct*100:.1f}%" if sc_pct else "N/A",
                   delta=f"{sc_chg*100:+.1f}% MoM" if sc_chg is not None else None,
                   delta_color="inverse")
        sm2.metric("Put/Call",     f"{pcr:.2f}" if pcr else "N/A")
        sm3.metric("Score",        f"{smart_score}/10")
        insts = smart.get("institutions", [])
        if insts:
            import pandas as pd
            rows = [{"Holder": i["holder"],
                     "% Owned": f"{i['pct_out']*100:.2f}%" if i.get("pct_out") else "N/A"}
                    for i in insts[:5]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning(smart.get("error","Smart money data unavailable."))

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ── Peers + Patterns + Radar ──────────────────────────────────────────────────

st.markdown("### Peers, Patterns & Radar")
pp_col, pat_col, rad_col = st.columns([2, 1, 1])

with pp_col:
    st.markdown("**Peer Comparison**")
    if "error" not in peers:
        import pandas as pd
        peer_data = peers.get("peer_data", [])
        if peer_data:
            rows = [{
                "Ticker":         p.get("ticker","?"),
                "P/E":            p.get("pe","N/A"),
                "Rev Growth":     p.get("revenue_growth","N/A"),
                "Profit Margin":  p.get("profit_margin","N/A"),
                "52W Return":     p.get("return_52w","N/A"),
            } for p in peer_data]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"Peer score: {peer_score}/10")
    else:
        st.warning(peers.get("error","Peer data unavailable."))

with pat_col:
    st.markdown("**Historical Patterns**")
    if "error" not in pats:
        st.metric("Similar Setups",  str(pats.get("similar_setups_count", 0)))
        st.metric("Avg 20d Return",  _pct(pats.get("avg_return_20d")),
                  delta=f"Win rate {pats['win_rate_20d']*100:.0f}%" if pats.get("win_rate_20d") is not None else None)
        named = pats.get("named_patterns", [])
        for name, desc in named:
            st.info(f"**{name}** — {desc}")
    else:
        st.warning(pats.get("error","Pattern data unavailable."))

with rad_col:
    st.markdown("**Score Radar**")
    radar_path = generate_radar_chart(active_ticker, {
        "Fundamental": fund_score, "Technical": tech_score,
        "News": news_score, "Analyst": anl_score,
        "Peers": peer_score, "Patterns": pat_score,
    }, output_dir="data")
    if radar_path and os.path.exists(radar_path):
        st.image(radar_path, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;font-size:0.8rem;color:#555e7b;">'
    '⚠️ Not financial advice. Always do your own research.</p>',
    unsafe_allow_html=True,
)
