"""
app.py
------
Streamlit web interface for Stock Advisor.

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub → connect on share.streamlit.io → add secrets
    (ANTHROPIC_API_KEY, NEWSAPI_KEY) in the Streamlit Cloud dashboard.
"""

import streamlit as st
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Stock Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load .env for local dev; on Streamlit Cloud use st.secrets ───────────────
from dotenv import load_dotenv
load_dotenv()

# Merge Streamlit secrets into environment so existing skills pick them up
for key in ("ANTHROPIC_API_KEY", "NEWSAPI_KEY", "GEMINI_API_KEY"):
    if key not in os.environ:
        try:
            if key in st.secrets:
                os.environ[key] = st.secrets[key]
        except Exception:
            pass  # no secrets.toml locally — that's fine, .env is used instead

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

def _score_color(score: float) -> str:
    if score >= 7:  return "normal"   # green delta
    if score >= 4:  return "off"      # yellow / neutral
    return "inverse"                  # red


def _signal_label(score: float) -> str:
    if score >= 8:  return "🟢 STRONG BUY"
    if score >= 7:  return "🟢 BUY"
    if score >= 6:  return "🟡 CAUTIOUS BUY"
    if score >= 5:  return "🟡 HOLD / NEUTRAL"
    if score >= 4:  return "🟠 CAUTIOUS REDUCE"
    if score >= 3:  return "🔴 REDUCE"
    return                  "🔴 AVOID"


def _pattern_to_score(pats: dict) -> int:
    if "error" in pats or pats.get("similar_setups_count", 0) == 0:
        return 5
    wr20  = pats.get("win_rate_20d")  or 0.5
    avg20 = pats.get("avg_return_20d") or 0
    score = int(wr20 * 6 + min(max(avg20 * 100, -3), 3) + 2)
    return max(0, min(10, score))


def _get_analyses(ticker: str) -> dict:
    """Run all 9 modules once per ticker, cached in session_state."""
    key = f"analyses_{ticker}"
    if key not in st.session_state:
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
        st.session_state[key] = results
    return st.session_state[key]


def _get_advice(ticker: str, provider: str, data: dict) -> dict:
    """Run AI recommendation once per ticker+provider, cached in session_state."""
    key = f"advice_{ticker}_{provider}"
    if key not in st.session_state:
        meta = data.get("meta", {})
        st.session_state[key] = generate_recommendation(
            ticker=ticker,
            instrument_type=meta.get("type", "stock"),
            instrument_meta=meta,
            fundamental=data.get("fundamental", {}),
            technical=data.get("technical", {}),
            news=data.get("news", {}),
            patterns=data.get("patterns", {}),
            analyst=data.get("analyst"),
            peers=data.get("peers"),
            social=data.get("social"),
            smart_money=data.get("smart_money"),
            earnings=data.get("earnings"),
            provider=provider,
        )
    return st.session_state[key]


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 Stock Advisor")
    st.markdown("Personal AI-powered investment analysis")
    st.divider()

    ticker_input = st.text_input(
        "Enter ticker symbol",
        placeholder="e.g. AAPL, MSFT, SPY",
        max_chars=10,
    ).upper().strip()

    ai_provider = st.radio(
        "AI Model",
        options=["Claude Haiku", "Gemini 2.5 Flash"],
        index=0,
        horizontal=True,
    )
    provider = "gemini" if ai_provider == "Gemini 2.5 Flash" else "claude"

    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    export_pdf_btn = st.checkbox("Export PDF report", value=False)

    st.divider()
    st.markdown(
        "**Data sources**\n"
        "- Market data: yfinance\n"
        "- News: NewsAPI\n"
        "- Social: StockTwits + Reddit\n"
        "- Filings: SEC EDGAR\n"
        f"- AI narrative: {ai_provider}\n"
    )
    st.markdown("---")
    st.caption("⚠️ Not financial advice. Always do your own research.")


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("📊 Stock Advisor")

if analyze_btn and ticker_input:
    st.session_state["analyzed_ticker"] = ticker_input

active_ticker = st.session_state.get("analyzed_ticker", "")

if not active_ticker:
    st.markdown(
        """
        ### Welcome
        Enter a ticker symbol in the sidebar and click **Analyze** to get started.

        **What you'll get:**
        - Fundamental & technical scores
        - Earnings deep-dive with SEC 8-K highlights
        - News, social media & smart money sentiment
        - Analyst consensus & peer comparison
        - Historical pattern matching
        - Claude AI investment recommendation
        - Radar chart & optional PDF report
        """
    )
    st.stop()


# ── Run analyses ──────────────────────────────────────────────────────────────

with st.spinner(f"Running 9 analyses for **{active_ticker}**…"):
    data = _get_analyses(active_ticker)

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
    st.error(f"Could not identify ticker **{active_ticker}**. Please check the symbol.")
    st.stop()

# ── Instrument header ─────────────────────────────────────────────────────────

itype = meta.get("type", "stock")
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.header(f"{meta.get('display_name', active_ticker)}  ({active_ticker})")
    if meta.get("sector"):
        st.caption(f"🏭 {meta['sector']}  ·  {meta.get('industry', '')}")
    elif meta.get("category"):
        st.caption(f"📦 {meta['category']}")
with col_h2:
    price = fund.get("current_price") or tech.get("current_price")
    if price:
        st.metric("Current Price", f"${price:,.2f}")

st.divider()

# ── Score dashboard ───────────────────────────────────────────────────────────

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

st.subheader("Score Summary")
cols = st.columns(10)
score_items = [
    ("Overall",      combined,    ""),
    ("Fundamental",  fund_score,  ""),
    ("Technical",    tech_score,  ""),
    ("Earnings",     earn_score,  ""),
    ("News",         news_score,  ""),
    ("Social",       social_score,""),
    ("Smart Money",  smart_score, ""),
    ("Analyst",      anl_score,   ""),
    ("Peers",        peer_score,  ""),
    ("Patterns",     pat_score,   ""),
]
for col, (label, score, _) in zip(cols, score_items):
    delta = f"{score:.1f}/10"
    col.metric(label, f"{score:.1f}", delta=None)

# Overall signal banner
sig_label = _signal_label(combined)
if combined >= 7:
    st.success(f"**{sig_label}** — Overall score {combined:.1f} / 10")
elif combined >= 5:
    st.warning(f"**{sig_label}** — Overall score {combined:.1f} / 10")
else:
    st.error(f"**{sig_label}** — Overall score {combined:.1f} / 10")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    f"🤖 {ai_provider}",
    "📊 Fundamental",
    "📈 Technical",
    "💰 Earnings",
    "📰 News",
    "💬 Social",
    "🐋 Smart Money",
    "🎯 Analyst",
    "🏁 Peers",
    "🔍 Patterns",
    "🕸️ Radar",
])

tab_claude, tab_fund, tab_tech, tab_earn, tab_news, \
tab_social, tab_smart, tab_analyst, tab_peers, tab_patterns, tab_radar = tabs


# ── Claude AI tab ─────────────────────────────────────────────────────────────

with tab_claude:
    st.subheader(f"{ai_provider} Investment Recommendation")
    with st.spinner("Generating AI analysis…"):
        advice = _get_advice(active_ticker, provider, data)

    if "error" in advice:
        st.error(f"AI API error: {advice['error']}")
    else:
        sections = advice.get("sections", {})
        narrative = advice.get("narrative", "")

        # Show recommendation prominently
        if sections.get("recommendation"):
            st.info(f"**RECOMMENDATION**\n\n{sections['recommendation']}")

        # Show bull/bear side by side
        bc, cc = st.columns(2)
        with bc:
            if sections.get("bull_case"):
                st.markdown(
                    f"""
                    <div style="background:#0d3b2e;border:1px solid #2ecc71;border-radius:8px;
                                padding:16px;height:100%;">
                    <div style="color:#2ecc71;font-weight:700;font-size:0.8rem;
                                text-transform:uppercase;letter-spacing:0.08em;
                                margin-bottom:8px;">🐂 Bull Case</div>
                    <div style="color:#c9d1d9;font-size:0.9rem;line-height:1.7;
                                white-space:pre-wrap;word-wrap:break-word;">{sections['bull_case']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with cc:
            if sections.get("bear_case"):
                st.markdown(
                    f"""
                    <div style="background:#3b0a0a;border:1px solid #e74c3c;border-radius:8px;
                                padding:16px;height:100%;">
                    <div style="color:#e74c3c;font-weight:700;font-size:0.8rem;
                                text-transform:uppercase;letter-spacing:0.08em;
                                margin-bottom:8px;">🐻 Bear Case</div>
                    <div style="color:#c9d1d9;font-size:0.9rem;line-height:1.7;
                                white-space:pre-wrap;word-wrap:break-word;">{sections['bear_case']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Remaining sections in expanders
        section_display = [
            ("fundamental",  "📊 Fundamental Analysis"),
            ("technical",    "📈 Technical Analysis"),
            ("earnings",     "💰 Earnings Analysis"),
            ("management",   "🗣️ Management Highlights"),
            ("news_media",   "📰 News Media Sentiment"),
            ("social",       "💬 Social Media Sentiment"),
            ("smart_money",  "🐋 Smart Money Activity"),
            ("analyst",      "🎯 Analyst View"),
            ("peers",        "🏁 Peer Standing"),
            ("patterns",     "🔍 Historical Patterns"),
            ("risks",        "⚠️ Key Risks"),
            ("confidence",   "🎯 Confidence"),
        ]
        for key, title in section_display:
            if sections.get(key):
                with st.expander(title, expanded=False):
                    st.markdown(sections[key])

    if export_pdf_btn:
        with st.spinner("Exporting PDF…"):
            chart_path = tech.get("chart_path", "")
            radar_path = generate_radar_chart(active_ticker, {
                "Fundamental": fund_score, "Technical": tech_score,
                "News": news_score,        "Analyst":   anl_score,
                "Peers": peer_score,       "Patterns":  pat_score,
            })
            sections_for_pdf = {
                "Fundamental Analysis": fund.get("summary", ""),
                "Technical Analysis":   tech.get("summary", ""),
                "Earnings Analysis":    earnings.get("summary", ""),
                "News & Sentiment":     news.get("summary", ""),
                "Historical Patterns":  pats.get("summary", ""),
                "Analyst Data":         anl.get("summary", ""),
                "Peer Comparison":      peers.get("summary", ""),
                "Claude Recommendation": advice.get("narrative", ""),
            }
            charts = [p for p in [chart_path, radar_path] if p]
            pdf_path = export_pdf(ticker_input, sections_for_pdf, charts)
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF Report",
                        data=f,
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf",
                    )


# ── Fundamental tab ───────────────────────────────────────────────────────────

with tab_fund:
    st.subheader("Fundamental Analysis")
    if "error" in fund:
        st.error(fund["error"])
    else:
        c1, c2, c3 = st.columns(3)
        if itype == "etf":
            c1.metric("Expense Ratio",  fund.get("expense_ratio_pct", "N/A"))
            c1.metric("Total Assets",   fund.get("total_assets_fmt",  "N/A"))
            c2.metric("Dividend Yield", fund.get("yield_pct",         "N/A"))
            c2.metric("YTD Return",     fund.get("ytd_return_pct",    "N/A"))
            c3.metric("3-Year Return",  fund.get("three_yr_return_pct","N/A"))
            c3.metric("Beta",           str(fund.get("beta",           "N/A")))
        else:
            c1.metric("Current Price",  f"${fund.get('current_price', 0):,.2f}" if fund.get("current_price") else "N/A")
            c1.metric("P/E Ratio",      str(fund.get("pe_ratio",       "N/A")))
            c2.metric("EPS (TTM)",      str(fund.get("eps",            "N/A")))
            c2.metric("Revenue Growth", f"{fund['revenue_growth']*100:.1f}%" if fund.get("revenue_growth") is not None else "N/A")
            c3.metric("Profit Margin",  f"{fund['profit_margin']*100:.1f}%" if fund.get("profit_margin") is not None else "N/A")
            c3.metric("Debt/Equity",    str(fund.get("debt_to_equity", "N/A")))

        st.metric("Fundamental Score", f"{fund_score} / 10")
        with st.expander("Full report"):
            st.text(fund.get("summary", ""))


# ── Technical tab ─────────────────────────────────────────────────────────────

with tab_tech:
    st.subheader("Technical Analysis")
    if "error" in tech:
        st.error(tech["error"])
    else:
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        hist = tech.get("hist")
        if hist is not None and not hist.empty:
            # Strip timezone
            hist = hist.copy()
            hist.index = hist.index.tz_localize(None)

            # Timeframe selector
            tf = st.radio(
                "Timeframe", ["1M", "3M", "6M", "1Y", "All"],
                index=2, horizontal=True, key="tech_tf",
            )
            last_date = hist.index[-1]
            offsets = {"1M": pd.DateOffset(months=1), "3M": pd.DateOffset(months=3),
                       "6M": pd.DateOffset(months=6), "1Y": pd.DateOffset(years=1)}
            if tf in offsets:
                hist = hist[hist.index >= last_date - offsets[tf]]

            close  = hist["Close"]
            ma50s  = close.rolling(50).mean()
            ma200s = close.rolling(200).mean()

            # RSI series
            delta   = close.diff()
            gain    = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
            loss    = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
            rsi_s   = 100 - 100 / (1 + gain / loss.replace(0, float("nan")))

            # MACD series
            ema12   = close.ewm(span=12).mean()
            ema26   = close.ewm(span=26).mean()
            macd_s  = ema12 - ema26
            sig_s   = macd_s.ewm(span=9).mean()
            hist_s  = macd_s - sig_s

            up_color   = "#26a69a"
            down_color = "#ef5350"
            bar_colors = [up_color if c >= o else down_color
                          for c, o in zip(hist["Close"], hist["Open"])]

            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                row_heights=[0.5, 0.17, 0.17, 0.16],
                vertical_spacing=0.02,
                subplot_titles=(f"{active_ticker} Price", "Volume", "RSI (14)", "MACD"),
            )

            # ── Row 1: Candlestick + MAs ──
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist["Open"], high=hist["High"],
                low=hist["Low"],   close=hist["Close"],
                name="Price",
                increasing_line_color=up_color,
                decreasing_line_color=down_color,
                increasing_fillcolor=up_color,
                decreasing_fillcolor=down_color,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=hist.index, y=ma50s, name="MA 50",
                line=dict(color="#ffb300", width=1.5, dash="dot"),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=hist.index, y=ma200s, name="MA 200",
                line=dict(color="#ef5350", width=1.5, dash="dot"),
            ), row=1, col=1)

            # ── Row 2: Volume ──
            fig.add_trace(go.Bar(
                x=hist.index, y=hist["Volume"],
                name="Volume", marker_color=bar_colors, showlegend=False,
            ), row=2, col=1)

            # ── Row 3: RSI ──
            fig.add_trace(go.Scatter(
                x=hist.index, y=rsi_s, name="RSI",
                line=dict(color="#ce93d8", width=1.5),
            ), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", line_width=1, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", line_width=1, row=3, col=1)

            # ── Row 4: MACD ──
            macd_bar_colors = [up_color if v >= 0 else down_color for v in hist_s]
            fig.add_trace(go.Bar(
                x=hist.index, y=hist_s, name="MACD Hist",
                marker_color=macd_bar_colors, showlegend=False,
            ), row=4, col=1)
            fig.add_trace(go.Scatter(
                x=hist.index, y=macd_s, name="MACD",
                line=dict(color="#4fc3f7", width=1.5),
            ), row=4, col=1)
            fig.add_trace(go.Scatter(
                x=hist.index, y=sig_s, name="Signal",
                line=dict(color="#ffb300", width=1.5),
            ), row=4, col=1)

            fig.update_layout(
                height=750,
                paper_bgcolor="#0f1117",
                plot_bgcolor="#0f1117",
                font=dict(color="#cccccc"),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.02, x=0,
                            bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
                xaxis_rangeslider_visible=False,
                margin=dict(l=0, r=0, t=30, b=0),
            )
            for i in range(1, 5):
                fig.update_xaxes(gridcolor="#1e2130", row=i, col=1)
                fig.update_yaxes(gridcolor="#1e2130", row=i, col=1)
            fig.update_yaxes(tickprefix="$", row=1, col=1)
            fig.update_yaxes(title_text="Vol",  row=2, col=1,
                             tickformat=".2s")
            fig.update_yaxes(title_text="RSI",  row=3, col=1, range=[0, 100])
            fig.update_yaxes(title_text="MACD", row=4, col=1)

            st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RSI (14)",   f"{tech['rsi']:.1f}" if tech.get("rsi") is not None else "N/A")
        c2.metric("MACD",       f"{tech['macd']:.3f}" if tech.get("macd") is not None else "N/A")
        c3.metric("MA 50",      f"${tech['ma50']:.2f}" if tech.get("ma50") else "N/A")
        c4.metric("MA 200",     f"${tech['ma200']:.2f}" if tech.get("ma200") else "N/A")
        st.metric("Technical Score", f"{tech_score} / 10")
        with st.expander("Full report"):
            st.text(tech.get("summary", ""))


# ── Earnings tab ──────────────────────────────────────────────────────────────

with tab_earn:
    st.subheader("Earnings Analysis")
    if "error" in earnings:
        st.error(earnings["error"])
    else:
        fwd = earnings.get("forward_data", {}) or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Next Earnings",  earnings.get("next_earnings_date", "N/A"))
        c2.metric("Forward EPS",    str(fwd.get("forward_eps", "N/A")))
        c3.metric("Forward P/E",    f"{fwd['forward_pe']:.1f}" if fwd.get("forward_pe") else "N/A")
        c4.metric("EPS Growth Fwd", f"{fwd['eps_growth_fwd']*100:+.1f}%" if fwd.get("eps_growth_fwd") is not None else "N/A")

        # Quarterly results table
        quarters = earnings.get("quarterly_results", [])
        if quarters:
            st.markdown("**Quarterly Trend**")
            import pandas as pd
            rows = []
            for q in quarters:
                rows.append({
                    "Quarter":         q.get("quarter_label", "?"),
                    "Revenue":         f"${q['revenue']/1e9:.2f}B" if q.get("revenue") else "N/A",
                    "Gross Margin":    f"{q['gross_margin']*100:.1f}%" if q.get("gross_margin") is not None else "N/A",
                    "Op Margin":       f"{q['operating_margin']*100:.1f}%" if q.get("operating_margin") is not None else "N/A",
                    "Free Cash Flow":  f"${q['free_cash_flow']/1e9:.2f}B" if q.get("free_cash_flow") else "N/A",
                    "YoY Revenue":     f"{q['revenue_yoy']*100:+.1f}%" if q.get("revenue_yoy") is not None else "N/A",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # EPS surprises
        surprises = earnings.get("earnings_surprises", [])
        if surprises:
            st.markdown("**EPS Surprise History**")
            rows2 = []
            for s in surprises:
                surp = s.get("surprise_pct")
                rxn  = s.get("stock_reaction")
                rows2.append({
                    "Date":       s.get("date") or "?",
                    "Est EPS":    f"${s['eps_estimate']:.2f}" if s.get("eps_estimate") is not None else "N/A",
                    "Act EPS":    f"${s['eps_actual']:.2f}" if s.get("eps_actual") is not None else "N/A",
                    "Surprise":   f"{surp:+.1f}%" if surp is not None else "N/A",
                    "Beat/Miss":  "✅ BEAT" if (surp or 0) > 0 else "❌ MISS",
                    "Stock Rxn":  f"{rxn*100:+.1f}%" if rxn is not None else "N/A",
                })
            st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)

        # Press release excerpt
        press = earnings.get("press_release_excerpt", "")
        if press:
            with st.expander("📄 SEC 8-K Press Release Excerpt (Management Commentary)"):
                st.text(press[:3000])

        st.metric("Earnings Score", f"{earn_score} / 10")


# ── News tab ──────────────────────────────────────────────────────────────────

with tab_news:
    st.subheader("News & Sentiment")
    if "error" in news:
        st.error(news["error"])
    else:
        c1, c2 = st.columns(2)
        c1.metric("Overall Sentiment", news.get("overall_sentiment", "N/A").upper())
        c2.metric("News Score",        f"{news_score} / 10")

        headlines = news.get("headlines", [])
        if headlines:
            st.markdown("**Headlines**")
            for h in headlines[:10]:
                sent = h.get("sentiment", "neutral")
                icon = "🟢" if sent == "positive" else "🔴" if sent == "negative" else "⚪"
                url  = h.get("url", "")
                title = h.get("title", "")
                if url:
                    st.markdown(f"{icon} [{title}]({url})")
                else:
                    st.markdown(f"{icon} {title}")
        with st.expander("Full report"):
            st.text(news.get("summary", ""))


# ── Social tab ────────────────────────────────────────────────────────────────

with tab_social:
    st.subheader("Social Media Sentiment")
    if "error" in social:
        st.error(social["error"])
    else:
        sc = social.get("sentiment_counts", {})
        total = sum(sc.values()) or 1

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Mood",  social.get("overall_sentiment", "N/A").upper())
        c2.metric("Bullish",       f"{sc.get('bullish',0)} ({sc.get('bullish',0)*100//total}%)")
        c3.metric("Bearish",       f"{sc.get('bearish',0)} ({sc.get('bearish',0)*100//total}%)")
        c4.metric("Social Score",  f"{social_score} / 10")

        st.caption(f"StockTwits: {social.get('stocktwits_count',0)} posts  ·  Reddit: {social.get('reddit_count',0)} posts")

        posts = social.get("top_posts", [])
        if posts:
            st.markdown("**Sample Posts**")
            for p in posts:
                sent = p.get("sentiment", "neutral")
                icon = "🟢" if sent == "bullish" else "🔴" if sent == "bearish" else "⚪"
                src  = p.get("source", "")
                age  = f" · {p['age_days']}d ago" if p.get("age_days") is not None else ""
                st.markdown(f"{icon} **[{src}]**{age}  {p.get('text','')[:100]}")

        with st.expander("Full report"):
            st.text(social.get("summary", ""))


# ── Smart Money tab ───────────────────────────────────────────────────────────

with tab_smart:
    st.subheader("Smart Money & Institutional Activity")
    if "error" in smart:
        st.error(smart["error"])
    else:
        sd = smart.get("short_data", {}) or {}
        opts = smart.get("options_flow", {}) or {}

        c1, c2, c3, c4 = st.columns(4)
        sc_pct = sd.get("short_pct_float")
        sc_chg = sd.get("short_change")
        c1.metric("Short % Float",    f"{sc_pct*100:.1f}%" if sc_pct else "N/A",
                  delta=f"{sc_chg*100:+.1f}% MoM" if sc_chg is not None else None,
                  delta_color="inverse")
        pcr = opts.get("put_call_ratio")
        c2.metric("Put/Call Ratio",   f"{pcr:.2f}" if pcr else "N/A")
        c3.metric("Upgrades (60d)",   str(smart.get("upgrades", "N/A")))
        c4.metric("Smart Score",      f"{smart_score} / 10")

        insts = smart.get("institutions", [])
        if insts:
            st.markdown("**Top Institutional Holders**")
            import pandas as pd
            rows = [{"Holder": i["holder"],
                     "% Ownership": f"{i['pct_out']*100:.2f}%" if i.get("pct_out") else "N/A"}
                    for i in insts]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if opts.get("unusual_activity"):
            st.markdown("**Unusual Options Activity**")
            for ua in opts["unusual_activity"]:
                st.code(ua)

        with st.expander("Full report"):
            st.text(smart.get("summary", ""))


# ── Analyst tab ───────────────────────────────────────────────────────────────

with tab_analyst:
    st.subheader("Analyst Consensus")
    if "error" in anl:
        st.error(anl["error"])
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Consensus",     str(anl.get("recommendation_key", "N/A")).upper())
        c2.metric("Price Target",  f"${anl['target_mean']:.2f}" if anl.get("target_mean") else "N/A",
                  delta=f"{anl['upside_pct']*100:+.1f}% upside" if anl.get("upside_pct") is not None else None)
        c3.metric("# Analysts",    str(anl.get("num_analysts", "N/A")))
        c4.metric("Next Earnings", str(anl.get("next_earnings_date", "N/A")))
        st.metric("Analyst Score", f"{anl_score} / 10")
        with st.expander("Full report"):
            st.text(anl.get("summary", ""))


# ── Peers tab ─────────────────────────────────────────────────────────────────

with tab_peers:
    st.subheader("Peer Comparison")
    if "error" in peers:
        st.error(peers["error"])
    else:
        peer_data = peers.get("peer_data", [])
        if peer_data:
            import pandas as pd
            rows = []
            for p in peer_data:
                rows.append({
                    "Ticker":          p.get("ticker", "?"),
                    "P/E":             p.get("pe", "N/A"),
                    "Revenue Growth":  p.get("revenue_growth", "N/A"),
                    "Profit Margin":   p.get("profit_margin", "N/A"),
                    "52W Return":      p.get("return_52w", "N/A"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.metric("Peer Score", f"{peer_score} / 10")
        with st.expander("Full report"):
            st.text(peers.get("summary", ""))


# ── Patterns tab ──────────────────────────────────────────────────────────────

with tab_patterns:
    st.subheader("Historical Pattern Analysis")
    if "error" in pats:
        st.error(pats["error"])
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Similar Setups",  str(pats.get("similar_setups_count", 0)))
        c2.metric("Avg 20d Return",  f"{pats['avg_return_20d']*100:+.1f}%" if pats.get("avg_return_20d") is not None else "N/A",
                  delta=f"Win rate {pats['win_rate_20d']*100:.0f}%" if pats.get("win_rate_20d") is not None else None)
        c3.metric("Avg 60d Return",  f"{pats['avg_return_60d']*100:+.1f}%" if pats.get("avg_return_60d") is not None else "N/A",
                  delta=f"Win rate {pats['win_rate_60d']*100:.0f}%" if pats.get("win_rate_60d") is not None else None)
        c4.metric("Pattern Score",   f"{pat_score} / 10")

        named = pats.get("named_patterns", [])
        if named:
            st.markdown("**Detected Patterns**")
            for name, desc in named:
                st.info(f"**{name}** — {desc}")

        with st.expander("Full report"):
            st.text(pats.get("summary", ""))


# ── Radar tab ─────────────────────────────────────────────────────────────────

with tab_radar:
    st.subheader("Score Radar Chart")
    scores = {
        "Fundamental": fund_score,
        "Technical":   tech_score,
        "News":        news_score,
        "Analyst":     anl_score,
        "Peers":       peer_score,
        "Patterns":    pat_score,
    }
    radar_path = generate_radar_chart(active_ticker, scores)
    if radar_path and os.path.exists(radar_path):
        st.image(radar_path, use_container_width=False, width=500)
    else:
        st.warning("Radar chart could not be generated.")


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption("⚠️ This is not financial advice. Always do your own research.")
