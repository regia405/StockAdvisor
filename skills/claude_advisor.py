"""
claude_advisor.py
-----------------
The "Claude skill" — uses the Anthropic API to turn raw analysis numbers
into a detailed, plain-English investment narrative and recommendation.

What it does:
  - Receives the output dicts from fundamental, technical, news, and
    pattern analysis
  - Builds a structured prompt containing all the numeric data
  - Calls Claude (claude-haiku-4-5) and asks for section-by-section
    commentary plus an overall recommendation
  - Returns the narrative text, broken into sections

Why Claude and not more rule-based scoring?
  Rule-based logic is great for computing scores, but poor at explaining
  *context*. A P/E of 31 means different things for a high-growth tech
  company vs a utility. Claude can interpret the combination of metrics
  together and explain what they mean for *this specific company*.

Usage:
    from skills.claude_advisor import generate_recommendation
    narrative = generate_recommendation(
        ticker="AAPL",
        instrument_type="stock",
        instrument_meta={...},
        fundamental=fund_result,
        technical=tech_result,
        news=news_result,
        patterns=pattern_result,
    )
    print(narrative["summary"])
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()


def _build_prompt(ticker, instrument_type, instrument_meta,
                  fundamental, technical, news, patterns,
                  analyst=None, peers=None,
                  social=None, smart_money=None,
                  earnings=None) -> str:
    """Assemble a rich structured prompt from all analysis results."""

    name = instrument_meta.get("display_name", ticker)
    sector = instrument_meta.get("sector", "")
    industry = instrument_meta.get("industry", "")
    category = instrument_meta.get("category", "")
    description = instrument_meta.get("description", "")

    # ---- Fundamental block ----
    if instrument_type == "etf":
        fund_block = f"""
FUNDAMENTAL (ETF) DATA:
  Expense Ratio      : {fundamental.get('expense_ratio_pct', 'N/A')}
  Total Assets (AUM) : {fundamental.get('total_assets_fmt', 'N/A')}
  Dividend Yield     : {fundamental.get('yield_pct', 'N/A')}
  YTD Return         : {fundamental.get('ytd_return_pct', 'N/A')}
  3-Year Ann. Return : {fundamental.get('three_yr_return_pct', 'N/A')}
  5-Year Ann. Return : {fundamental.get('five_yr_return_pct', 'N/A')}
  Beta               : {fundamental.get('beta', 'N/A')}
  Category           : {category}
  Fundamental Score  : {fundamental.get('fundamental_score', 'N/A')} / 10
"""
    else:
        fund_block = f"""
FUNDAMENTAL DATA:
  Current Price   : {fundamental.get('current_price', 'N/A')}
  P/E Ratio (TTM) : {fundamental.get('pe_ratio', 'N/A')}
  EPS (TTM)       : {fundamental.get('eps', 'N/A')}
  Revenue Growth  : {fundamental.get('revenue_growth_pct', 'N/A')}
  Debt / Equity   : {fundamental.get('debt_to_equity', 'N/A')}
  Profit Margin   : {fundamental.get('profit_margin_pct', 'N/A')}
  Fundamental Score: {fundamental.get('fundamental_score', 'N/A')} / 10
"""

    # ---- Technical block ----
    tech_block = f"""
TECHNICAL DATA:
  Current Price : {technical.get('current_price', 'N/A')}
  RSI (14)      : {technical.get('rsi', 'N/A')}
  MACD          : {technical.get('macd', 'N/A')}
  Signal Line   : {technical.get('signal', 'N/A')}
  MA50          : {technical.get('ma50', 'N/A')}
  MA200         : {technical.get('ma200', 'N/A')}
  Volatility    : {technical.get('volatility_pct', 'N/A')}
  Technical Score: {technical.get('technical_score', 'N/A')} / 10
"""

    # ---- News block ----
    if "error" not in news:
        headlines_text = ""
        for h in news.get("headlines", [])[:3]:
            headlines_text += f"  - [{h.get('sentiment','?').upper()}] {h.get('title','')}\n"
        news_block = f"""
NEWS & SENTIMENT:
  Headlines Found  : {news.get('headlines', []).__len__()}
  Sentiment Counts : {news.get('sentiment_counts', {})}
  Overall Sentiment: {news.get('overall_sentiment', 'N/A')}
  News Score       : {news.get('news_score', 'N/A')} / 10
  Top Headlines:
{headlines_text}"""
    else:
        news_block = "\nNEWS & SENTIMENT: Unavailable\n"

    # ---- Pattern block ----
    if "error" not in patterns:
        named = patterns.get("named_patterns", [])
        named_text = "\n".join(f"  * {n}: {d}" for n, d in named) if named else "  None detected"
        fp = patterns.get("current_fingerprint", {})
        pattern_block = f"""
HISTORICAL PATTERNS:
  Fingerprint : RSI={fp.get('rsi_zone','?').upper()}, MA={fp.get('ma_state','?').upper()}, Trend={fp.get('trend','?').upper()}, Vol={fp.get('vol_regime','?').upper()}
  Similar past setups : {patterns.get('similar_setups_count', 0)}
  Avg 20-day return   : {f"{patterns['avg_return_20d']*100:+.1f}%" if patterns.get('avg_return_20d') is not None else 'N/A'}
  Win rate (20d)      : {f"{patterns['win_rate_20d']*100:.0f}%" if patterns.get('win_rate_20d') is not None else 'N/A'}
  Avg 60-day return   : {f"{patterns['avg_return_60d']*100:+.1f}%" if patterns.get('avg_return_60d') is not None else 'N/A'}
  Win rate (60d)      : {f"{patterns['win_rate_60d']*100:.0f}%" if patterns.get('win_rate_60d') is not None else 'N/A'}
Named Patterns:
{named_text}
"""
    else:
        pattern_block = "\nHISTORICAL PATTERNS: Unavailable\n"

    # ---- Analyst block ----
    if analyst and "error" not in analyst:
        analyst_block = f"""
ANALYST & MARKET DATA:
  Analyst Coverage  : {analyst.get('num_analysts', 'N/A')} analysts
  Consensus         : {str(analyst.get('recommendation_key', 'N/A')).upper()} (mean: {analyst.get('recommendation_mean', 'N/A')} / 5.0)
  Price Target Mean : {f"${analyst['target_mean']:.2f}" if analyst.get('target_mean') else 'N/A'}
  Target Range      : {f"${analyst['target_low']:.2f} - ${analyst['target_high']:.2f}" if analyst.get('target_low') else 'N/A'}
  Upside to Target  : {f"{analyst['upside_pct']*100:+.1f}%" if analyst.get('upside_pct') is not None else 'N/A'}
  Next Earnings     : {analyst.get('next_earnings_date', 'N/A')}
  Insider Activity  : {analyst.get('insider_net_sentiment', 'N/A').upper()}
  Analyst Score     : {analyst.get('analyst_score', 'N/A')} / 10
"""
    else:
        analyst_block = "\nANALYST DATA: Unavailable\n"

    # ---- Peer block ----
    if peers and "error" not in peers:
        peer_rows = ""
        for p in peers.get("peer_data", [])[:5]:
            peer_rows += f"  {p.get('ticker','?'):<6} PE={p.get('pe','N/A'):<7} Growth={p.get('revenue_growth','N/A'):<8} Margin={p.get('profit_margin','N/A')}\n"
        peers_block = f"""
PEER COMPARISON:
  Peers: {', '.join(peers.get('peers_analyzed', []))}
  Peer Score : {peers.get('peer_score', 'N/A')} / 10
{peer_rows}"""
    else:
        peers_block = "\nPEER COMPARISON: Unavailable\n"

    # ---- Social sentiment block ----
    if social and "error" not in social:
        sc = social.get("sentiment_counts", {})
        total = sum(sc.values()) or 1
        social_block = f"""
SOCIAL MEDIA SENTIMENT (retail traders / investors):
  Sources      : StockTwits + Reddit (WSB, r/stocks, r/investing)
  Posts Found  : {social.get('stocktwits_count',0)} StockTwits + {social.get('reddit_count',0)} Reddit
  Bullish      : {sc.get('bullish',0)} ({sc.get('bullish',0)*100//total}%)
  Neutral      : {sc.get('neutral',0)} ({sc.get('neutral',0)*100//total}%)
  Bearish      : {sc.get('bearish',0)} ({sc.get('bearish',0)*100//total}%)
  Overall Mood : {social.get('overall_sentiment','N/A').upper()}
  Social Score : {social.get('social_score','N/A')} / 10
"""
    else:
        social_block = "\nSOCIAL MEDIA SENTIMENT: Unavailable\n"

    # ---- Earnings block ----
    if earnings and "error" not in earnings:
        fwd  = earnings.get("forward_data", {}) or {}
        qrs  = earnings.get("quarterly_results", []) or []
        surp = earnings.get("earnings_surprises", []) or []
        beats = sum(1 for s in surp if (s.get("surprise_pct") or 0) > 0)
        next_earn = earnings.get("next_earnings_date", "N/A")
        q0 = qrs[0] if qrs else {}
        press_excerpt = (earnings.get("press_release_excerpt") or "")[:600]

        earnings_block = f"""
EARNINGS ANALYSIS:
  Next Earnings Date  : {next_earn}
  Forward EPS         : {fwd.get('forward_eps', 'N/A')}  (Forward PE: {fwd.get('forward_pe', 'N/A')})
  EPS Growth (fwd)    : {f"{fwd['eps_growth_fwd']*100:+.1f}%" if fwd.get('eps_growth_fwd') is not None else 'N/A'}
  EPS Beat Rate       : {beats}/{len(surp)} last quarters
  Latest Quarter      : Revenue {q0.get('revenue', 'N/A')}  Gross Margin {f"{q0['gross_margin']*100:.1f}%" if q0.get('gross_margin') is not None else 'N/A'}  FCF {q0.get('free_cash_flow', 'N/A')}
  Earnings Score      : {earnings.get('earnings_score', 'N/A')} / 10
  Management Highlights (from latest 8-K press release):
{press_excerpt if press_excerpt else '  (Not available)'}
"""
    else:
        earnings_block = "\nEARNINGS ANALYSIS: Unavailable\n"

    # ---- Smart money block ----
    if smart_money and "error" not in smart_money:
        sd = smart_money.get("short_data", {})
        opts = smart_money.get("options_flow", {}) or {}
        sc_pct = sd.get("short_pct_float")
        sc_chg = sd.get("short_change")
        pcr = opts.get("put_call_ratio")
        smart_block = f"""
SMART MONEY & MARKET ACTIVITY:
  Short % of Float  : {f"{sc_pct*100:.1f}%" if sc_pct else "N/A"}
  Short Change (MoM): {f"{sc_chg*100:+.1f}%" if sc_chg is not None else "N/A"}
  Put/Call Ratio    : {pcr if pcr else "N/A"}
  Analyst Upgrades  : {smart_money.get('upgrades','N/A')} upgrades vs {smart_money.get('downgrades','N/A')} downgrades (last 60d)
  Earnings Beats    : {sum(1 for e in smart_money.get('earnings_history',[]) if (e.get('surprise_pct') or 0) > 0)}/{len(smart_money.get('earnings_history',[]))} last quarters
  Smart Money Score : {smart_money.get('smart_score','N/A')} / 10
"""
    else:
        smart_block = "\nSMART MONEY DATA: Unavailable\n"

    prompt = f"""You are an expert financial analyst writing a detailed investment analysis report.

Instrument: {name} ({ticker})
Type: {instrument_type.upper()}
{f"Sector: {sector} / {industry}" if sector else ""}
{f"Description: {description}" if description else ""}
{fund_block}
{tech_block}
{news_block}
{pattern_block}
{analyst_block}
{peers_block}
{social_block}
{smart_block}
{earnings_block}

Write an analysis report with EXACTLY these sections. Be specific — mention actual numbers. Do NOT add disclaimers.

FUNDAMENTAL ANALYSIS:
[2-3 sentences interpreting the fundamental metrics and what they say about valuation and business quality.]

TECHNICAL ANALYSIS:
[2-3 sentences on price action, momentum, and key levels to watch.]

EARNINGS ANALYSIS:
[2-3 sentences on recent earnings quality, beat/miss history, revenue growth trend, and FCF strength.]

MANAGEMENT HIGHLIGHTS:
[1-2 sentences summarising the most important points from the latest earnings press release — what management emphasised about growth, margins, or guidance.]

NEWS & SENTIMENT:
[1-2 sentences on news coverage quality and what to monitor.]

HISTORICAL PATTERNS:
[1-2 sentences on what past similar setups historically led to.]

ANALYST VIEW:
[1-2 sentences on what the analyst community thinks and whether the consensus is compelling.]

PEER STANDING:
[1-2 sentences on how this instrument ranks vs its closest competitors.]

NEWS MEDIA SENTIMENT:
[1-2 sentences on what financial journalists and news outlets are reporting.]

SOCIAL MEDIA SENTIMENT:
[1-2 sentences on what retail traders on StockTwits and Reddit are saying. Is retail sentiment aligned or diverging from the news/analyst view?]

SMART MONEY ACTIVITY:
[1-2 sentences on short interest, options flow, institutional holders, and upgrade momentum. What are professionals actually doing vs saying?]

BULL CASE:
[2-3 specific reasons this could outperform over the next 3-6 months. Name price targets.]

BEAR CASE:
[2-3 specific reasons this could underperform. Name downside levels.]

RECOMMENDATION:
[2-3 sentences with a clear directional view: accumulate / hold / reduce / avoid. Specific price levels for entry, stop-loss, and target. The single most important catalyst to watch.]

KEY RISKS:
[3-4 bullet points of specific, concrete risks right now.]

CONFIDENCE:
[Single line: rate your confidence in this analysis 1-10 and explain in one sentence why (e.g. data completeness, news coverage, analyst coverage).]
"""
    return prompt


def generate_recommendation(ticker: str, instrument_type: str,
                             instrument_meta: dict,
                             fundamental: dict, technical: dict,
                             news: dict, patterns: dict,
                             analyst: dict = None, peers: dict = None,
                             social: dict = None, smart_money: dict = None,
                             earnings: dict = None) -> dict:
    """
    Call the Claude API to generate a detailed narrative analysis.

    Args:
        ticker           – e.g. "AAPL"
        instrument_type  – e.g. "stock", "etf", "crypto"
        instrument_meta  – from instrument_classifier.classify()
        fundamental      – from fundamental_analysis.analyze()
        technical        – from technical_analysis.analyze()
        news             – from news_fetcher.analyze()
        patterns         – from pattern_analysis.analyze()

    Returns:
        dict with keys: narrative (full text), sections (dict of parsed
        sections), summary (formatted string for printing)
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "error": "ANTHROPIC_API_KEY not set. Add it to your .env file.",
            "summary": "Claude Advisor: API key not configured.",
        }

    # Add formatted percentage versions of key fields so the prompt
    # is easier for Claude to interpret.
    fundamental = dict(fundamental)
    if fundamental.get("revenue_growth") is not None:
        fundamental["revenue_growth_pct"] = f"{fundamental['revenue_growth'] * 100:.1f}%"
    if fundamental.get("profit_margin") is not None:
        fundamental["profit_margin_pct"] = f"{fundamental['profit_margin'] * 100:.1f}%"
    if fundamental.get("expense_ratio") is not None:
        fundamental["expense_ratio_pct"] = f"{fundamental['expense_ratio'] * 100:.3f}%"
    if fundamental.get("etf_yield") is not None:
        fundamental["yield_pct"] = f"{fundamental['etf_yield'] * 100:.2f}%"
    if fundamental.get("ytd_return") is not None:
        fundamental["ytd_return_pct"] = f"{fundamental['ytd_return'] * 100:.1f}%"
    if fundamental.get("three_yr_return") is not None:
        fundamental["three_yr_return_pct"] = f"{fundamental['three_yr_return'] * 100:.1f}%"
    if fundamental.get("five_yr_return") is not None:
        fundamental["five_yr_return_pct"] = f"{fundamental['five_yr_return'] * 100:.1f}%"
    if fundamental.get("total_assets") is not None:
        assets = fundamental["total_assets"]
        if assets >= 1e12:
            fundamental["total_assets_fmt"] = f"${assets/1e12:.1f}T"
        elif assets >= 1e9:
            fundamental["total_assets_fmt"] = f"${assets/1e9:.1f}B"
        else:
            fundamental["total_assets_fmt"] = f"${assets/1e6:.0f}M"

    technical = dict(technical)
    if technical.get("volatility") is not None:
        technical["volatility_pct"] = f"{technical['volatility'] * 100:.2f}%"

    prompt = _build_prompt(
        ticker, instrument_type, instrument_meta,
        fundamental, technical, news, patterns,
        analyst=analyst, peers=peers,
        social=social, smart_money=smart_money,
        earnings=earnings,
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2200,
            messages=[{"role": "user", "content": prompt}],
        )
        narrative = message.content[0].text
    except Exception as e:
        return {
            "error": str(e),
            "summary": f"Claude Advisor: API call failed — {e}",
        }

    # Parse sections from the narrative
    sections = _parse_sections(narrative)

    summary_lines = [
        f"Claude AI Analysis -- {ticker}",
        "=" * 40,
        narrative,
        "",
        "This is not financial advice. Always do your own research.",
    ]

    return {
        "ticker": ticker,
        "narrative": narrative,
        "sections": sections,
        "summary": "\n".join(summary_lines),
    }


def generate_portfolio_narrative(portfolio_summary: str, holdings_analyses: list) -> dict:
    """
    Ask Claude to write a portfolio-level narrative based on all holding scores.

    Args:
        portfolio_summary   – text from get_portfolio_status()["summary"]
        holdings_analyses   – list of dicts, each with ticker + scores

    Returns dict with narrative, summary
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set.", "summary": "Claude: API key missing."}

    holdings_text = "\n".join(
        f"  {h['ticker']}: Fund={h.get('fundamental_score','?')}/10, "
        f"Tech={h.get('technical_score','?')}/10, "
        f"Combined={h.get('combined_score','?'):.1f}/10, "
        f"Return={h.get('return_pct','?'):+.1f}%"
        for h in holdings_analyses
    )

    prompt = f"""You are a portfolio analyst. Given this portfolio status and per-holding scores, write a concise portfolio assessment.

PORTFOLIO STATUS:
{portfolio_summary}

HOLDINGS SCORES:
{holdings_text}

Write a 3-paragraph assessment:
1. Overall portfolio health and risk profile
2. Strongest and weakest holdings and what to do about them
3. One specific rebalancing suggestion (what to trim, what to add, or what to hold)

Be direct. Name specific tickers. Do not add disclaimers.
"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        narrative = msg.content[0].text
    except Exception as e:
        return {"error": str(e), "summary": f"Portfolio narrative failed: {e}"}

    return {
        "narrative": narrative,
        "summary": f"Claude Portfolio Assessment\n{'=' * 40}\n{narrative}\n\nThis is not financial advice. Always do your own research.",
    }


def _parse_sections(text: str) -> dict:
    """
    Extract named sections from Claude's response into a dict.
    """
    section_keys = {
        "FUNDAMENTAL ANALYSIS":  "fundamental",
        "TECHNICAL ANALYSIS":    "technical",
        "EARNINGS ANALYSIS":     "earnings",
        "MANAGEMENT HIGHLIGHTS": "management",
        "NEWS & SENTIMENT":      "news",
        "HISTORICAL PATTERNS":   "patterns",
        "ANALYST VIEW":           "analyst",
        "PEER STANDING":          "peers",
        "NEWS MEDIA SENTIMENT":   "news_media",
        "SOCIAL MEDIA SENTIMENT": "social",
        "SMART MONEY ACTIVITY":   "smart_money",
        "BULL CASE":             "bull_case",
        "BEAR CASE":             "bear_case",
        "RECOMMENDATION":        "recommendation",
        "KEY RISKS":             "risks",
        "CONFIDENCE":            "confidence",
    }
    sections = {}
    lines = text.splitlines()
    current_key = None
    current_lines = []

    for line in lines:
        stripped = line.strip().rstrip(":")
        if stripped in section_keys:
            if current_key:
                sections[section_keys[current_key]] = "\n".join(current_lines).strip()
            current_key = stripped
            current_lines = []
        else:
            if current_key:
                current_lines.append(line)

    if current_key:
        sections[section_keys[current_key]] = "\n".join(current_lines).strip()

    return sections


if __name__ == "__main__":
    import sys
    from skills.instrument_classifier import classify
    from skills.fundamental_analysis import analyze as fund_analyze
    from skills.technical_analysis import analyze as tech_analyze
    from skills.news_fetcher import analyze as news_analyze
    from skills.pattern_analysis import analyze as pat_analyze

    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    meta = classify(symbol)
    result = generate_recommendation(
        ticker=symbol,
        instrument_type=meta["type"],
        instrument_meta=meta,
        fundamental=fund_analyze(symbol),
        technical=tech_analyze(symbol),
        news=news_analyze(symbol),
        patterns=pat_analyze(symbol),
    )
    print(result.get("error") or result["summary"])
