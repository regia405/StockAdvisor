"""
smart_money.py
--------------
Analyses what institutional and professional money is doing — things that
don't make mainstream headlines but are critical to understand before investing.

Covers:
  Short Interest      — % of float shorted, days to cover, month-over-month
                        change. High + rising shorts = bearish pressure.
                        High + falling shorts = short squeeze potential.
  Options Flow        — Put/call ratio from the live options chain.
                        More puts than calls = professionals are hedging.
                        Unusual volume (vol >> open interest) = big players
                        making directional bets.
  Institutional Holders — Top funds and their % ownership concentration.
  Upgrade/Downgrade Momentum — Not just current rating, but whether analysts
                        are upgrading or downgrading over the last 60 days.
  Earnings Surprise History — Consistent beats change how a stock trades
                        around earnings and affect institutional confidence.

All data via yfinance. No API keys required.

Usage:
    from skills.smart_money import analyze
    result = analyze("AAPL")
    print(result["summary"])
"""

import yfinance as yf
import pandas as pd


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _get_short_data(info: dict) -> dict:
    """Extract short interest metrics from yfinance info dict."""
    short_pct    = info.get("shortPercentOfFloat")
    short_ratio  = info.get("shortRatio")          # days to cover
    shares_now   = info.get("sharesShort")
    shares_prior = info.get("sharesShortPriorMonth")

    change = None
    if shares_now and shares_prior and shares_prior > 0:
        change = (shares_now - shares_prior) / shares_prior

    return {
        "short_pct_float": short_pct,
        "short_ratio":     short_ratio,
        "shares_short":    shares_now,
        "short_change":    change,
    }


def _get_options_flow(stock) -> dict | None:
    """
    Compute put/call volume ratio and detect unusual options activity
    (when volume significantly exceeds open interest — a sign of fresh
    institutional positioning).
    """
    try:
        expiries = stock.options
        if not expiries:
            return None

        # Use the nearest expiry — most liquid, most current signal
        chain = stock.option_chain(expiries[0])
        calls = chain.calls.fillna(0)
        puts  = chain.puts.fillna(0)

        call_vol = int(calls["volume"].sum())
        put_vol  = int(puts["volume"].sum())
        pcr      = round(put_vol / call_vol, 2) if call_vol > 0 else None

        # Unusual = volume > 1.5x open interest (fresh money, not existing positions)
        unusual = []
        uc = calls[calls["volume"] > calls["openInterest"] * 1.5].sort_values("volume", ascending=False).head(2)
        up = puts[puts["volume"]  > puts["openInterest"]  * 1.5].sort_values("volume", ascending=False).head(2)
        for _, r in uc.iterrows():
            unusual.append(f"Unusual CALL  ${r['strike']:.0f} strike  vol={int(r['volume']):,}  OI={int(r['openInterest']):,}")
        for _, r in up.iterrows():
            unusual.append(f"Unusual PUT   ${r['strike']:.0f} strike  vol={int(r['volume']):,}  OI={int(r['openInterest']):,}")

        return {
            "put_call_ratio":   pcr,
            "call_volume":      call_vol,
            "put_volume":       put_vol,
            "unusual_activity": unusual[:4],
        }
    except Exception:
        return None


def _get_top_institutions(stock) -> list:
    """Return top 5 institutional holders with % ownership."""
    try:
        df = stock.institutional_holders
        if df is None or df.empty:
            return []
        result = []
        for _, row in df.head(5).iterrows():
            pct = row.get("% Out") or row.get("pctHeld")
            result.append({
                "holder": str(row.get("Holder", "Unknown"))[:40],
                "pct_out": float(pct) if pct is not None else None,
                "shares":  int(row["Shares"]) if row.get("Shares") is not None else None,
            })
        return result
    except Exception:
        return []


def _get_upgrade_downgrade_momentum(stock) -> tuple:
    """
    Count analyst upgrades vs downgrades in the last 60 days.
    Returns (upgrade_count, downgrade_count, recent_actions_list).
    """
    try:
        df = stock.upgrades_downgrades
        if df is None or df.empty:
            return None, None, []

        # Normalise index timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=60)
        recent = df[df.index >= cutoff]

        if recent.empty:
            return 0, 0, []

        UPGRADE_GRADES   = {"buy", "strong buy", "outperform", "overweight",
                            "upgrade", "positive", "accumulate"}
        DOWNGRADE_GRADES = {"sell", "strong sell", "underperform", "underweight",
                            "downgrade", "negative", "reduce", "underperform"}

        ups   = 0
        downs = 0
        recents = []

        for date_idx, row in recent.iterrows():
            firm   = str(row.get("Firm", "Unknown"))
            action = str(row.get("Action", row.get("ToGrade", "?"))).lower()
            grade  = str(row.get("ToGrade", "")).lower()

            if action in UPGRADE_GRADES or grade in UPGRADE_GRADES:
                ups += 1
            elif action in DOWNGRADE_GRADES or grade in DOWNGRADE_GRADES:
                downs += 1

            recents.append(
                f"{date_idx.strftime('%Y-%m-%d')}  {firm[:25]:<25}  {row.get('ToGrade', row.get('Action', '?'))}"
            )

        return ups, downs, recents[:5]

    except Exception:
        return None, None, []


def _get_earnings_surprises(stock) -> list:
    """
    Return the last 4 quarters of EPS surprise data.
    Consistent beats = institutional confidence; misses = selling pressure.
    """
    try:
        # Try earnings_history first, fall back to earnings
        df = getattr(stock, "earnings_history", None)
        if df is None or (hasattr(df, "empty") and df.empty):
            return []

        results = []
        for _, row in df.tail(4).iterrows():
            est  = row.get("epsEstimate") or row.get("EPS Estimate")
            act  = row.get("epsActual")   or row.get("Reported EPS")
            surp = row.get("surprisePercent")
            if est is not None and act is not None:
                results.append({
                    "eps_estimate": float(est),
                    "eps_actual":   float(act),
                    "surprise_pct": float(surp) if surp is not None else None,
                })
        return results
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_short(short_pct, short_change) -> int:
    """
    Low shorts = less overhead selling pressure = bullish.
    Declining shorts = potential short squeeze = very bullish.
    """
    if short_pct is None:
        return 1
    if short_pct < 0.03:
        base = 2
    elif short_pct < 0.08:
        base = 1
    else:
        base = 0

    # Declining short interest = potential squeeze, add a point
    if short_change is not None and short_change < -0.10:
        base = min(2, base + 1)
    return base


def _score_options(options) -> int:
    """PCR < 0.7 = call-heavy = bullish; PCR > 1.0 = put-heavy = bearish."""
    if not options:
        return 1
    pcr = options.get("put_call_ratio")
    if pcr is None:
        return 1
    if pcr < 0.7:
        return 2
    if pcr < 1.0:
        return 1
    return 0


def _score_institutions(institutions) -> int:
    """High institutional concentration = confidence from big money."""
    if not institutions:
        return 1
    total_pct = sum((i.get("pct_out") or 0) for i in institutions)
    if total_pct > 0.20:
        return 2
    if total_pct > 0.05:
        return 1
    return 0


def _score_upgrades(ups, downs) -> int:
    if ups is None:
        return 1
    if ups > downs:
        return 2
    if ups == downs:
        return 1
    return 0


def _score_earnings(history) -> int:
    if not history:
        return 1
    beats = sum(1 for e in history if (e.get("surprise_pct") or 0) > 0)
    if beats >= 3:
        return 2
    if beats >= 2:
        return 1
    return 0


def _score_label(score: int) -> str:
    if score >= 8:
        return "  Signal: VERY BULLISH. Smart money is accumulating — strong institutional conviction."
    if score >= 6:
        return "  Signal: BULLISH. Options flow and institutional data favour the upside."
    if score >= 4:
        return "  Signal: MIXED. Conflicting signals from professional investors."
    if score >= 2:
        return "  Signal: CAUTIOUS. Short interest and options suggest hedging/exits."
    return   "  Signal: BEARISH. Smart money appears to be reducing or shorting."


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def analyze(ticker: str) -> dict:
    """
    Analyse institutional / smart-money activity for a stock.

    Returns dict with keys:
        ticker, short_data, options_flow, institutions,
        upgrades, downgrades, earnings_history, smart_score, summary
    """
    ticker = ticker.upper().strip()

    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
    except Exception:
        return {"ticker": ticker, "error": f"Could not fetch data for '{ticker}'."}

    short        = _get_short_data(info)
    options      = _get_options_flow(stock)
    institutions = _get_top_institutions(stock)
    ups, downs, grade_list = _get_upgrade_downgrade_momentum(stock)
    earnings     = _get_earnings_surprises(stock)

    smart_score = (
        _score_short(short.get("short_pct_float"), short.get("short_change"))
        + _score_options(options)
        + _score_institutions(institutions)
        + _score_upgrades(ups, downs)
        + _score_earnings(earnings)
    )

    # --- Build summary ---
    def fmt(v, pct=False, decimals=2):
        if v is None:
            return "N/A"
        if pct:
            return f"{v * 100:.{decimals}f}%"
        return f"{v:,.{decimals}f}"

    sc = short.get("short_change")
    if sc is not None and sc > 0.10:
        chg_note = " (shorts INCREASING -- bearish pressure building)"
    elif sc is not None and sc < -0.10:
        chg_note = " (shorts DECREASING -- potential squeeze setup)"
    else:
        chg_note = ""

    lines = [
        f"Smart Money & Market Activity -- {ticker}",
        "=" * 40,
        "",
        "  SHORT INTEREST:",
        f"    Short % of Float  : {fmt(short.get('short_pct_float'), pct=True)}",
        f"    Days to Cover     : {fmt(short.get('short_ratio'), decimals=1)}",
        f"    MoM Change        : {fmt(sc, pct=True, decimals=1)}{chg_note}",
    ]

    if options:
        pcr = options.get("put_call_ratio")
        if pcr is not None:
            if pcr < 0.7:
                flow_note = "call-heavy = bullish options positioning"
            elif pcr > 1.0:
                flow_note = "put-heavy = professionals are hedging / bearish"
            else:
                flow_note = "balanced flow"
        else:
            flow_note = ""
        lines += [
            "",
            "  OPTIONS FLOW:",
            f"    Put/Call Ratio    : {fmt(pcr)}  ({flow_note})",
            f"    Call Volume       : {options.get('call_volume', 0):,}",
            f"    Put Volume        : {options.get('put_volume', 0):,}",
        ]
        if options.get("unusual_activity"):
            lines.append("    Unusual Activity  (volume >> open interest = fresh big-money bets):")
            for ua in options["unusual_activity"]:
                lines.append(f"      * {ua}")

    if institutions:
        lines += ["", "  TOP INSTITUTIONAL HOLDERS:"]
        for inst in institutions:
            pct_str = f"{inst['pct_out']*100:.2f}%" if inst.get("pct_out") else "N/A"
            lines.append(f"    {inst['holder']:<40}  {pct_str:>6}")

    if ups is not None:
        net = ups - (downs or 0)
        net_label = "net UPGRADES -- momentum improving" if net > 0 else "net DOWNGRADES -- momentum deteriorating" if net < 0 else "flat"
        lines += [
            "",
            "  ANALYST MOMENTUM (last 60 days):",
            f"    Upgrades          : {ups}",
            f"    Downgrades        : {downs}",
            f"    Net               : {net:+d}  ({net_label})",
        ]
        if grade_list:
            lines.append("    Recent actions:")
            for g in grade_list[:4]:
                lines.append(f"      {g}")

    if earnings:
        beats = sum(1 for e in earnings if (e.get("surprise_pct") or 0) > 0)
        lines += ["", f"  EARNINGS SURPRISE HISTORY  ({beats}/{len(earnings)} beats):"]
        for e in earnings:
            surp = e.get("surprise_pct")
            if surp is not None:
                tag = "BEAT" if surp > 0 else "MISS"
                lines.append(
                    f"    {tag}  Est ${e['eps_estimate']:.2f}  "
                    f"Act ${e['eps_actual']:.2f}  ({surp:+.1f}%)"
                )

    lines += [
        "",
        f"  Smart Money Score   : {smart_score} / 10",
        "",
        _score_label(smart_score),
        "",
        "This is not financial advice. Always do your own research.",
    ]

    return {
        "ticker":           ticker,
        "short_data":       short,
        "options_flow":     options,
        "institutions":     institutions,
        "upgrades":         ups,
        "downgrades":       downs,
        "earnings_history": earnings,
        "smart_score":      smart_score,
        "summary":          "\n".join(lines),
    }


if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    r = analyze(symbol)
    print(r.get("error") or r["summary"])
