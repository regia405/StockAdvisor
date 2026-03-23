"""
social_sentiment.py
-------------------
Fetches real-time social media sentiment for a stock from retail traders
and investors — distinct from what news media reports.

Sources:
  StockTwits — free public API, messages are user-labeled bullish/bearish
  Reddit     — r/wallstreetbets, r/stocks, r/investing via free JSON API
  Twitter/X  — requires paid API ($100+/mo); not included
  Threads    — no public API; not included

Why this matters:
  News sentiment captures what journalists think.
  Social sentiment captures what actual retail traders/investors are doing
  and feeling in real time — often a leading or contrarian indicator.

Usage:
    from skills.social_sentiment import analyze
    result = analyze("AAPL")
    print(result["summary"])
"""

import time
import requests
from textblob import TextBlob


REDDIT_HEADERS = {"User-Agent": "StockAdvisor/1.0 (personal finance tool)"}
STOCKTWITS_HEADERS = {"User-Agent": "StockAdvisor/1.0"}
REQUEST_TIMEOUT = 10


# ---------------------------------------------------------------------------
# StockTwits
# ---------------------------------------------------------------------------

def _fetch_stocktwits(ticker: str) -> list:
    """
    Pull up to 30 recent messages from StockTwits.
    Messages with user-applied bullish/bearish labels are the most reliable
    signals — TextBlob is used as a fallback for unlabeled messages.
    """
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        resp = requests.get(url, headers=STOCKTWITS_HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return []
        messages = resp.json().get("messages", [])
    except Exception:
        return []

    results = []
    for msg in messages[:30]:
        body = msg.get("body", "")

        # StockTwits users can tag posts bullish/bearish
        labeled = msg.get("entities", {}).get("sentiment", {})
        if labeled and labeled.get("basic"):
            sentiment = labeled["basic"].lower()   # "bullish" or "bearish"
        else:
            # Fall back to TextBlob
            polarity = TextBlob(body).sentiment.polarity
            if polarity > 0.1:
                sentiment = "bullish"
            elif polarity < -0.1:
                sentiment = "bearish"
            else:
                sentiment = "neutral"

        results.append({
            "source":    "StockTwits",
            "text":      body[:120],
            "sentiment": sentiment,
        })

    return results


# ---------------------------------------------------------------------------
# Reddit
# ---------------------------------------------------------------------------

def _fetch_reddit(ticker: str) -> list:
    """
    Search recent posts from r/wallstreetbets, r/stocks, r/investing.
    Only includes posts from the last 7 days. TextBlob scores the title.
    """
    subreddits = [
        ("wallstreetbets", 15),
        ("stocks",         10),
        ("investing",      10),
    ]
    results = []
    now = time.time()

    for sub, limit in subreddits:
        url = (
            f"https://www.reddit.com/r/{sub}/search.json"
            f"?q={ticker}&sort=new&restrict_sr=on&limit={limit}&type=link"
        )
        try:
            resp = requests.get(url, headers=REDDIT_HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                continue
            posts = resp.json().get("data", {}).get("children", [])
        except Exception:
            continue

        for post in posts:
            p = post.get("data", {})
            title = p.get("title", "")
            created = p.get("created_utc", 0)
            age_days = (now - created) / 86400 if created else 999
            if age_days > 7:
                continue

            text = f"{title} {p.get('selftext', '')[:200]}"
            polarity = TextBlob(text).sentiment.polarity
            if polarity > 0.1:
                sentiment = "bullish"
            elif polarity < -0.1:
                sentiment = "bearish"
            else:
                sentiment = "neutral"

            results.append({
                "source":       f"r/{sub}",
                "text":         title[:120],
                "sentiment":    sentiment,
                "score":        p.get("score", 0),         # Reddit upvotes
                "upvote_ratio": p.get("upvote_ratio", 0.5),
                "age_days":     round(age_days, 1),
            })

    return results


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _compute_score(bull_ratio: float, bear_ratio: float,
                   st_labeled: list) -> int:
    """
    0–10 score based on bull/bear ratio, boosted/penalised by the more
    reliable StockTwits labeled signals.
    """
    if bull_ratio > 0.65:
        score = 8
    elif bull_ratio > 0.55:
        score = 7
    elif bull_ratio > 0.45:
        score = 5
    elif bull_ratio > 0.35:
        score = 3
    else:
        score = 1

    # Adjust based on StockTwits labeled (user-applied, more deliberate)
    if st_labeled:
        st_bull = sum(1 for p in st_labeled if p["sentiment"] == "bullish") / len(st_labeled)
        if st_bull > 0.65:
            score = min(10, score + 1)
        elif st_bull < 0.35:
            score = max(0,  score - 1)

    return score


def _score_label(score: int) -> str:
    if score >= 8:
        return "  Signal: VERY BULLISH. Strong positive buzz on social media."
    if score >= 6:
        return "  Signal: BULLISH. More bulls than bears in retail community."
    if score >= 4:
        return "  Signal: MIXED. Retail sentiment is divided."
    if score >= 2:
        return "  Signal: BEARISH. Negative retail sentiment outweighs positive."
    return   "  Signal: VERY BEARISH. Heavy bearish conviction in retail community."


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def analyze(ticker: str) -> dict:
    """
    Fetch and score social media sentiment for a stock.

    Returns a dict with keys:
        ticker, social_score, overall_sentiment, stocktwits_count,
        reddit_count, sentiment_counts, top_posts, summary
    """
    ticker = ticker.upper().strip()

    stocktwits_posts = _fetch_stocktwits(ticker)
    reddit_posts     = _fetch_reddit(ticker)
    all_posts        = stocktwits_posts + reddit_posts

    if not all_posts:
        return {
            "ticker":          ticker,
            "social_score":    5,
            "overall_sentiment": "neutral",
            "stocktwits_count": 0,
            "reddit_count":     0,
            "sentiment_counts": {"bullish": 0, "neutral": 0, "bearish": 0},
            "top_posts":        [],
            "summary": (
                f"Social Media Sentiment -- {ticker}\n"
                f"{'=' * 40}\n"
                f"  No social data found. Feeds may be rate-limited or ticker\n"
                f"  has no recent posts. (Twitter/X requires paid API.)\n\n"
                f"  Social Score      : 5 / 10  (neutral default)\n\n"
                f"This is not financial advice. Always do your own research."
            ),
        }

    # Count sentiments
    counts = {"bullish": 0, "neutral": 0, "bearish": 0}
    for p in all_posts:
        counts[p.get("sentiment", "neutral")] += 1

    total      = len(all_posts)
    bull_ratio = counts["bullish"] / total
    bear_ratio = counts["bearish"] / total

    st_labeled = [p for p in stocktwits_posts if p["sentiment"] != "neutral"]
    score      = _compute_score(bull_ratio, bear_ratio, st_labeled)

    if bull_ratio > 0.55:
        overall = "bullish"
    elif bear_ratio > 0.55:
        overall = "bearish"
    else:
        overall = "mixed"

    # Top posts by Reddit score, then StockTwits order
    reddit_sorted = sorted(
        [p for p in reddit_posts], key=lambda x: x.get("score", 0), reverse=True
    )
    top_posts = (stocktwits_posts[:3] + reddit_sorted[:3])[:6]

    summary_lines = [
        f"Social Media Sentiment -- {ticker}",
        "=" * 40,
        f"  Note: Twitter/X requires paid API; not included.",
        f"  Sources           : StockTwits + Reddit (WSB / r/stocks / r/investing)",
        f"  Posts Analysed    : {total}  (StockTwits: {len(stocktwits_posts)}, Reddit: {len(reddit_posts)})",
        f"  Bullish           : {counts['bullish']} ({bull_ratio*100:.0f}%)",
        f"  Neutral           : {counts['neutral']} ({counts['neutral']/total*100:.0f}%)",
        f"  Bearish           : {counts['bearish']} ({bear_ratio*100:.0f}%)",
        f"  Overall Mood      : {overall.upper()}",
        "",
        f"  Social Score      : {score} / 10",
        "",
        _score_label(score),
        "",
        "  Sample Posts:",
    ]

    for i, p in enumerate(top_posts, 1):
        tag = "[+]" if p["sentiment"] == "bullish" else "[-]" if p["sentiment"] == "bearish" else "[=]"
        age = f" ({p['age_days']}d ago)" if p.get("age_days") is not None else ""
        summary_lines.append(f"    {i}. {tag} [{p['source']}]{age} {p['text'][:65]}")

    summary_lines += ["", "This is not financial advice. Always do your own research."]

    return {
        "ticker":            ticker,
        "social_score":      score,
        "overall_sentiment": overall,
        "stocktwits_count":  len(stocktwits_posts),
        "reddit_count":      len(reddit_posts),
        "sentiment_counts":  counts,
        "top_posts":         top_posts,
        "summary":           "\n".join(summary_lines),
    }


if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    r = analyze(symbol)
    print(r.get("summary"))
