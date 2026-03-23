"""
news_fetcher.py
---------------
Fetches recent news headlines for a stock ticker and analyzes sentiment.
Uses free RSS feeds and TextBlob sentiment analysis.

Produces:
  - Recent headlines (from financial news RSS feeds)
  - Sentiment score per headline (negative, neutral, positive)
  - Overall sentiment bias (bullish/bearish/neutral)
  - A composite news score out of 10

Usage:
    from skills.news_fetcher import analyze
    result = analyze("AAPL")
    print(result["summary"])
"""

import feedparser
from textblob import TextBlob
import re
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------

def _analyze_sentiment(text):
    """
    Analyze sentiment of text using TextBlob polarity (-1 to 1).
    Returns sentiment label and polarity score.
    """
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1.0 (negative) to 1.0 (positive)
        
        if polarity > 0.1:
            return "positive", polarity
        elif polarity < -0.1:
            return "negative", polarity
        else:
            return "neutral", polarity
    except Exception:
        return "neutral", 0


def _score_news_sentiment(headlines_with_sentiment):
    """
    Score overall sentiment from headlines.
    
    Scoring based on proportion of positive headlines:
      >60% positive  → 2 pts  (bullish)
      30-60% positive → 1 pt   (mixed)
      <30% positive  → 0 pts  (bearish)
    """
    if not headlines_with_sentiment:
        return 1  # neutral default
    
    try:
        sentiments = [s for _, s, _ in headlines_with_sentiment]
        positive_count = sum(1 for s in sentiments if s == "positive")
        total = len(sentiments)
        positive_ratio = positive_count / total if total > 0 else 0
        
        if positive_ratio > 0.6:
            return 2
        elif positive_ratio >= 0.3:
            return 1
        else:
            return 0
    except Exception:
        return 1


def _score_recency(pub_date_str):
    """
    Score based on how recent the news is.
    More recent = fresher signal.
    
    Returns points 0-2 based on age.
    """
    try:
        # Parse various date formats
        if not pub_date_str:
            return 0
        
        pub_date = feedparser._parse_date(pub_date_str)
        if not pub_date:
            return 0
        
        now = datetime.utcnow()
        age_hours = (now - datetime(*pub_date[:6])).total_seconds() / 3600
        
        if age_hours <= 24:
            return 2
        elif age_hours <= 72:
            return 1
        else:
            return 0
    except Exception:
        return 0


def _score_volume(headlines_count):
    """
    Score based on volume of headlines (more coverage = more attention).
    
    Scoring:
      ≥5 headlines  → 2 pts  (lots of coverage)
      3-4 headlines → 1 pt   (moderate coverage)
      <3 headlines  → 0 pts  (light coverage)
    """
    if headlines_count >= 5:
        return 2
    elif headlines_count >= 3:
        return 1
    else:
        return 0


# ---------------------------------------------------------------------------
# News fetching
# ---------------------------------------------------------------------------

def _fetch_news_for_ticker(ticker):
    """
    Fetch recent news headlines for a ticker from financial RSS feeds.
    Returns list of (headline, sentiment, polarity, source, date) tuples.
    """
    headlines = []
    
    # Financial news RSS feeds that cover stocks
    feeds = [
        f"https://feeds.bloomberg.com/markets/news.rss?ticker={ticker}",
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}",
        "https://feeds.cnbc.com/cnbcnews/",
        "https://feeds.reuters.com/finance/markets",
    ]
    
    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            
            # Extract entries from the feed
            for entry in feed.entries[:5]:  # limit to 5 per feed
                title = entry.get("title", "")
                
                # Filter for relevant entries (contain ticker or financial keywords)
                if title:
                    summary = entry.get("summary", "") or ""
                    content = f"{title} {summary}"
                    
                    # Basic ticker relevance check
                    if ticker.lower() in content.lower() or _is_market_related(content):
                        sentiment, polarity = _analyze_sentiment(title)
                        pub_date = entry.get("published", "")
                        source = feed.feed.get("title", "Financial News")
                        
                        headlines.append({
                            "title": title,
                            "sentiment": sentiment,
                            "polarity": polarity,
                            "source": source,
                            "date": pub_date,
                            "url": entry.get("link", ""),
                        })
        except Exception:
            # Feed fetch error, continue to next feed
            continue
    
    return headlines


def _is_market_related(text):
    """Check if text contains market-related keywords."""
    keywords = [
        "stock", "market", "rally", "surge", "rally", "drop", "fall", "crash",
        "earnings", "revenue", "profit", "loss", "analyst", "upgrade", "downgrade",
        "bullish", "bearish", "trading", "price", "shares", "dividend",
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def analyze(ticker: str) -> dict:
    """
    Fetch and analyze financial news sentiment for a stock ticker.

    Args:
        ticker: Stock symbol string, e.g. "AAPL", "MSFT", "TSLA"

    Returns:
        A dict with keys:
          ticker, headlines, overall_sentiment, news_score, summary
        headlines is a list of dicts with title, sentiment, source, date
    """
    ticker = ticker.upper().strip()

    # --- Fetch news headlines -----------------------------------------------
    headlines = _fetch_news_for_ticker(ticker)
    
    if not headlines:
        # Fallback: try a broader search
        headlines = _fetch_news_for_ticker("market")
        
        if not headlines:
            return {
                "ticker": ticker,
                "error": f"Could not fetch news for '{ticker}'. "
                         "News feeds may be temporarily unavailable.",
            }

    # --- Prepare headlines for scoring
    headlines_with_sentiment = [
        (h["title"], h["sentiment"], h["polarity"])
        for h in headlines
    ]

    # --- Calculate composite score ------------------------------------------
    score = (
        _score_news_sentiment(headlines_with_sentiment)
        + _score_volume(len(headlines))
        + (_score_recency(headlines[0].get("date", "")) if headlines else 0)
    )
    # Score is now 0–6, normalize to 0–10
    score = int((score / 6) * 10) if score > 0 else 5

    # --- Determine overall sentiment ----------------------------------------
    sentiment_counts = {
        "positive": sum(1 for _, s, _ in headlines_with_sentiment if s == "positive"),
        "neutral": sum(1 for _, s, _ in headlines_with_sentiment if s == "neutral"),
        "negative": sum(1 for _, s, _ in headlines_with_sentiment if s == "negative"),
    }
    
    total = len(headlines_with_sentiment)
    if sentiment_counts["positive"] > sentiment_counts["negative"]:
        overall_sentiment = "bullish"
    elif sentiment_counts["negative"] > sentiment_counts["positive"]:
        overall_sentiment = "bearish"
    else:
        overall_sentiment = "neutral"

    # --- Build a plain-English summary --------------------------------------
    summary_lines = [
        f"News & Sentiment Analysis — {ticker}",
        "=" * 40,
        f"  Headlines Found  : {len(headlines)}",
        f"  Positive         : {sentiment_counts['positive']} ({sentiment_counts['positive']*100//total}%)" if total > 0 else "  Positive         : 0 (0%)",
        f"  Neutral          : {sentiment_counts['neutral']} ({sentiment_counts['neutral']*100//total}%)" if total > 0 else "  Neutral          : 0 (0%)",
        f"  Negative         : {sentiment_counts['negative']} ({sentiment_counts['negative']*100//total}%)" if total > 0 else "  Negative         : 0 (0%)",
        f"  Overall Sentiment: {overall_sentiment.upper()}",
        "",
        f"  News Score       : {score} / 10",
        "",
        _score_label(score),
        "",
    ]
    
    # Add top headlines
    if headlines:
        summary_lines.append("  Recent Headlines:")
        for i, h in enumerate(headlines[:3], 1):
            tag = "[+]" if h["sentiment"] == "positive" else "[-]" if h["sentiment"] == "negative" else "[=]"
            title = h["title"][:70] + ("..." if len(h["title"]) > 70 else "")
            url = h.get("url", "")
            summary_lines.append(f"    {i}. {tag} {title}")
            if url:
                summary_lines.append(f"       {url}")
    
    summary_lines.append("")
    summary_lines.append("This is not financial advice. Always do your own research.")
    
    summary = "\n".join(summary_lines)

    return {
        "ticker": ticker,
        "headlines": headlines[:5],
        "sentiment_counts": sentiment_counts,
        "overall_sentiment": overall_sentiment,
        "news_score": score,
        "summary": summary,
    }


def _score_label(score: int) -> str:
    """Return a plain-English interpretation of the news score."""
    if score >= 8:
        return "  Signal: VERY BULLISH. Positive news and strong sentiment."
    if score >= 6:
        return "  Signal: BULLISH. Mostly positive coverage."
    if score >= 4:
        return "  Signal: MIXED. Conflicting news signals."
    if score >= 2:
        return "  Signal: BEARISH. Mostly negative coverage."
    return   "  Signal: VERY BEARISH. Negative sentiment dominates."


# ---------------------------------------------------------------------------
# Quick self-test — run this file directly to try it out
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = analyze(symbol)
    if "error" in result:
        print(result["error"])
    else:
        print(result["summary"])
