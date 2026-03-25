# 📈 Stock Advisor

AI-powered stock analysis tool that combines 9 analysis modules with Claude AI to generate investment recommendations. Available as both a CLI and a Streamlit web app.

> ⚠️ **Not financial advice.** Always do your own research.

## Features

- **Fundamental Analysis** — P/E, EPS, revenue growth, profit margins, debt/equity
- **Technical Analysis** — RSI, MACD, moving averages, chart generation
- **Earnings Analysis** — quarterly trends, EPS surprises, SEC 8-K filings
- **News Sentiment** — headline aggregation and sentiment scoring via NewsAPI
- **Social Sentiment** — StockTwits and Reddit mood tracking
- **Smart Money** — short interest, institutional holders, options flow
- **Analyst Consensus** — Wall Street ratings and price targets
- **Peer Comparison** — sector-relative valuation and performance
- **Historical Patterns** — similar setup matching with win rates
- **Claude AI Recommendation** — narrative synthesis of all signals with bull/bear cases
- **Portfolio Management** — track holdings, risk metrics, correlation, what-if analysis
- **Price Alerts** — set above/below triggers and check against live prices
- **PDF Reports** — exportable reports with radar charts

## Setup

### Prerequisites

- Python 3.10+
- [Anthropic API key](https://console.anthropic.com/) (for Claude AI)
- [NewsAPI key](https://newsapi.org/) (for news sentiment)

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=<your_anthropic_key>
NEWSAPI_KEY=<your_newsapi_key>
```

For Streamlit Cloud deployment, add these as secrets in the dashboard instead.

## Usage

### Web App

```bash
# Full interactive dashboard
streamlit run app.py

# Public-facing report view with progress bar and PDF export
streamlit run report_app.py
```

### CLI

```bash
# Full analysis (all 9 modules + Claude AI)
python main.py analyze AAPL
python main.py analyze AAPL --pdf

# Quick score table for multiple tickers
python main.py watchlist AAPL MSFT TSLA NVDA

# Individual modules
python main.py fundamental AAPL
python main.py technical AAPL
python main.py news AAPL
python main.py earnings AAPL
python main.py patterns AAPL
python main.py analyst AAPL
python main.py peers AAPL
python main.py social AAPL
python main.py smartmoney AAPL

# Portfolio
python main.py portfolio
python main.py portfolio analyze
python main.py portfolio risk
python main.py portfolio correlation
python main.py portfolio chart
python main.py portfolio create
python main.py whatif TSLA 10

# Price alerts
python main.py alerts
python main.py alerts add AAPL below 200
python main.py alerts check
python main.py alerts remove AAPL

# Weekly digest
python main.py digest
```

## Project Structure

```
StockAdvisor/
├── app.py                  # Streamlit dashboard (full interactive)
├── report_app.py           # Streamlit report view (public-facing)
├── main.py                 # CLI entry point
├── requirements.txt
├── skills/
│   ├── instrument_classifier.py   # Stock/ETF identification
│   ├── fundamental_analysis.py    # Valuation & quality metrics
│   ├── technical_analysis.py      # Price & momentum indicators
│   ├── earnings_analysis.py       # Quarterly results & SEC filings
│   ├── news_fetcher.py            # News headlines & sentiment
│   ├── social_sentiment.py        # StockTwits & Reddit sentiment
│   ├── smart_money.py             # Institutional & options activity
│   ├── analyst_data.py            # Wall Street consensus
│   ├── peer_comparison.py         # Sector peer benchmarking
│   ├── pattern_analysis.py        # Historical pattern matching
│   ├── claude_advisor.py          # Claude AI narrative generation
│   ├── portfolio_manager.py       # Portfolio tracking & risk
│   └── report_builder.py         # Radar charts & PDF export
├── data/
│   ├── portfolio.json             # User holdings
│   └── alerts.json                # Price alert configuration
└── .streamlit/
    └── config.toml                # Dark theme & server config
```

## Data Sources

| Source | Used For |
|--------|----------|
| yfinance | Market data, fundamentals, earnings |
| NewsAPI | News headlines |
| SEC EDGAR | 8-K filings, institutional data |
| StockTwits | Social sentiment |
| Reddit | Social sentiment |
| Claude (Haiku) | AI narrative & recommendations |

## Scoring

Each module produces a score from 0–10. The overall score is the average of all 9 module scores, mapped to a signal:

| Score | Signal |
|-------|--------|
| 8–10 | 🟢 STRONG BUY |
| 7–8 | 🟢 BUY |
| 6–7 | 🟡 CAUTIOUS BUY |
| 5–6 | 🟡 HOLD / NEUTRAL |
| 4–5 | 🟠 CAUTIOUS REDUCE |
| 3–4 | 🔴 REDUCE |
| 0–3 | 🔴 AVOID |
