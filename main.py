# main.py
from fastapi import FastAPI
import threading
import time
import feedparser
import numpy as np
import re
import urllib.request
import ssl

# -----------------------------
# GLOBAL VARIABLES
# -----------------------------
app = FastAPI()

# Stores: score (-1..1), label, headline count, last updated timestamp, last error
sentiment_data = {
    "score": 0.0,
    "label": "neutral",
    "headlines_analyzed": 0,
    "last_updated": None,
    "last_error": None,
}

# -----------------------------
# FIX 1: Multiple RSS sources with proper browser-like headers
# investing.com blocks bots — use a pool of fallback feeds
# -----------------------------
RSS_FEEDS = [
    # Kitco — dedicated gold/metals news site
    "https://www.kitco.com/rss/kitconews.rss",
    # MarketWatch top stories (covers commodities)
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    # Reuters business (broad financial coverage)
    "https://feeds.reuters.com/reuters/businessNews",
    # Yahoo Finance gold futures (GC=F)
    "https://finance.yahoo.com/rss/headline?s=GC%3DF",
]

# FIX 5: Request headers that mimic a real browser so sites don't block us
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

# SSL context (skip verification for feeds that have cert issues)
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def fetch_feed(url: str) -> list[str]:
    """Fetch RSS feed headlines using proper HTTP headers."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=10) as response:
            raw = response.read()
        feed = feedparser.parse(raw)
        return [entry.title for entry in feed.entries if hasattr(entry, "title")]
    except Exception as e:
        print(f"  [feed error] {url}: {e}")
        return []


# -----------------------------
# SMART SENTIMENT SCORING FUNCTION
# -----------------------------

# FIX 6: Use word-boundary regex so "up" doesn't match inside "support"
BULLISH_PATTERNS = [re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE) for w in [
    "rise", "rises", "rising", "gain", "gains", "soar", "soars", "soaring",
    "bullish", "rally", "rallies", "surge", "surges", "surging", "jump", "jumps",
    "climb", "climbs", "climbing", "advance", "advances", "breakout",
    "gold up", "xauusd up", "higher", "upside", "record high", "strong demand",
]]

BEARISH_PATTERNS = [re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE) for w in [
    "fall", "falls", "falling", "drop", "drops", "dropping", "decline", "declines",
    "declining", "slip", "slips", "slipping", "bearish", "plunge", "plunges",
    "plunging", "sink", "sinks", "sinking", "tumble", "tumbles", "crash",
    "xauusd down", "gold down", "lower", "downside", "pressure", "weak demand",
    "sell-off", "selloff",
]]

NEUTRAL_PATTERNS = [re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE) for w in [
    "neutral", "sideways", "unchanged", "steady", "flat", "consolidat",
]]

# Gold-relevance filter — skip headlines that aren't about gold/XAU
GOLD_KEYWORDS = re.compile(
    r"\b(gold|xau|xauusd|bullion|precious metal|ounce|troy|comex|gc=f)\b",
    re.IGNORECASE,
)


def score_headline(text: str) -> float | None:
    """
    Score a headline. Returns None if not gold-related (so it's skipped).
    +ve = bullish, -ve = bearish, 0 = neutral.
    """
    # Skip headlines not mentioning gold at all
    if not GOLD_KEYWORDS.search(text):
        return None

    score = 0.0
    for pat in BULLISH_PATTERNS:
        if pat.search(text):
            score += 1.0
    for pat in BEARISH_PATTERNS:
        if pat.search(text):
            score -= 1.0
    for pat in NEUTRAL_PATTERNS:
        if pat.search(text):
            score *= 0.5  # dampen extremes

    return score


# -----------------------------
# SENTIMENT UPDATER
# -----------------------------
def update_sentiment():
    """Runs in background thread. Fetches feeds and updates sentiment_data."""
    while True:
        print("[sentiment] Updating...")
        all_headlines: list[str] = []

        # FIX 1: Try all feeds, collect whichever ones work
        for url in RSS_FEEDS:
            headlines = fetch_feed(url)
            print(f"  Got {len(headlines)} headlines from {url}")
            all_headlines.extend(headlines)

        # Deduplicate (same story may appear on multiple feeds)
        all_headlines = list(dict.fromkeys(all_headlines))

        try:
            if all_headlines:
                # Score only gold-relevant headlines; skip None results
                scores = [score_headline(h) for h in all_headlines]
                gold_scores = [s for s in scores if s is not None]

                print(f"  Total headlines: {len(all_headlines)}, Gold-relevant: {len(gold_scores)}")

                if gold_scores:
                    avg_score = float(np.mean(gold_scores))
                    # Smooth into -1..1 range
                    smoothed = float(np.tanh(avg_score / 3))

                    if smoothed > 0.1:
                        label = "bullish"
                    elif smoothed < -0.1:
                        label = "bearish"
                    else:
                        label = "neutral"

                    sentiment_data["score"] = round(smoothed, 4)
                    sentiment_data["label"] = label
                    sentiment_data["headlines_analyzed"] = len(gold_scores)
                    sentiment_data["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    sentiment_data["last_error"] = None
                    print(f"  Result: {label} ({smoothed:.4f}) from {len(gold_scores)} gold headlines")
                else:
                    sentiment_data["last_error"] = "No gold-relevant headlines found in feeds"
                    print("  No gold-relevant headlines found.")
            else:
                sentiment_data["last_error"] = "All RSS feeds returned 0 headlines (possible block or network issue)"
                print("  All feeds returned 0 headlines.")

        except Exception as e:
            err = f"Scoring error: {e}"
            sentiment_data["last_error"] = err
            print(f"  {err}")

        time.sleep(60)  # update every 1 minute


# FIX 4: Run once immediately on startup before the loop sleeps
def start_updater():
    update_sentiment()  # first run right away


# Start updater in background thread
threading.Thread(target=start_updater, daemon=True).start()


# -----------------------------
# API ENDPOINTS
# -----------------------------

@app.get("/sentiment/{symbol}")
def get_sentiment(symbol: str):
    """
    Returns gold/XAUUSD sentiment.
    Accepts: gold, Gold, GOLD, xauusd, XAUUSD, XAU, xau
    """
    # FIX 2: Case-insensitive symbol matching
    sym = symbol.lower().strip()
    if sym not in ("gold", "xauusd", "xau", "gc=f", "gc"):
        return {
            "error": f"Unknown symbol '{symbol}'. Use gold, XAUUSD, or XAU.",
            "supported": ["gold", "XAUUSD", "XAU"],
        }

    return {
        "symbol": symbol.upper(),
        "sentiment_score": sentiment_data["score"],   # -1.0 (bearish) to +1.0 (bullish)
        "sentiment_label": sentiment_data["label"],   # "bullish" | "bearish" | "neutral"
        "headlines_analyzed": sentiment_data["headlines_analyzed"],
        "last_updated": sentiment_data["last_updated"],
        "last_error": sentiment_data["last_error"],   # FIX 3: surface errors visibly
    }


@app.get("/health")
def health():
    """Quick health check — shows current sentiment without needing a symbol."""
    return {
        "status": "ok",
        "sentiment": sentiment_data,
    }