# main.py
from fastapi import FastAPI
import threading
import time
import feedparser
import numpy as np
import re
import urllib.request
import ssl
from threading import Lock
from copy import deepcopy
import os

# -----------------------------
# GLOBAL VARIABLES
# -----------------------------
app = FastAPI()

SYMBOLS = [
    "XAUUSD", "BTCUSD",
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "EURCHF", "GBPCHF",
]

# Thread-safe sentiment store with lock
sentiment_lock = Lock()
sentiment_store: dict[str, dict] = {
    sym: {"score": 0.0, "label": "neutral", "headlines_analyzed": 0,
          "last_updated": None, "last_error": None}
    for sym in SYMBOLS
}

# -----------------------------
# RSS FEEDS
# -----------------------------
RSS_FEEDS = [
    "https://www.kitco.com/rss/kitconews.rss",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://finance.yahoo.com/rss/headline?s=EURUSD%3DX",
    "https://finance.yahoo.com/rss/headline?s=GC%3DF",
    "https://finance.yahoo.com/rss/headline?s=GBPUSD%3DX",
    "https://finance.yahoo.com/rss/headline?s=USDJPY%3DX",
    "https://finance.yahoo.com/rss/headline?s=BTC-USD",
    "https://cointelegraph.com/rss",
    "https://coindesk.com/arc/outboundfeeds/rss/",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

# SSL context with verification disabled for free RSS feeds (acceptable for public data)
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

# -----------------------------
# SYMBOL CONFIG - COMPLETE WITH ALL 17 SYMBOLS
# -----------------------------
SYMBOL_CONFIG: dict[str, dict] = {

    "XAUUSD": {
        "keywords": ["gold", "xau", "bullion", "precious metal", "yellow metal", "safe haven metal", "xauusd"],
        "base": "xau", "quote": "usd",
        "base_strong": [
            "gold demand rises", "demand for gold", "investors buy gold", "flee to gold",
            "flock to gold", "gold buying", "gold outperforms", "gold shines",
            "safe haven demand", "gold appeal", "gold supported", "gold attracts",
            "gold seen as refuge", "interest in gold", "gold outlook positive",
        ],
        "base_weak": [
            "gold loses appeal", "gold selling", "gold under pressure", "gold weakens",
            "gold outlook negative", "investors dump gold", "gold shunned",
            "gold loses shine", "gold unattractive",
        ],
        "quote_strong": [
            "dollar strengthens", "dollar surges", "dollar rallies", "greenback rises",
            "usd gains", "dollar dominates", "dollar buying", "strong dollar",
            "dollar outperforms", "dollar climbs", "greenback advances",
        ],
        "quote_weak": [
            "dollar weakens", "dollar falls", "dollar drops", "greenback slides",
            "usd declines", "dollar selling", "weak dollar", "dollar under pressure",
            "dollar loses ground", "greenback retreats",
        ],
        "pair_up": [
            "gold rises", "gold gains", "gold surges", "gold rallies", "gold jumps",
            "gold climbs", "gold advances", "gold higher", "gold soars", "gold up",
            "xauusd rises", "xauusd higher", "bullion gains", "bullion rises",
            "gold hits record", "gold at high", "gold breaks out",
        ],
        "pair_down": [
            "gold falls", "gold drops", "gold declines", "gold slides", "gold slips",
            "gold tumbles", "gold plunges", "gold lower", "gold down", "gold sinks",
            "xauusd falls", "xauusd lower", "bullion drops", "bullion falls",
            "gold hits low", "gold breaks down",
        ],
    },

    "BTCUSD": {
        "keywords": ["bitcoin", "btc", "crypto", "cryptocurrency", "digital currency", "btcusd", "btc/usd"],
        "base": "btc", "quote": "usd",
        "base_strong": [
            "bitcoin demand", "investors buy bitcoin", "institutional buying",
            "crypto adoption", "bitcoin etf inflows", "bitcoin accumulation",
            "bullish on bitcoin", "bitcoin optimism", "bitcoin sentiment positive",
            "crypto market optimism", "bitcoin momentum", "bitcoin breakout",
        ],
        "base_weak": [
            "bitcoin selling", "crypto fear", "bitcoin regulation fears",
            "bitcoin etf outflows", "bitcoin dump", "crypto crash fears",
            "bearish on bitcoin", "bitcoin pessimism", "bitcoin negative sentiment",
        ],
        "quote_strong": [
            "dollar strengthens", "greenback rises", "usd gains", "strong dollar",
        ],
        "quote_weak": [
            "dollar weakens", "greenback falls", "usd drops", "weak dollar",
        ],
        "pair_up": [
            "bitcoin rises", "bitcoin gains", "bitcoin surges", "bitcoin rallies",
            "bitcoin jumps", "bitcoin climbs", "bitcoin higher", "bitcoin soars",
            "bitcoin up", "btc up", "btc rises", "btc gains", "btc higher",
            "crypto rally", "crypto rises", "crypto surges", "crypto gains",
            "bitcoin breaks record", "bitcoin hits high", "bitcoin all time high",
        ],
        "pair_down": [
            "bitcoin falls", "bitcoin drops", "bitcoin declines", "bitcoin slides",
            "bitcoin tumbles", "bitcoin plunges", "bitcoin lower", "bitcoin down",
            "btc down", "btc falls", "btc drops", "btc lower",
            "crypto selloff", "crypto falls", "crypto drops", "crypto lower",
            "bitcoin crashes", "bitcoin hits low",
        ],
    },

    "EURUSD": {
        "keywords": ["eurusd", "eur/usd", "euro", "eur", "single currency"],
        "base": "eur", "quote": "usd",
        "base_strong": [
            "euro gains strength", "euro gaining", "euro outperforms", "euro supported",
            "demand for euro", "investors buy euro", "euro appeal", "bullish euro",
            "euro zone growth", "ecb hawkish", "ecb rate hike", "euro zone strong",
            "euro zone recovery", "euro positive", "eur strength",
        ],
        "base_weak": [
            "euro loses ground", "euro under pressure", "euro weakening", "bearish euro",
            "euro zone recession", "ecb dovish", "ecb rate cut", "euro zone weak",
            "euro zone slowdown", "euro negative", "eur weakness", "euro shunned",
        ],
        "quote_strong": [
            "dollar strengthens", "dollar surges", "dollar rallies", "greenback rises",
            "usd gains", "dollar dominates", "strong dollar", "dollar outperforms",
            "fed hawkish", "fed rate hike", "us economy strong",
        ],
        "quote_weak": [
            "dollar weakens", "dollar falls", "dollar drops", "greenback slides",
            "usd declines", "weak dollar", "dollar under pressure", "fed dovish",
            "fed rate cut", "us economy weak",
        ],
        "pair_up": [
            "eurusd rises", "eurusd gains", "eurusd higher", "eurusd up",
            "eurusd rallies", "eurusd climbs", "eurusd advances", "eurusd surges",
            "euro rises against dollar", "euro gains against dollar",
            "euro higher vs dollar", "euro strengthens against dollar",
            "eur/usd rises", "eur/usd higher",
        ],
        "pair_down": [
            "eurusd falls", "eurusd drops", "eurusd lower", "eurusd down",
            "eurusd slides", "eurusd declines", "eurusd slips", "eurusd weakens",
            "euro falls against dollar", "euro drops against dollar",
            "euro lower vs dollar", "euro weakens against dollar",
            "eur/usd falls", "eur/usd lower",
        ],
    },

    "GBPUSD": {
        "keywords": ["gbpusd", "gbp/usd", "pound", "sterling", "cable", "gbp", "british pound"],
        "base": "gbp", "quote": "usd",
        "base_strong": [
            "pound gains strength", "sterling outperforms", "pound supported",
            "demand for pound", "bullish pound", "boe hawkish", "boe rate hike",
            "uk economy strong", "uk growth", "pound positive", "gbp strength",
            "investors buy pound", "pound appeal",
        ],
        "base_weak": [
            "pound loses ground", "sterling under pressure", "pound weakening",
            "bearish pound", "boe dovish", "boe rate cut", "uk economy weak",
            "uk recession", "pound negative", "gbp weakness", "pound shunned",
            "uk slowdown",
        ],
        "quote_strong": [
            "dollar strengthens", "greenback rises", "usd gains", "strong dollar",
            "fed hawkish", "fed rate hike",
        ],
        "quote_weak": [
            "dollar weakens", "greenback falls", "usd drops", "weak dollar",
            "fed dovish", "fed rate cut",
        ],
        "pair_up": [
            "gbpusd rises", "gbpusd gains", "gbpusd higher", "gbpusd up",
            "pound rises", "sterling rises", "cable rises", "pound higher",
            "sterling higher", "pound gains against dollar", "sterling gains",
            "pound strengthens against dollar", "gbp/usd rises",
        ],
        "pair_down": [
            "gbpusd falls", "gbpusd drops", "gbpusd lower", "gbpusd down",
            "pound falls", "sterling falls", "cable drops", "pound lower",
            "sterling lower", "pound weakens against dollar", "gbp/usd falls",
        ],
    },

    "USDJPY": {
        "keywords": ["usdjpy", "usd/jpy", "dollar yen", "yen", "jpy", "japanese yen"],
        "base": "usd", "quote": "jpy",
        "base_strong": [
            "dollar strengthens", "greenback rises", "usd gains", "strong dollar",
            "fed hawkish", "fed rate hike", "us economy strong",
        ],
        "base_weak": [
            "dollar weakens", "greenback falls", "usd drops", "weak dollar",
            "fed dovish", "fed rate cut",
        ],
        "quote_strong": [
            "yen gains strength", "yen outperforms", "yen supported", "safe haven yen",
            "demand for yen", "investors buy yen", "boj hawkish", "boj rate hike",
            "yen appeal", "yen positive", "yen strengthening",
        ],
        "quote_weak": [
            "yen loses ground", "yen under pressure", "yen weakening",
            "boj dovish", "boj keeps rates low", "yen negative", "yen weakness",
            "yen shunned", "carry trade yen",
        ],
        "pair_up": [
            "usdjpy rises", "usdjpy higher", "usdjpy up", "usdjpy climbs",
            "dollar rises against yen", "dollar higher vs yen", "yen weakens",
            "yen falls", "yen drops", "yen slides", "dollar yen higher",
            "usd/jpy rises",
        ],
        "pair_down": [
            "usdjpy falls", "usdjpy lower", "usdjpy down", "usdjpy drops",
            "dollar falls against yen", "dollar lower vs yen", "yen strengthens",
            "yen rises", "yen gains", "yen climbs", "dollar yen lower",
            "usd/jpy falls",
        ],
    },

    "USDCHF": {
        "keywords": ["usdchf", "usd/chf", "franc", "chf", "swiss franc", "swissie"],
        "base": "usd", "quote": "chf",
        "base_strong": [
            "dollar strengthens", "greenback rises", "usd gains", "strong dollar",
            "fed hawkish", "fed rate hike",
        ],
        "base_weak": [
            "dollar weakens", "greenback falls", "usd drops", "weak dollar",
            "fed dovish", "fed rate cut",
        ],
        "quote_strong": [
            "franc gains", "safe haven franc", "franc outperforms", "franc supported",
            "demand for franc", "investors buy franc", "snb hawkish",
            "swiss economy strong", "franc positive", "franc strengthening",
            "risk off franc", "flight to safety franc",
        ],
        "quote_weak": [
            "franc weakens", "franc loses ground", "franc under pressure",
            "snb dovish", "snb intervention", "franc negative", "franc weakness",
        ],
        "pair_up": [
            "usdchf rises", "usdchf higher", "usdchf up", "usdchf climbs",
            "dollar rises against franc", "franc weakens", "franc falls",
            "franc drops", "usd/chf rises",
        ],
        "pair_down": [
            "usdchf falls", "usdchf lower", "usdchf down", "usdchf drops",
            "dollar falls against franc", "franc strengthens", "franc rises",
            "franc gains", "usd/chf falls", "safe haven franc demand",
        ],
    },

    "AUDUSD": {
        "keywords": ["audusd", "aud/usd", "aussie", "aud", "australian dollar", "australian"],
        "base": "aud", "quote": "usd",
        "base_strong": [
            "aussie gains", "australian dollar gains", "aud outperforms",
            "rba hawkish", "rba rate hike", "australia economy strong",
            "china growth positive", "commodity prices rise", "iron ore rises",
            "risk on aussie", "aussie supported", "aud positive",
        ],
        "base_weak": [
            "aussie loses ground", "australian dollar weakens", "aud under pressure",
            "rba dovish", "rba rate cut", "australia economy weak",
            "china slowdown", "commodity prices fall", "risk off aussie",
            "aud negative", "aussie shunned",
        ],
        "quote_strong": [
            "dollar strengthens", "greenback rises", "usd gains", "strong dollar",
        ],
        "quote_weak": [
            "dollar weakens", "greenback falls", "usd drops", "weak dollar",
        ],
        "pair_up": [
            "audusd rises", "audusd higher", "audusd up", "audusd gains",
            "aussie rises", "aussie higher", "australian dollar rises",
            "aud rises", "aud/usd rises", "aussie strengthens",
        ],
        "pair_down": [
            "audusd falls", "audusd lower", "audusd down", "audusd drops",
            "aussie falls", "aussie lower", "australian dollar falls",
            "aud falls", "aud/usd falls", "aussie weakens",
        ],
    },

    "USDCAD": {
        "keywords": ["usdcad", "usd/cad", "loonie", "cad", "canadian dollar", "canadian"],
        "base": "usd", "quote": "cad",
        "base_strong": [
            "dollar strengthens", "greenback rises", "usd gains", "strong dollar",
            "fed hawkish", "fed rate hike",
        ],
        "base_weak": [
            "dollar weakens", "greenback falls", "usd drops", "weak dollar",
            "fed dovish", "fed rate cut",
        ],
        "quote_strong": [
            "loonie gains", "canadian dollar gains", "cad outperforms",
            "boc hawkish", "boc rate hike", "canada economy strong",
            "oil prices rise", "crude rises", "loonie supported", "cad positive",
        ],
        "quote_weak": [
            "loonie loses ground", "canadian dollar weakens", "cad under pressure",
            "boc dovish", "boc rate cut", "canada economy weak",
            "oil prices fall", "crude drops", "loonie falls", "cad negative",
        ],
        "pair_up": [
            "usdcad rises", "usdcad higher", "usdcad up", "usdcad climbs",
            "dollar rises against cad", "loonie weakens", "loonie falls",
            "canadian dollar weakens", "usd/cad rises",
        ],
        "pair_down": [
            "usdcad falls", "usdcad lower", "usdcad down", "usdcad drops",
            "dollar falls against cad", "loonie strengthens", "loonie rises",
            "canadian dollar strengthens", "usd/cad falls",
        ],
    },

    "NZDUSD": {
        "keywords": ["nzdusd", "nzd/usd", "kiwi", "nzd", "new zealand dollar", "new zealand"],
        "base": "nzd", "quote": "usd",
        "base_strong": [
            "kiwi gains", "new zealand dollar gains", "nzd outperforms",
            "rbnz hawkish", "rbnz rate hike", "new zealand economy strong",
            "kiwi supported", "nzd positive",
        ],
        "base_weak": [
            "kiwi loses ground", "new zealand dollar weakens", "nzd under pressure",
            "rbnz dovish", "rbnz rate cut", "new zealand economy weak",
            "kiwi falls", "nzd negative",
        ],
        "quote_strong": [
            "dollar strengthens", "greenback rises", "usd gains", "strong dollar",
        ],
        "quote_weak": [
            "dollar weakens", "greenback falls", "usd drops", "weak dollar",
        ],
        "pair_up": [
            "nzdusd rises", "nzdusd higher", "nzdusd up", "nzdusd gains",
            "kiwi rises", "kiwi higher", "new zealand dollar rises",
            "nzd rises", "nzd/usd rises",
        ],
        "pair_down": [
            "nzdusd falls", "nzdusd lower", "nzdusd down", "nzdusd drops",
            "kiwi falls", "kiwi lower", "new zealand dollar falls",
            "nzd falls", "nzd/usd falls",
        ],
    },

    "EURGBP": {
        "keywords": ["eurgbp", "eur/gbp", "euro pound", "euro sterling", "euro vs pound", "euro against pound"],
        "base": "eur", "quote": "gbp",
        "base_strong": [
            "euro outperforms pound", "euro gains vs pound", "ecb hawkish",
            "euro zone strong", "euro positive", "eur strength",
        ],
        "base_weak": [
            "euro loses to pound", "euro weakens vs pound", "ecb dovish",
            "euro zone weak", "euro negative",
        ],
        "quote_strong": [
            "pound outperforms euro", "sterling gains vs euro", "boe hawkish",
            "uk economy strong", "gbp strength",
        ],
        "quote_weak": [
            "pound loses to euro", "sterling weakens vs euro", "boe dovish",
            "uk economy weak", "gbp weakness",
        ],
        "pair_up": [
            "eurgbp rises", "eurgbp higher", "eurgbp up",
            "euro rises against pound", "euro gains against sterling",
            "euro higher vs pound", "eur/gbp rises",
        ],
        "pair_down": [
            "eurgbp falls", "eurgbp lower", "eurgbp down",
            "euro falls against pound", "euro drops against sterling",
            "euro lower vs pound", "eur/gbp falls",
        ],
    },

    "EURJPY": {
        "keywords": ["eurjpy", "eur/jpy", "euro yen", "euro vs yen", "euro against yen"],
        "base": "eur", "quote": "jpy",
        "base_strong": [
            "euro outperforms yen", "euro gains vs yen", "ecb hawkish", "euro positive",
        ],
        "base_weak": [
            "euro weakens vs yen", "ecb dovish", "euro negative",
        ],
        "quote_strong": [
            "yen gains vs euro", "safe haven yen", "boj hawkish", "yen strengthening",
        ],
        "quote_weak": [
            "yen weakens vs euro", "boj dovish", "yen weakness",
        ],
        "pair_up": [
            "eurjpy rises", "eurjpy higher", "eurjpy up",
            "euro rises against yen", "euro higher vs yen", "eur/jpy rises",
        ],
        "pair_down": [
            "eurjpy falls", "eurjpy lower", "eurjpy down",
            "euro falls against yen", "euro lower vs yen", "eur/jpy falls",
        ],
    },

    "GBPJPY": {
        "keywords": ["gbpjpy", "gbp/jpy", "pound yen", "sterling yen", "pound vs yen"],
        "base": "gbp", "quote": "jpy",
        "base_strong": [
            "pound outperforms yen", "sterling gains vs yen", "boe hawkish",
            "uk economy strong", "gbp positive",
        ],
        "base_weak": [
            "pound weakens vs yen", "boe dovish", "uk economy weak", "gbp negative",
        ],
        "quote_strong": [
            "yen gains vs pound", "safe haven yen", "boj hawkish", "yen strengthening",
        ],
        "quote_weak": [
            "yen weakens vs pound", "boj dovish", "yen weakness",
        ],
        "pair_up": [
            "gbpjpy rises", "gbpjpy higher", "gbpjpy up",
            "pound rises against yen", "sterling higher vs yen", "gbp/jpy rises",
        ],
        "pair_down": [
            "gbpjpy falls", "gbpjpy lower", "gbpjpy down",
            "pound falls against yen", "sterling lower vs yen", "gbp/jpy falls",
        ],
    },

    "AUDJPY": {
        "keywords": ["audjpy", "aud/jpy", "aussie yen", "australian yen", "aussie vs yen"],
        "base": "aud", "quote": "jpy",
        "base_strong": [
            "aussie outperforms yen", "aud gains vs yen", "rba hawkish",
            "risk on sentiment", "commodity prices rise",
        ],
        "base_weak": [
            "aussie weakens vs yen", "rba dovish", "risk off sentiment",
            "commodity prices fall",
        ],
        "quote_strong": [
            "yen gains vs aussie", "safe haven yen", "boj hawkish", "risk off yen",
        ],
        "quote_weak": [
            "yen weakens vs aussie", "boj dovish", "risk on yen selling",
        ],
        "pair_up": [
            "audjpy rises", "audjpy higher", "audjpy up",
            "aussie gains vs yen", "aud higher vs yen", "aud/jpy rises",
        ],
        "pair_down": [
            "audjpy falls", "audjpy lower", "audjpy down",
            "aussie drops vs yen", "aud lower vs yen", "aud/jpy falls",
        ],
    },

    "CADJPY": {
        "keywords": ["cadjpy", "cad/jpy", "canadian yen", "loonie yen", "cad vs yen"],
        "base": "cad", "quote": "jpy",
        "base_strong": [
            "loonie gains vs yen", "cad outperforms yen", "boc hawkish", "oil rises",
        ],
        "base_weak": [
            "loonie weakens vs yen", "boc dovish", "oil falls",
        ],
        "quote_strong": [
            "yen gains vs cad", "safe haven yen", "boj hawkish",
        ],
        "quote_weak": [
            "yen weakens vs cad", "boj dovish",
        ],
        "pair_up": [
            "cadjpy rises", "cadjpy higher", "cadjpy up",
            "cad gains vs yen", "loonie higher vs yen", "cad/jpy rises",
        ],
        "pair_down": [
            "cadjpy falls", "cadjpy lower", "cadjpy down",
            "cad drops vs yen", "loonie lower vs yen", "cad/jpy falls",
        ],
    },

    "EURCHF": {
        "keywords": ["eurchf", "eur/chf", "euro franc", "euro swiss", "euro vs franc"],
        "base": "eur", "quote": "chf",
        "base_strong": [
            "euro outperforms franc", "euro gains vs franc", "ecb hawkish",
            "risk on euro", "euro zone strong",
        ],
        "base_weak": [
            "euro weakens vs franc", "ecb dovish", "euro zone weak",
        ],
        "quote_strong": [
            "franc gains vs euro", "safe haven franc", "snb hawkish",
            "flight to safety franc", "risk off franc",
        ],
        "quote_weak": [
            "franc weakens vs euro", "snb dovish", "snb intervention",
        ],
        "pair_up": [
            "eurchf rises", "eurchf higher", "eurchf up",
            "euro rises against franc", "eur/chf rises",
        ],
        "pair_down": [
            "eurchf falls", "eurchf lower", "eurchf down",
            "euro falls against franc", "eur/chf falls",
        ],
    },

    "GBPCHF": {
        "keywords": ["gbpchf", "gbp/chf", "pound franc", "sterling franc", "pound vs franc"],
        "base": "gbp", "quote": "chf",
        "base_strong": [
            "pound outperforms franc", "sterling gains vs franc", "boe hawkish",
            "uk economy strong",
        ],
        "base_weak": [
            "pound weakens vs franc", "boe dovish", "uk economy weak",
        ],
        "quote_strong": [
            "franc gains vs pound", "safe haven franc", "snb hawkish",
            "flight to safety", "risk off",
        ],
        "quote_weak": [
            "franc weakens vs pound", "snb dovish",
        ],
        "pair_up": [
            "gbpchf rises", "gbpchf higher", "gbpchf up",
            "pound rises against franc", "gbp/chf rises",
        ],
        "pair_down": [
            "gbpchf falls", "gbpchf lower", "gbpchf down",
            "pound falls against franc", "gbp/chf falls",
        ],
    },
}

# General directional patterns
BULLISH_PATTERNS = [re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE) for w in [
    "rises", "gains", "surges", "rallies", "jumps", "climbs", "advances",
    "higher", "soars", "strengthens", "breakout", "upside", "outperforms",
    "recovery", "rebound", "buying", "demand", "supported", "positive",
]]
BEARISH_PATTERNS = [re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE) for w in [
    "falls", "drops", "declines", "slides", "slips", "tumbles", "plunges",
    "lower", "sinks", "weakens", "selloff", "downside", "underperforms",
    "recession", "slowdown", "selling", "pressure", "negative", "shunned",
]]
NEUTRAL_PATTERNS = [re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE) for w in [
    "unchanged", "steady", "flat", "consolidates", "sideways", "mixed", "cautious",
]]


# -----------------------------
# RSS FETCHER
# -----------------------------
def fetch_feed(url: str) -> list[str]:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=10) as response:
            raw = response.read()
        feed = feedparser.parse(raw)
        titles    = [entry.title   for entry in feed.entries if hasattr(entry, "title")]
        summaries = [entry.summary for entry in feed.entries if hasattr(entry, "summary")]
        return titles + summaries
    except Exception as e:
        print(f"  [feed error] {url}: {e}")
        return []


# -----------------------------
# SMART SCORING ENGINE
# -----------------------------
def score_headline(text: str, symbol: str) -> float | None:
    cfg = SYMBOL_CONFIG.get(symbol)
    if cfg is None:
        return None

    t = text.lower()

    if not any(kw in t for kw in cfg["keywords"]):
        return None

    score = 0.0

    # Tier 1: Explicit pair direction (weight 3)
    for phrase in cfg["pair_up"]:
        if phrase in t:
            score += 3.0
    for phrase in cfg["pair_down"]:
        if phrase in t:
            score -= 3.0

    # Tier 2: Base currency strength/weakness (weight 2)
    for phrase in cfg.get("base_strong", []):
        if phrase in t:
            score += 2.0
    for phrase in cfg.get("base_weak", []):
        if phrase in t:
            score -= 2.0

    # Tier 3: Quote currency strength/weakness inverted (weight 2)
    for phrase in cfg.get("quote_strong", []):
        if phrase in t:
            score -= 2.0
    for phrase in cfg.get("quote_weak", []):
        if phrase in t:
            score += 2.0

    # Tier 4: General directional words (weight 1)
    for pat in BULLISH_PATTERNS:
        if pat.search(t):
            score += 1.0
    for pat in BEARISH_PATTERNS:
        if pat.search(t):
            score -= 1.0

    # Tier 5: Neutral dampening
    for pat in NEUTRAL_PATTERNS:
        if pat.search(t):
            score *= 0.5

    return score


# -----------------------------
# BACKGROUND UPDATER
# -----------------------------
def update_all_sentiment():
    # Get update interval from environment variable (default 300 seconds = 5 minutes)
    update_interval = int(os.getenv("UPDATE_INTERVAL_SECONDS", 300))
    
    while True:
        print("\n[sentiment] Fetching news...")
        all_headlines: list[str] = []
        
        for url in RSS_FEEDS:
            items = fetch_feed(url)
            print(f"  {len(items)} items from {url}")
            all_headlines.extend(items)
            # Free memory
            items = None

        # Remove duplicates and limit to last 500 unique headlines to manage memory
        all_headlines = list(dict.fromkeys(all_headlines))[:500]
        print(f"  Total unique items: {len(all_headlines)}")

        for symbol in SYMBOLS:
            try:
                scores   = [score_headline(h, symbol) for h in all_headlines]
                relevant = [s for s in scores if s is not None]

                if relevant:
                    avg      = float(np.mean(relevant))
                    smoothed = float(np.tanh(avg / 3))
                    label    = "bullish" if smoothed > 0.1 else "bearish" if smoothed < -0.1 else "neutral"
                    
                    # Thread-safe update with lock
                    with sentiment_lock:
                        sentiment_store[symbol].update({
                            "score": round(smoothed, 4), 
                            "label": label,
                            "headlines_analyzed": len(relevant),
                            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "last_error": None,
                        })
                    print(f"  {symbol}: {label} ({smoothed:.4f}) — {len(relevant)} headlines")
                else:
                    with sentiment_lock:
                        sentiment_store[symbol].update({
                            "score": 0.0, 
                            "label": "neutral", 
                            "headlines_analyzed": 0,
                            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "last_error": "No relevant headlines — defaulting to neutral",
                        })
            except Exception as e:
                with sentiment_lock:
                    sentiment_store[symbol]["last_error"] = str(e)
                print(f"  {symbol}: ERROR — {e}")
        
        # Free memory before next iteration
        all_headlines = None
        
        print(f"\n[sentiment] Next update in {update_interval} seconds...")
        time.sleep(update_interval)


# Start background thread
update_thread = threading.Thread(target=update_all_sentiment, daemon=True)
update_thread.start()


# -----------------------------
# API ENDPOINTS
# -----------------------------
@app.get("/sentiment/{symbol}")
def get_sentiment(symbol: str):
    clean = symbol.upper().strip()
    
    # Remove common MT4/MT5 suffixes
    suffixes_to_remove = [".A", ".B", ".M", ".PRO", ".ECN", ".RAW", ".MICRO", ".MINI"]
    for suffix in suffixes_to_remove:
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)]
            break
    
    # Thread-safe read with deep copy
    with sentiment_lock:
        if clean not in sentiment_store:
            return {
                "error": f"Symbol '{symbol}' not supported.", 
                "supported_symbols": SYMBOLS
            }
        data = deepcopy(sentiment_store[clean])
    
    return {
        "symbol": clean,
        "sentiment_score": data["score"],
        "sentiment_label": data["label"],
        "headlines_analyzed": data["headlines_analyzed"],
        "last_updated": data["last_updated"],
        "last_error": data["last_error"],
    }


@app.get("/health")
def health():
    with sentiment_lock:
        # Return a copy of sentiment store summary
        summary = {}
        for sym, data in sentiment_store.items():
            summary[sym] = {
                "label": data["label"],
                "last_updated": data["last_updated"],
                "headlines_analyzed": data["headlines_analyzed"]
            }
    
    return {
        "status": "ok", 
        "update_interval_seconds": int(os.getenv("UPDATE_INTERVAL_SECONDS", 300)),
        "symbols_count": len(SYMBOLS),
        "symbols": SYMBOLS,
        "last_update_summary": summary
    }


@app.get("/")
def root():
    return {
        "message": "Gold & Forex Sentiment Engine",
        "version": "1.0.0",
        "supported_symbols": SYMBOLS,
        "total_symbols": len(SYMBOLS),
        "endpoints": [
            "/sentiment/{symbol} - Get sentiment for specific symbol",
            "/health - Check service health and status",
            "/ - This help message"
        ],
        "example": "/sentiment/BTCUSD",
        "note": "Suffixes like .A are automatically stripped - use base symbol name"
    }


# -----------------------------
# MAIN ENTRY POINT (for local development)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)