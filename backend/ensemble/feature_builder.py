import asyncio
import re
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Import your existing scorers (adjust paths to match your project)
from backend.signals.ml_signal import compute   as ml_score_fn
from backend.signals.linguistic_signal import compute as ling_score_fn
from backend.signals.external_signal   import compute  as ext_score_fn

_vader = SentimentIntensityAnalyzer()


def _run_async(coro):
    """Run an async coroutine synchronously (safe for scripts/training)."""
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)

# Source reputation lookup — expand as needed
SOURCE_REPUTATION = {
    "reuters.com": 0.95, "bbc.com": 0.93, "apnews.com": 0.94,
    "infowars.com": 0.05, "naturalnews.com": 0.04,
}

def build_feature_row(article: dict, offline: bool = False) -> dict:
    """
    article = {
        "text": str,
        "title": str,
        "source_domain": str,    # e.g. "reuters.com"
        "pub_date": str,         # ISO format, optional
    }
    Returns a flat dict of numeric features.
    """
    text  = article.get("text", "")
    title = article.get("title", "")
    domain = article.get("source_domain", "").lower()

    # ── Signal scores from your three modules ─────────────────
    ml_score         = ml_score_fn(text).score                      # float 0-1
    linguistic_score = ling_score_fn(text).score                    # float 0-1
    external_score   = _run_async(ext_score_fn(text, domain, offline=offline)).score # float 0-1

    # ── VADER sentiment ────────────────────────────────────────
    vader = _vader.polarity_scores(text)
    vader_compound = vader["compound"]   # -1 to +1
    vader_neg      = vader["neg"]
    vader_pos      = vader["pos"]

    # ── Linguistic surface features ────────────────────────────
    words     = text.split()
    word_count = len(words)
    caps_ratio = (
        sum(1 for w in words if w.isupper() and len(w) > 1) / max(word_count, 1)
    )
    exclaim_density = text.count("!") / max(word_count, 1)
    question_density = text.count("?") / max(word_count, 1)
    quote_count = len(re.findall(r'"[^"]{10,}"', text))  # quoted phrases ≥10 chars
    avg_sentence_len = word_count / max(text.count("."), 1)

    # ── Source features ────────────────────────────────────────
    source_reputation = SOURCE_REPUTATION.get(domain, 0.5)
    has_known_source  = 1.0 if domain in SOURCE_REPUTATION else 0.0

    # ── Title features ─────────────────────────────────────────
    title_caps_ratio = (
        sum(1 for w in title.split() if w.isupper()) / max(len(title.split()), 1)
    )
    title_exclaim = 1.0 if "!" in title else 0.0

    return {
        # Core signal scores
        "ml_score":           ml_score,
        "linguistic_score":   linguistic_score,
        "external_score":     external_score,
        # VADER
        "vader_compound":     vader_compound,
        "vader_neg":          vader_neg,
        "vader_pos":          vader_pos,
        # Text surface
        "caps_ratio":         caps_ratio,
        "exclaim_density":    exclaim_density,
        "question_density":   question_density,
        "quote_count":        float(quote_count),
        "avg_sentence_len":   avg_sentence_len,
        "word_count":         float(word_count),
        # Source
        "source_reputation":  source_reputation,
        "has_known_source":   has_known_source,
        # Title
        "title_caps_ratio":   title_caps_ratio,
        "title_exclaim":      title_exclaim,
    }


def build_feature_row_precomputed(
    article: dict,
    ml_score: float,
    linguistic_score: float,
    external_score: float,
) -> dict:
    """
    Same feature vector as build_feature_row() but accepts pre-computed signal
    scores instead of re-running the three signal modules.  Use this in the
    orchestrator after the Celery fan-out so signals are never called twice.

    article = {"text": str, "title": str, "source_domain": str}
    """
    text   = article.get("text", "")
    title  = article.get("title", "")
    domain = article.get("source_domain", "").lower()

    vader = _vader.polarity_scores(text)

    words      = text.split()
    word_count = len(words)

    return {
        "ml_score":           ml_score,
        "linguistic_score":   linguistic_score,
        "external_score":     external_score,
        "vader_compound":     vader["compound"],
        "vader_neg":          vader["neg"],
        "vader_pos":          vader["pos"],
        "caps_ratio":         sum(1 for w in words if w.isupper() and len(w) > 1) / max(word_count, 1),
        "exclaim_density":    text.count("!") / max(word_count, 1),
        "question_density":   text.count("?") / max(word_count, 1),
        "quote_count":        float(len(re.findall(r'"[^"]{10,}"', text))),
        "avg_sentence_len":   word_count / max(text.count("."), 1),
        "word_count":         float(word_count),
        "source_reputation":  SOURCE_REPUTATION.get(domain, 0.5),
        "has_known_source":   1.0 if domain in SOURCE_REPUTATION else 0.0,
        "title_caps_ratio":   sum(1 for w in title.split() if w.isupper()) / max(len(title.split()), 1),
        "title_exclaim":      1.0 if "!" in title else 0.0,
    }