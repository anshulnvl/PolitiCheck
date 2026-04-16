import os
import re
import httpx
import asyncpg
import whois
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import logging
from functools import lru_cache

log = logging.getLogger(__name__)

GOOGLE_KEY = os.getenv("GOOGLE_FACTCHECK_KEY", "")
# Use environment variable for external news database connection
# Disable external fact-checking if not configured
EXTERNAL_NEWS_DATABASE_URL = os.getenv("EXTERNAL_NEWS_DATABASE_URL")
if not EXTERNAL_NEWS_DATABASE_URL:
    log.warning(
        "EXTERNAL_NEWS_DATABASE_URL not set. External fact-checking will be disabled. "
        "Set env var to postgres://user:pass@host/db to enable."
    )

GOOGLE_FC_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
USE_SPACY_CLAIM_SCORER = os.getenv("EXTERNAL_USE_SPACY", "false").lower() == "true"

_LOCAL_SOURCE_REPUTATION = {
    "reuters.com": 0.95,
    "bbc.com": 0.93,
    "apnews.com": 0.94,
    "infowars.com": 0.05,
    "naturalnews.com": 0.04,
}


@dataclass
class ExternalSignalResult:
    score: float                          # 0 = trustworthy, 1 = suspicious
    features: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


async def compute(text: str, source_domain: str = "", offline: bool = False) -> ExternalSignalResult:
    features = {}

    # Training can run in offline mode to avoid network/socket failures.
    if offline:
        google_result = {"google_fc_claims_found": 0, "google_fc_verdicts": []}
        claimbuster_result = _local_checkworthy_score(text)
        reputation_result = _local_source_reputation(source_domain)
    else:
        # Run all three checks, but isolate failures so one source never breaks the whole task.
        try:
            google_result = await _google_factcheck(text)
        except Exception as e:
            google_result = {"google_fc_error": str(e), "google_fc_claims_found": 0, "google_fc_verdicts": []}

        try:
            claimbuster_result = _local_checkworthy_score(text)
        except Exception as e:
            claimbuster_result = {
                "claimbuster_error": str(e),
                "claimbuster_max_score": 0,
                "claimbuster_avg_score": 0,
                "claimbuster_sentence_scores": [],
                "claimbuster_source": "fallback_error",
            }

        try:
            reputation_result = await _source_reputation(source_domain)
        except Exception as e:
            reputation_result = {"reputation_score": 0.5, "reputation_error": str(e), "reputation_source": "error_fallback"}

    features.update(google_result)
    features.update(claimbuster_result)
    features.update(reputation_result)

    score = _compute_score(features)
    return ExternalSignalResult(score=score, features=features)


# ─── Google Fact Check ────────────────────────────────────────────────────────

async def _google_factcheck(text: str) -> dict:
    if not GOOGLE_KEY:
        return {"google_fc_claims_found": 0, "google_fc_verdicts": []}

    # Use first 200 chars as query (API limit)
    query = text[:200].strip()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(GOOGLE_FC_URL, params={
                "key": GOOGLE_KEY,
                "query": query,
                "languageCode": "en"
            })
            resp.raise_for_status()
            data = resp.json()

        claims = data.get("claims", [])
        verdicts = []
        for claim in claims[:5]:   # cap at 5
            for review in claim.get("claimReview", []):
                verdicts.append({
                    "publisher": review.get("publisher", {}).get("name"),
                    "rating": review.get("textualRating"),
                    "url": review.get("url"),
                })

        return {
            "google_fc_claims_found": len(claims),
            "google_fc_verdicts": verdicts,
            "google_fc_has_false_rating": any(
                "false" in (v.get("rating") or "").lower() for v in verdicts
            )
        }
    except Exception as e:
        return {"google_fc_error": str(e), "google_fc_claims_found": 0}


# ─── ClaimBuster ─────────────────────────────────────────────────────────────

# async def _claimbuster_score(text: str) -> dict:
#     if not CLAIMBUSTER_KEY:
#         return {"claimbuster_max_score": None, "claimbuster_avg_score": None}

#     # Split into sentences, score each (max 10 to stay within rate limit)
#     sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20][:10]
#     scores = []

#     try:
#         async with httpx.AsyncClient(timeout=15) as client:
#             for sentence in sentences:
#                 resp = await client.post(
#                     CLAIMBUSTER_URL,
#                     headers={"x-api-key": CLAIMBUSTER_KEY},
#                     json={"input_text": sentence}
#                 )
#                 resp.raise_for_status()
#                 data = resp.json()
#                 score = data.get("results", [{}])[0].get("score", 0)
#                 scores.append(score)

#         return {
#             "claimbuster_max_score": max(scores) if scores else 0,
#             "claimbuster_avg_score": sum(scores) / len(scores) if scores else 0,
#             "claimbuster_sentence_scores": scores
#         }
#     except Exception as e:
#         return {"claimbuster_error": str(e), "claimbuster_max_score": None}
    


def _local_checkworthy_score(text: str) -> dict:
    """
    Mimics ClaimBuster: scores sentences for 'check-worthiness'
    based on presence of named entities, numbers, and strong claims.
    """
    raw_sentences = _split_sentences(text[:5000])
    claim_verbs = {
        "said", "claim", "claims", "claimed", "prove", "show", "shows", "confirm",
        "deny", "announce", "announced", "reveal", "revealed", "state", "stated", "allege", "alleged",
    }

    sentence_scores = []
    spacy_nlp = _get_nlp() if USE_SPACY_CLAIM_SCORER else None

    for sentence in raw_sentences:
        s = 0.0
        tokens = [tok for tok in re.split(r"\s+", sentence) if tok]
        if not tokens:
            continue

        if spacy_nlp is not None:
            doc = spacy_nlp(sentence)
            ents = [e for e in doc.ents if e.label_ in ("PERSON", "ORG", "GPE", "EVENT")]
            nums = [t for t in doc if t.like_num or t.text.endswith('%')]
            has_claim_verb = any(t.lemma_.lower() in claim_verbs for t in doc)
        else:
            # Fallback heuristics avoid heavy NLP libs in worker processes.
            ents = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", sentence)
            nums = re.findall(r"\b\d+(?:[\.,]\d+)?%?\b", sentence)
            has_claim_verb = bool(re.search(r"\b(" + "|".join(claim_verbs) + r")\b", sentence, flags=re.IGNORECASE))

        s += min(len(ents) * 0.2, 0.4)
        s += min(len(nums) * 0.15, 0.3)
        if has_claim_verb:
            s += 0.2

        if len(tokens) < 5:
            s *= 0.3

        sentence_scores.append({"sentence": sentence, "score": round(min(s, 1.0), 4)})

    numeric_scores = [item["score"] for item in sentence_scores]

    return {
        "claimbuster_max_score": max(numeric_scores) if numeric_scores else 0,
        "claimbuster_avg_score": round(sum(numeric_scores) / len(numeric_scores), 4) if numeric_scores else 0,
        "claimbuster_sentence_scores": sentence_scores,
        "claimbuster_source": "spacy" if spacy_nlp is not None else "local_heuristic",
    }


@lru_cache(maxsize=1)
def _get_nlp():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception as exc:
        log.warning("spaCy claim scorer unavailable, using heuristic fallback: %s", exc)
        return None


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _local_source_reputation(domain: str) -> dict:
    if not domain:
        return {"reputation_score": 0.5, "reputation_source": "offline_no_domain"}

    rep = _LOCAL_SOURCE_REPUTATION.get(domain.lower(), 0.5)
    return {
        "reputation_score": round(rep, 4),
        "reputation_source": "offline_static",
    }


# ─── Source Reputation ────────────────────────────────────────────────────────

async def _source_reputation(domain: str) -> dict:
    if not domain:
        return {"reputation_score": 0.5, "reputation_source": "no_domain"}

    # 1. Try PostgreSQL cache first
    if not EXTERNAL_NEWS_DATABASE_URL:
        return {
            "reputation_score": 0.5,
            "fact_check_available": False,
            "reputation_source": "db_not_configured",
        }
    
    try:
        conn = await asyncpg.connect(EXTERNAL_NEWS_DATABASE_URL, timeout=5)
        row = await conn.fetchrow(
            "SELECT newsguard_score, mbfc_rating, domain_age_days FROM source_reputation WHERE domain = $1",
            domain
        )
        await conn.close()

        if row:
            return _row_to_reputation(row, source="db_cache")
    except Exception:
        pass  # Fall through to WHOIS fallback

    # 2. Fallback — domain age via WHOIS
    try:
        w = whois.whois(domain)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if creation and hasattr(creation, "tzinfo"):
            creation_dt = creation if creation.tzinfo else creation.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - creation_dt).days
        else:
            age_days = 0
        rep_score = max(0.0, 1.0 - (age_days / 3650))  # older → more trusted
        return {
            "reputation_score": round(rep_score, 4),
            "domain_age_days": age_days,
            "reputation_source": "whois_fallback"
        }
    except Exception as e:
        return {"reputation_score": 0.5, "reputation_error": str(e), "reputation_source": "unknown"}


def _row_to_reputation(row, source: str) -> dict:
    ng = row["newsguard_score"]   # 0–100
    mbfc = row["mbfc_rating"]     # "HIGH", "MIXED", "LOW", "SATIRE"

    MBFC_MAP = {"HIGH": 0.1, "MIXED": 0.4, "LOW": 0.9, "SATIRE": 0.7}
    if ng is not None:
        rep_score = 1.0 - (ng / 100)    # 100 = totally trusted = score 0
    elif mbfc in MBFC_MAP:
        rep_score = MBFC_MAP[mbfc]
    else:
        rep_score = 0.5

    return {
        "reputation_score": round(rep_score, 4),
        "newsguard_score": ng,
        "mbfc_rating": mbfc,
        "reputation_source": source
    }


# ─── Aggregation ─────────────────────────────────────────────────────────────

def _compute_score(f: dict) -> float:
    score = 0.5   # neutral default

    # Bad reputation pulls score up (toward 1 = fake)
    rep = f.get("reputation_score", 0.5)
    score = score * 0.4 + rep * 0.6

    # False fact-check verdicts are a strong signal
    if f.get("google_fc_has_false_rating"):
        score = min(score + 0.3, 1.0)

    # High ClaimBuster score = sentence is check-worthy but not yet verified
    cb = f.get("claimbuster_max_score")
    if cb is not None:
        score = score * 0.7 + cb * 0.3

    return round(max(0.0, min(1.0, score)), 4)