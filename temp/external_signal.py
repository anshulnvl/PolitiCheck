import os
import re
import httpx
# import asyncpg
# import whois
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

GOOGLE_KEY = os.getenv("GOOGLE_FACTCHECK_KEY", "")
CLAIMBUSTER_KEY = os.getenv("CLAIMBUSTER_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/newsdb")

GOOGLE_FC_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
CLAIMBUSTER_URL = "https://idir.uta.edu/claimbuster/api/v2/score/text/"


@dataclass
class ExternalSignalResult:
    score: float                          # 0 = trustworthy, 1 = suspicious
    features: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


async def compute(text: str, source_domain: str = "") -> ExternalSignalResult:
    features = {}

    # Run all three checks, collect results
    google_result  = await _google_factcheck(text)
    claimbuster_result = await _claimbuster_score(text)
    reputation_result  = await _source_reputation(source_domain)

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

async def _claimbuster_score(text: str) -> dict:
    if not CLAIMBUSTER_KEY:
        return {"claimbuster_max_score": None, "claimbuster_avg_score": None}

    # Split into sentences, score each (max 10 to stay within rate limit)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20][:10]
    scores = []

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            for sentence in sentences:
                resp = await client.post(
                    CLAIMBUSTER_URL,
                    headers={"x-api-key": CLAIMBUSTER_KEY},
                    json={"input_text": sentence}
                )
                resp.raise_for_status()
                data = resp.json()
                score = data.get("results", [{}])[0].get("score", 0)
                scores.append(score)

        return {
            "claimbuster_max_score": max(scores) if scores else 0,
            "claimbuster_avg_score": sum(scores) / len(scores) if scores else 0,
            "claimbuster_sentence_scores": scores
        }
    except Exception as e:
        return {"claimbuster_error": str(e), "claimbuster_max_score": None}


# ─── Source Reputation ────────────────────────────────────────────────────────

async def _source_reputation(domain: str) -> dict:
    if not domain:
        return {"reputation_score": 0.5, "reputation_source": "no_domain"}

    # 1. Try PostgreSQL cache first
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        row = await conn.fetchrow(
            "SELECT newsguard_score, mbfc_rating, domain_age_days FROM fact_check_cache WHERE domain = $1",
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
        age_days = (datetime.now(timezone.utc) - creation.replace(tzinfo=timezone.utc)).days if creation else 0
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