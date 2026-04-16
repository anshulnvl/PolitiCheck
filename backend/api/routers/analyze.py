"""
api/routers/analyze.py
======================
POST /api/analyze — main credibility-analysis endpoint.

Calls the pipeline orchestrator, then reshapes its internal result into the
public API contract documented in the project spec.

Request
-------
{
    "text":       str,                          # article body, URL, or statement
    "input_type": "article" | "url" | "statement"  # optional, default "article"
}

Response
--------
See AnalyzeResponse below.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.pipeline.orchestrator import run_pipeline

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analyze"])

# input_type values accepted by this API → what the orchestrator expects
_INPUT_TYPE_MAP = {
    "article":   "text",
    "url":       "url",
    "statement": "text",
}


# ── Request / Response schemas ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Article body, URL, or statement to analyse")
    input_type: str = Field("article", description="article | url | statement")


class SignalResult(BaseModel):
    score: int = Field(..., ge=0, le=100, description="0 = no concern, 100 = maximum concern")
    description: str


class Claim(BaseModel):
    publisher: str
    rating: str
    url: str


class AnalyzeResponse(BaseModel):
    verdict: str                                   # FAKE | MISLEADING | CREDIBLE | UNVERIFIABLE
    credibility_score: int                         # 0 (fake) → 100 (credible)
    confidence: int                                # 0 → 100
    verdict_title: str
    verdict_summary: str
    ml_signal: SignalResult
    linguistic_signal: SignalResult
    external_signal: SignalResult
    claims: List[Claim]
    linguistic_flags: List[str]
    highlighted_sentences: List[str]
    bias_score: int                                # −100 (very negative) → +100 (very positive)
    bias_explanation: str
    context: str
    missing: str
    detail: str


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse, summary="Analyse text credibility")
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    pipeline_type = _INPUT_TYPE_MAP.get(req.input_type, "auto")

    try:
        result = await run_pipeline(req.text, input_type=pipeline_type)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Signal workers timed out — ensure Celery workers are running")
    except Exception as exc:
        log.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return _transform(result)


# ── Pipeline result → API response ────────────────────────────────────────────

def _transform(r: dict) -> AnalyzeResponse:
    cred_score = round((r.get("credibility_score") or 0.5) * 100)
    conf       = round((r.get("confidence")        or 0.0) * 100)

    sig      = r.get("signals", {})
    ml_sig   = sig.get("ml",         {})
    ling_sig = sig.get("linguistic", {})
    ext_sig  = sig.get("external",   {})

    ml_score   = round((ml_sig.get("score")   or 0.0) * 100)
    ling_score = round((ling_sig.get("score") or 0.0) * 100)
    ext_score  = round((ext_sig.get("score")  or 0.0) * 100)

    ling_features = ling_sig.get("features", {})
    ext_features  = ext_sig.get("features",  {})
    ling_flags    = ling_sig.get("flags",    [])
    explanation   = r.get("explanation", {})

    # ── Verdict mapping ───────────────────────────────────────────────────────
    raw_verdict = r.get("verdict", "UNCERTAIN")
    if raw_verdict == "UNCERTAIN":
        has_false_rating = ext_features.get("google_fc_has_false_rating", False)
        n_claims         = ext_features.get("google_fc_claims_found", 0)
        if ling_score >= 40 or has_false_rating or n_claims > 0:
            verdict = "MISLEADING"
        else:
            verdict = "UNVERIFIABLE"
    else:
        verdict = raw_verdict  # CREDIBLE or FAKE

    # ── Bias: VADER compound is −1 → +1 ──────────────────────────────────────
    bias_score = round((ling_features.get("vader_compound") or 0.0) * 100)

    # ── External claims ───────────────────────────────────────────────────────
    claims = [
        Claim(
            publisher=v.get("publisher", ""),
            rating=v.get("rating", ""),
            url=v.get("url", ""),
        )
        for v in ext_features.get("google_fc_verdicts", [])
        if isinstance(v, dict)
    ]

    # ── Highlighted sentences (ClaimBuster score ≥ 0.5) ──────────────────────
    highlighted = [
        item["sentence"]
        for item in ext_features.get("claimbuster_sentence_scores", [])
        if isinstance(item, dict) and item.get("score", 0) >= 0.5 and item.get("sentence")
    ]

    return AnalyzeResponse(
        verdict=verdict,
        credibility_score=cred_score,
        confidence=conf,
        verdict_title=_verdict_title(verdict),
        verdict_summary=_verdict_summary(verdict, cred_score, conf, r.get("document", {})),
        ml_signal=SignalResult(
            score=ml_score,
            description=_ml_description(ml_score, ml_sig.get("features", {})),
        ),
        linguistic_signal=SignalResult(
            score=ling_score,
            description=_ling_description(ling_score, ling_flags),
        ),
        external_signal=SignalResult(
            score=ext_score,
            description=_ext_description(ext_score, ext_features),
        ),
        claims=claims,
        linguistic_flags=ling_flags,
        highlighted_sentences=highlighted,
        bias_score=bias_score,
        bias_explanation=_bias_explanation(bias_score, ling_features),
        context=_context(r),
        missing=_missing(r),
        detail=_detail(explanation),
    )


# ── Text generators ───────────────────────────────────────────────────────────

def _verdict_title(verdict: str) -> str:
    return {
        "CREDIBLE":      "Credible Content",
        "FAKE":          "Likely Misinformation",
        "MISLEADING":    "Potentially Misleading",
        "UNVERIFIABLE":  "Cannot Be Verified",
    }.get(verdict, "Analysis Complete")


def _verdict_summary(verdict: str, score: int, confidence: int, doc: dict) -> str:
    source = doc.get("source", "")
    src    = f" from {source}" if source else ""

    summaries = {
        "CREDIBLE": (
            f"This content{src} shows strong markers of credibility (score: {score}/100). "
            "No significant red flags were detected across language, machine-learning, or external signals."
        ),
        "FAKE": (
            f"This content{src} shows significant markers of misinformation (score: {score}/100). "
            "Multiple independent signals flag this content as unreliable."
        ),
        "MISLEADING": (
            f"This content{src} contains potentially misleading elements (score: {score}/100). "
            "While claims may not be outright false, the framing or language appears deceptive or exaggerated."
        ),
        "UNVERIFIABLE": (
            f"This content{src} could not be independently verified (score: {score}/100). "
            "Insufficient external data exists to confirm or refute these claims."
        ),
    }
    return summaries.get(verdict, f"Analysis complete (score: {score}/100, confidence: {confidence}%).")


def _ml_description(score: int, features: dict) -> str:
    confidence_pct = round((features.get("confidence") or 0.0) * 100)
    if score < 25:
        level = "high confidence this is credible content"
    elif score < 50:
        level = "this content appears probably credible"
    elif score < 75:
        level = "moderate concern about this content's credibility"
    else:
        level = "high likelihood this is misinformation"
    detail = f" (model confidence: {confidence_pct}%)" if confidence_pct else ""
    return f"The neural classifier indicates {level}{detail}."


def _ling_description(score: int, flags: list) -> str:
    if score < 20:
        base = "Language analysis finds no significant credibility concerns."
    elif score < 40:
        base = "Language analysis detects minor rhetorical concerns."
    elif score < 60:
        base = "Language analysis finds moderate use of aggressive or manipulative rhetoric."
    elif score < 80:
        base = "Language analysis detects significant manipulative language patterns."
    else:
        base = "Language analysis finds severe inflammatory or deceptive rhetoric."
    if flags:
        base += f" Key signals: {', '.join(flags[:3])}."
    return base


def _ext_description(score: int, features: dict) -> str:
    n_claims   = features.get("google_fc_claims_found", 0)
    has_false  = features.get("google_fc_has_false_rating", False)
    mbfc       = features.get("mbfc_rating", "")
    rep        = features.get("reputation_score")

    parts = []
    if n_claims > 0:
        qualifier = "with false/misleading ratings" if has_false else "found"
        parts.append(f"{n_claims} fact-check(s) {qualifier}")
    else:
        parts.append("No matching fact-checks found")

    if mbfc:
        parts.append(f"source credibility rating: {mbfc}")
    elif rep is not None:
        parts.append(f"source trust score: {round(rep * 100)}%")

    return ". ".join(parts) + "."


def _bias_explanation(bias_score: int, features: dict) -> str:
    if bias_score > 50:
        direction = "strongly positive or emotional"
    elif bias_score > 20:
        direction = "moderately positive"
    elif bias_score > -20:
        direction = "relatively neutral"
    elif bias_score > -50:
        direction = "moderately negative or critical"
    else:
        direction = "strongly negative or alarmist"

    neg = round((features.get("vader_neg") or 0.0) * 100)
    pos = round((features.get("vader_pos") or 0.0) * 100)
    return (
        f"Sentiment analysis indicates a {direction} tone (bias score: {bias_score:+d}). "
        f"Positive content: {pos}%, negative content: {neg}%."
    )


_FEATURE_LABELS = {
    "ml_score":         "AI classifier",
    "linguistic_score": "linguistic pattern analysis",
    "external_score":   "external fact-checking",
    "vader_compound":   "sentiment tone",
    "vader_neg":        "negative sentiment",
    "vader_pos":        "positive sentiment",
    "caps_ratio":       "excessive capitalisation",
    "exclaim_density":  "exclamation mark density",
    "source_reputation":"source reputation",
    "has_known_source": "known source",
    "title_caps_ratio": "title capitalisation",
    "title_exclaim":    "title punctuation",
    "quote_count":      "use of quotations",
    "word_count":       "article length",
    "avg_sentence_len": "sentence length",
}


def _context(r: dict) -> str:
    doc        = r.get("document", {})
    title      = doc.get("title",      "")
    source     = doc.get("source",     "")
    word_count = doc.get("word_count", 0)
    top_drivers = r.get("explanation", {}).get("top_drivers", [])

    parts = []
    if title:
        parts.append(f'"{title}"')
    if source:
        parts.append(f"published by {source}")
    if word_count:
        parts.append(f"{word_count} words")
    if top_drivers:
        raw_feat  = top_drivers[0].get("feature", "")
        human     = _FEATURE_LABELS.get(raw_feat, raw_feat.replace("_", " "))
        parts.append(f"primary driver: {human}")

    return ("Analysis context: " + ", ".join(parts) + ".") if parts else "No contextual metadata available."


def _missing(r: dict) -> str:
    ext_features = r.get("signals", {}).get("external", {}).get("features", {})
    doc          = r.get("document", {})

    gaps = []
    if not ext_features.get("google_fc_claims_found"):
        gaps.append(
            "this article has not been reviewed by any indexed fact-checking organisation "
            "(PolitiFact, Snopes, AFP, etc.) — absence of a fact-check does not imply the content is false"
        )
    if not doc.get("source"):
        gaps.append("source domain could not be identified, so publisher reputation was not assessed")
    if not doc.get("title"):
        gaps.append("article title unavailable")
    if ext_features.get("reputation_source") == "unknown":
        gaps.append("publisher reputation data is unavailable for this domain")

    return ("Verification gaps: " + "; ".join(gaps) + ".") if gaps else "All primary verification sources were consulted."


def _detail(explanation: dict) -> str:
    top_drivers = explanation.get("top_drivers", [])
    if not top_drivers:
        return "No detailed SHAP explanation available."

    lines = ["Key factors driving this assessment:"]
    for driver in top_drivers[:5]:
        feat      = driver.get("feature", "").replace("_", " ")
        shap_val  = driver.get("shap", 0.0)
        direction = "toward credible" if shap_val < 0 else "toward fake"
        lines.append(f"  \u2022 {feat}: {direction} ({shap_val:+.3f})")

    base_val = explanation.get("base_value")
    if base_val is not None:
        lines.append(f"Model baseline: {base_val:.3f}")

    return "\n".join(lines)
