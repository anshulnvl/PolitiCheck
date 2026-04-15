"""
aggressive_linguistic_signal.py
================================
Linguistic signal module for fake news detection, focused on detecting
aggressive, inflammatory, and manipulative language patterns.

Score: 0.0 = calm/neutral/credible  →  1.0 = highly aggressive/manipulative
"""

import re
import math
import spacy
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

# ---------------------------------------------------------------------------
# NLP setup
# ---------------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")
vader = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Lexicons & Pattern Banks
# ---------------------------------------------------------------------------

# ── Threat / Violence language ──────────────────────────────────────────────
THREAT_PATTERNS: List[str] = [
    r"\b(kill|murder|destroy|eliminate|annihilate|eradicate|wipe out)\b",
    r"\b(attack|assault|bomb|shoot|stab|beat|burn)\b",
    r"\b(crush|smash|obliterate|take down|take out)\b",
    r"\b(die|death|dead|blood|slaughter|massacre|genocide)\b",
    r"\b(punish|hang|execute|lock (them|him|her|you) up)\b",
    r"\b(threat(en)?|intimidate|coerce|force)\b",
]

# ── Dehumanisation language ─────────────────────────────────────────────────
DEHUMANISATION_PATTERNS: List[str] = [
    r"\b(animal(s)?|beast(s)?|vermin|pest(s)?|parasite(s)?|rat(s)?|cockroach(es)?)\b",
    r"\b(monster(s)?|demon(s)?|devil(s)?|evil incarnate)\b",
    r"\b(infestation|plague|cancer|disease|infection)\b",  # used metaphorically
    r"\b(invasion|invader(s)?|horde|swarm|flood(ing)?)\b",  # dehumanising migration framing
    r"\b(subhuman|inhuman|less than human|not (even )?human)\b",
    r"\b(filth|scum|trash|garbage|dirt|savage(s)?|barbarian(s)?)\b",
]

# ── Us-vs-Them / Tribal framing ─────────────────────────────────────────────
US_VS_THEM_PATTERNS: List[str] = [
    r"\b(enemy|enemies|traitor(s)?|treason|betrayal|fifth column)\b",
    r"\b(puppet(s)?|shill(s)?|paid (agent|actor)|plant(s)?)\b",
    r"\b(globalist(s)?|elitist(s)?|establishment|deep state|cabal)\b",
    r"\b(wake up|sheeple|blind(ed)?|brainwash(ed)?|asleep)\b",
    r"\b(real (american|patriot|citizen)|true (patriot|believer|christian))\b",
    r"\b(them|they|those people|their kind|these people)\b",  # vague othering
]

# ── Incitement / Call-to-action aggression ───────────────────────────────────
INCITEMENT_PATTERNS: List[str] = [
    r"\b(fight back|stand up|rise up|resist|revolt|resist(ance)?|rebellion)\b",
    r"\b(take action|do something|enough is enough|time (to act|for action))\b",
    r"\b(don't let them|stop them|we must|we need to|you must|you need to)\b",
    r"\b(share this|spread the word|tell everyone|warn (your|others))\b",
    r"\b(now or never|last chance|before it's too late|urgent action)\b",
]

# ── Absolutist / Black-and-white thinking ────────────────────────────────────
ABSOLUTIST_WORDS: List[str] = [
    "always", "never", "every", "all", "none", "no one", "everyone",
    "totally", "completely", "absolutely", "entirely", "utterly",
    "impossible", "certain", "definitely", "obviously", "clearly",
    "undeniable", "irrefutable", "proven", "fact", "truth",
    "worst", "best", "greatest", "most evil", "pure evil",
]

# ── Fear-mongering language ──────────────────────────────────────────────────
FEAR_PATTERNS: List[str] = [
    r"\b(crisis|catastrophe|disaster|apocalypse|collapse|meltdown)\b",
    r"\b(terrifying|horrifying|alarming|shocking|devastating|dire)\b",
    r"\b(threat|danger|risk|peril|hazard|menace)\b",
    r"\b(panic|fear|terror|dread|nightmare|horror)\b",
    r"\b(unprecedented|never before|never seen|end of|downfall)\b",
    r"\b(survival|survive|extinction|existential)\b",
]

# ── Conspiracy / Epistemic aggression ────────────────────────────────────────
CONSPIRACY_PATTERNS: List[str] = [
    r"\b(cover(ed)?[ -]up|cover[ -]up|coverup)\b",
    r"\b(they (don't|won't) tell you|mainstream media|fake news|msm)\b",
    r"\b(what (they|he|she|the government) (hides?|doesn't want you to know))\b",
    r"\b(truth(er)?|red[ -]pill(ed)?|wake up|open your eyes)\b",
    r"\b(agenda|plot|scheme|conspiracy|orchestrated|planned|engineered)\b",
    r"\b(false flag|staged|crisis actor|hoax|fake|fabricated)\b",
]

# ── Pejorative personal attacks ─────────────────────────────────────────────
PERSONAL_ATTACK_PATTERNS: List[str] = [
    r"\b(idiot(s)?|moron(s)?|imbecile(s)?|stupid|dumb|brainless)\b",
    r"\b(liar(s)?|pathological liar|compulsive liar|fraud|con man|snake)\b",
    r"\b(corrupt|criminal|crook|thief|pedophile|pervert|sicko)\b",
    r"\b(loser(s)?|failure|worthless|deplorable|disgusting|despicable)\b",
    r"\b(hypocrite(s)?|fake|phony|charlatan|opportunist)\b",
]

# ── Rhetorical aggression markers ───────────────────────────────────────────
RHETORICAL_AGGRESSION: List[str] = [
    r"\b(wake up people|open your eyes|think about it|connect the dots)\b",
    r"\b(ask yourself|ask (why|how)|follow the money)\b",
    r"\b(this is (insane|madness|outrageous|unacceptable|unbelievable))\b",
    r"\b(I can't believe|how (dare|can) (they|he|she|anyone))\b",
    r"\b(disgusting|reprehensible|vile|nauseating|sickening)\b",
]

# ── Compile all patterns ─────────────────────────────────────────────────────
_COMPILED: Dict[str, List[re.Pattern]] = {
    "threat":            [re.compile(p, re.IGNORECASE) for p in THREAT_PATTERNS],
    "dehumanisation":    [re.compile(p, re.IGNORECASE) for p in DEHUMANISATION_PATTERNS],
    "us_vs_them":        [re.compile(p, re.IGNORECASE) for p in US_VS_THEM_PATTERNS],
    "incitement":        [re.compile(p, re.IGNORECASE) for p in INCITEMENT_PATTERNS],
    "fear_mongering":    [re.compile(p, re.IGNORECASE) for p in FEAR_PATTERNS],
    "conspiracy":        [re.compile(p, re.IGNORECASE) for p in CONSPIRACY_PATTERNS],
    "personal_attack":   [re.compile(p, re.IGNORECASE) for p in PERSONAL_ATTACK_PATTERNS],
    "rhetorical_aggr":   [re.compile(p, re.IGNORECASE) for p in RHETORICAL_AGGRESSION],
}

_ABSOLUTIST_RE = re.compile(
    r'\b(' + '|'.join(re.escape(w) for w in ABSOLUTIST_WORDS) + r')\b',
    re.IGNORECASE
)

_ALLCAPS_RE     = re.compile(r'\b[A-Z]{2,}\b')
_EXCLAIM_RE     = re.compile(r'!')
_QUESTION_RE    = re.compile(r'\?')
_ELLIPSIS_RE    = re.compile(r'\.{2,}|…')
_MULTI_PUNCT_RE = re.compile(r'[!?]{2,}')
_NUMBER_RE      = re.compile(r'\b\d+\.?\d*%?\b')
_QUOTE_RE       = re.compile(r'["\u201c\u201d\u2018\u2019\'`]')
_PASSIVE_RE     = re.compile(
    r'\b(am|is|are|was|were|be|been|being)\s+\w+ed\b', re.IGNORECASE
)
_FIRST_PERSON_RE = re.compile(
    r'\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b', re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class AggressiveSignalResult:
    """
    score        : 0.0 (neutral/credible) → 1.0 (highly aggressive)
    category_scores : per-category raw hit counts & normalised subscores
    features     : all computed features
    flags        : human-readable list of triggered signals
    error        : populated only on exception
    """
    score: float
    category_scores: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, Any]          = field(default_factory=dict)
    flags: List[str]                  = field(default_factory=list)
    error: str                        = None


# ---------------------------------------------------------------------------
# Pattern matching helpers
# ---------------------------------------------------------------------------

def _count_pattern_group(text: str, patterns: List[re.Pattern]) -> Tuple[int, List[str]]:
    """Return (total_match_count, list_of_matched_strings)."""
    total, matched = 0, []
    for pat in patterns:
        hits = pat.findall(text)
        if hits:
            total += len(hits)
            matched.extend(hits if isinstance(hits[0], str) else [h[0] for h in hits])
    return total, matched


def _normalise(count: int, length: int, scale: float = 200.0) -> float:
    """Soft-normalise a raw count by text length and a scaling factor."""
    if length == 0:
        return 0.0
    return min(1.0, (count / length) * scale)


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute(text: str, title: str = "") -> AggressiveSignalResult:
    """
    Compute an aggression/fake-news linguistic signal score.

    Parameters
    ----------
    text  : article body
    title : optional headline (prepended for pattern matching)

    Returns
    -------
    AggressiveSignalResult
    """
    if not text or len(text.strip()) < 30:
        return AggressiveSignalResult(score=0.5, error="Text too short for analysis")

    try:
        full_text = f"{title} {text}" if title else text
        lower     = full_text.lower()
        word_len  = max(len(full_text.split()), 1)

        # spaCy parse (capped for performance)
        doc       = nlp(full_text[:120_000])
        sentences = list(doc.sents)
        sent_count = max(len(sentences), 1)
        tokens    = [t for t in doc if t.is_alpha]
        tok_count = max(len(tokens), 1)

        features: Dict[str, Any] = {}
        flags:    List[str]      = []
        cat_scores: Dict[str, float] = {}

        # ── 1. Pattern category scores ───────────────────────────────────────
        for cat, patterns in _COMPILED.items():
            count, matched = _count_pattern_group(full_text, patterns)
            norm_score     = _normalise(count, word_len, scale=150.0)
            features[f"{cat}_count"]   = count
            features[f"{cat}_matches"] = list(set(matched))[:10]  # top 10 unique
            cat_scores[cat]            = round(norm_score, 4)
            if count > 0:
                flags.append(f"{cat.upper()} ({count} hits): {', '.join(set(matched))[:80]}")

        # ── 2. Absolutist / black-white language ─────────────────────────────
        abs_hits = _ABSOLUTIST_RE.findall(full_text)
        features["absolutist_count"] = len(abs_hits)
        features["absolutist_words"] = list(set(w.lower() for w in abs_hits))[:10]
        cat_scores["absolutist"]     = round(_normalise(len(abs_hits), word_len, 100.0), 4)

        # ── 3. ALL-CAPS word ratio ────────────────────────────────────────────
        caps_words = _ALLCAPS_RE.findall(full_text)
        features["allcaps_ratio"] = len(caps_words) / tok_count
        if features["allcaps_ratio"] > 0.05:
            flags.append(f"ALLCAPS ratio {features['allcaps_ratio']:.2%}")

        # ── 4. Punctuation aggression ─────────────────────────────────────────
        exclaim_count = len(_EXCLAIM_RE.findall(text))
        multi_punct   = len(_MULTI_PUNCT_RE.findall(text))
        features["exclamation_density"]    = round(exclaim_count / max(len(text), 1) * 100, 4)
        features["multi_punct_count"]      = multi_punct
        features["ellipsis_count"]         = len(_ELLIPSIS_RE.findall(text))
        if multi_punct > 0:
            flags.append(f"Multi-punctuation (!!, ??) found: {multi_punct}×")

        # ── 5. VADER sentiment (full + title-only) ────────────────────────────
        vader_full  = vader.polarity_scores(full_text)
        vader_title = vader.polarity_scores(title) if title else {"compound": 0.0}
        features["vader_compound"]       = vader_full["compound"]
        features["vader_neg"]            = vader_full["neg"]
        features["vader_pos"]            = vader_full["pos"]
        features["vader_title_compound"] = vader_title["compound"]
        features["sentiment_extremity"]  = abs(vader_full["compound"])  # 0–1
        if features["sentiment_extremity"] > 0.6:
            flags.append(f"Extreme sentiment: compound={vader_full['compound']:.2f}")

        # ── 6. Sentence-level aggression density ──────────────────────────────
        # Count sentences that contain ≥1 aggressive pattern hit
        agg_sentences = 0
        all_agg_patterns = [p for plist in _COMPILED.values() for p in plist]
        for sent in sentences:
            if any(pat.search(sent.text) for pat in all_agg_patterns):
                agg_sentences += 1
        features["aggressive_sentence_ratio"] = round(agg_sentences / sent_count, 4)
        cat_scores["sent_aggression"]         = features["aggressive_sentence_ratio"]

        # ── 7. Named entity density (high = more factual grounding) ──────────
        features["ner_density"] = round((len(doc.ents) / max(len(doc), 1)) * 100, 4)

        # ── 8. Quote density (sourcing signal) ────────────────────────────────
        quote_count = len(_QUOTE_RE.findall(text))
        features["quote_density"] = round((quote_count / max(len(text), 1)) * 100, 4)

        # ── 9. Number / statistic density (factual grounding) ─────────────────
        number_count = len(_NUMBER_RE.findall(full_text))
        features["number_count"] = number_count

        # ── 10. Reading level ─────────────────────────────────────────────────
        features["fk_grade"] = textstat.flesch_kincaid_grade(text)

        # ── 11. Sentence length variance (erratic = emotional) ────────────────
        sent_lens = [
            len([t for t in s if t.is_alpha]) for s in sentences
        ]
        if len(sent_lens) > 1:
            mean_sl = sum(sent_lens) / len(sent_lens)
            variance = sum((l - mean_sl) ** 2 for l in sent_lens) / len(sent_lens)
            features["sentence_length_variance"] = round(variance, 2)
        else:
            features["sentence_length_variance"] = 0.0

        # ── 12. Passive voice ratio ───────────────────────────────────────────
        passive_sents = sum(1 for s in sentences if _PASSIVE_RE.search(s.text))
        features["passive_voice_ratio"] = round(passive_sents / sent_count, 4)

        # ── 13. First-person pronoun ratio (opinion/subjective) ───────────────
        fp_count = len(_FIRST_PERSON_RE.findall(full_text))
        features["first_person_ratio"] = round(fp_count / tok_count, 4)

        # ── 14. Rhetorical question density ──────────────────────────────────
        rq_count = sum(
            1 for s in sentences
            if _QUESTION_RE.search(s.text) and not any(
                c.dep_ == "nsubj" for c in nlp(s.text)
            )
        )
        features["rhetorical_question_count"] = rq_count

        # ── 15. Title-body sentiment divergence ───────────────────────────────
        if title:
            body_vader = vader.polarity_scores(text)
            features["title_body_sentiment_gap"] = round(
                abs(vader_title["compound"] - body_vader["compound"]), 4
            )
        else:
            features["title_body_sentiment_gap"] = 0.0

        # ── Composite score ───────────────────────────────────────────────────
        score = _aggregate_score(features, cat_scores)

        return AggressiveSignalResult(
            score=score,
            category_scores=cat_scores,
            features=features,
            flags=flags,
        )

    except Exception as exc:
        return AggressiveSignalResult(score=0.5, error=str(exc))


# ---------------------------------------------------------------------------
# Scoring aggregator
# ---------------------------------------------------------------------------

def _aggregate_score(f: Dict[str, Any], cat: Dict[str, float]) -> float:
    """
    Weighted aggregation of sub-signals into a final 0–1 score.

    Positive weights → push toward aggressive (fake)
    Negative weights → push toward credible (real)
    """

    # ── Category weights (tuned heuristically; retrain with labeled data) ──
    CATEGORY_WEIGHTS = {
        "threat":          0.20,
        "dehumanisation":  0.18,
        "us_vs_them":      0.10,
        "incitement":      0.14,
        "fear_mongering":  0.08,
        "conspiracy":      0.14,
        "personal_attack": 0.12,
        "rhetorical_aggr": 0.08,
        "absolutist":      0.06,
        "sent_aggression": 0.10,
    }

    weighted_cat = sum(
        cat.get(k, 0.0) * w for k, w in CATEGORY_WEIGHTS.items()
    )

    # ── Surface feature adjustments ───────────────────────────────────────
    surface = 0.0
    surface += min(f["allcaps_ratio"] * 4.0,      0.30)   # SHOUTING
    surface += min(f["exclamation_density"] * 8.0, 0.20)  # !!!
    surface += min(f["multi_punct_count"] * 0.05,  0.15)  # !? combinations
    surface += min(f["sentiment_extremity"] * 0.4, 0.25)  # emotional extremity
    surface += min(f["first_person_ratio"] * 1.5,  0.10)  # subjective
    surface += min(f["title_body_sentiment_gap"] * 0.3, 0.15)  # bait-and-switch

    # ── Credibility discounts (negative adjustments) ──────────────────────
    discount = 0.0
    discount += min(f["ner_density"] * 0.04, 0.20)        # named entities = factual
    discount += min(f["quote_density"] * 1.5, 0.15)       # quotations = sourced
    discount += min(f["number_count"] * 0.015, 0.15)      # statistics = factual
    if f["fk_grade"] >= 10:
        discount += 0.10                                   # complex prose = journalistic

    # ── Sentence structure penalty ────────────────────────────────────────
    var_penalty = min(math.log1p(f["sentence_length_variance"]) * 0.01, 0.10)

    raw = weighted_cat + surface - discount + var_penalty

    # Sigmoid-like soft clamp to [0, 1]
    clamped = max(0.0, min(1.0, raw))
    return round(clamped, 4)


# ---------------------------------------------------------------------------
# Convenience: human-readable report
# ---------------------------------------------------------------------------

def explain(result: AggressiveSignalResult) -> str:
    """Return a plain-text summary of the result."""
    if result.error:
        return f"[ERROR] {result.error}"

    lines = [
        f"Aggression Score : {result.score:.4f}  ({'HIGH' if result.score > 0.6 else 'MEDIUM' if result.score > 0.35 else 'LOW'})",
        "",
        "Category Scores:",
    ]
    for cat, val in sorted(result.category_scores.items(), key=lambda x: -x[1]):
        bar = "█" * int(val * 20)
        lines.append(f"  {cat:<22} {val:.3f}  {bar}")

    if result.flags:
        lines += ["", "Triggered Flags:"]
        for flag in result.flags:
            lines.append(f"  • {flag}")

    lines += [
        "",
        "Key Features:",
        f"  VADER compound        : {result.features.get('vader_compound', 0):.3f}",
        f"  Allcaps ratio         : {result.features.get('allcaps_ratio', 0):.3f}",
        f"  Aggressive sent ratio : {result.features.get('aggressive_sentence_ratio', 0):.3f}",
        f"  NER density           : {result.features.get('ner_density', 0):.2f}",
        f"  FK grade              : {result.features.get('fk_grade', 0):.1f}",
        f"  Title-body sentiment Δ: {result.features.get('title_body_sentiment_gap', 0):.3f}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     SAMPLE_TITLE = "BREAKING: They Are DESTROYING Our Country — Wake Up Before It's Too Late!!!"
#     SAMPLE_TEXT  = """
#     These monsters are flooding our borders and the traitors in government are letting them.
#     They want to REPLACE you. It's a coordinated plot by globalist elites who fund the media.
#     Mainstream media won't tell you this — they are paid actors covering up the TRUTH.
#     We must fight back NOW before it's too late. Rise up! Share this everywhere!!!
#     Every single one of these criminals needs to be locked up. They are vermin, plain and simple.
#     Ask yourself: why does nobody talk about this? Connect the dots. The evidence is undeniable.
#     This is the worst attack on our civilization in history. You are either with us or against us.
#     """

#     result = compute(SAMPLE_TEXT, SAMPLE_TITLE)
#     print(explain(result))