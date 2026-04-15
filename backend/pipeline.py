from backend.signals.preprocessing  import process
from backend.signals.linguistic_signal import compute as linguistic_compute
from backend.signals.external_signal  import compute as external_compute
from backend.signals.ml_signal        import compute as ml_compute
import asyncio

async def run_pipeline(raw_input: str) -> dict:
    # Step 1: Clean + extract
    doc = process(raw_input)
    if doc.error:
        return {"error": doc.error}

    # Steps 2–4: Run signals in parallel
    linguistic, external, ml = await asyncio.gather(
        asyncio.to_thread(linguistic_compute, doc.body, doc.title),
        external_compute(doc.body, doc.source_domain),
        asyncio.to_thread(ml_compute, doc.body),
    )

    # Step 5: Aggregate (adjust weights to taste)
    final_score = (
        linguistic.score * 0.30 +
        external.score  * 0.40 +
        ml.score        * 0.30
    )

    return {
        "final_score": round(final_score, 4),
        "verdict": _verdict(final_score),
        "signals": {
            "linguistic": {"score": linguistic.score, "features": linguistic.features},
            "external":   {"score": external.score,   "features": external.features},
            "ml":         {"score": ml.score,          "features": ml.features},
        },
        "document": {
            "title": doc.title,
            "source": doc.source_domain,
            "word_count": doc.word_count,
            "language": doc.language,
        }
    }

def _verdict(score: float) -> str:
    if score < 0.35: return "LIKELY_REAL"
    if score < 0.60: return "UNCERTAIN"
    return "LIKELY_FAKE"