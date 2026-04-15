"""
api/routers/feedback.py
=======================
POST /api/feedback — submit user corrections / ratings for an analysis.

Stores feedback in the `feedback` table (same PostgreSQL instance as analyses).
Non-fatal: DB errors are logged but never surface a 500 to the caller.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, String, Text

from backend.pipeline.db import Base, engine

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["feedback"])


# ── ORM model ─────────────────────────────────────────────────────────────────

class Feedback(Base):
    __tablename__ = "feedback"

    id             = Column(String(32),  primary_key=True)   # MD5 hex
    analysis_id    = Column(String(32),  index=True)         # FK to analyses.id (soft)
    user_verdict   = Column(String(16),  nullable=False)     # CREDIBLE|FAKE|MISLEADING|UNVERIFIABLE
    comment        = Column(Text)
    input_snippet  = Column(Text)                            # first 500 chars of submitted text
    created_at     = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


# Create table on import (idempotent — Base.metadata knows about Feedback now)
try:
    Base.metadata.create_all(engine)
except Exception as _e:
    log.warning("Could not create feedback table: %s", _e)


# ── Request / Response schemas ────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    analysis_id:   Optional[str] = None
    user_verdict:  str                    # CREDIBLE | FAKE | MISLEADING | UNVERIFIABLE
    comment:       Optional[str] = None
    input_snippet: Optional[str] = None   # optional excerpt of the text that was analysed


class FeedbackResponse(BaseModel):
    id:     str
    status: str


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/feedback", response_model=FeedbackResponse, summary="Submit credibility feedback")
def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    ts = datetime.now(timezone.utc).isoformat()
    row_id = hashlib.md5(
        f"{req.analysis_id or ''}{req.user_verdict}{ts}".encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()

    try:
        from sqlalchemy.orm import Session

        row = Feedback(
            id=row_id,
            analysis_id=req.analysis_id,
            user_verdict=req.user_verdict,
            comment=req.comment,
            input_snippet=(req.input_snippet or "")[:500] or None,
        )
        with Session(engine) as session:
            session.add(row)
            session.commit()
    except Exception as exc:
        log.warning("Failed to save feedback (non-fatal): %s", exc)

    return FeedbackResponse(id=row_id, status="ok")
