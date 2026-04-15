"""
pipeline/db.py
==============
SQLAlchemy model for persisting pipeline results + save_to_postgres helper.

Table: analyses
  id               TEXT  PRIMARY KEY   MD5(input text)
  verdict          TEXT  NOT NULL
  credibility_score REAL NOT NULL
  confidence       REAL NOT NULL
  source_domain    TEXT
  signals_json     TEXT               JSON blob of per-signal details
  explanation_json TEXT               JSON blob of SHAP top-drivers
  created_at       TIMESTAMPTZ        server default now()

save_to_postgres() is non-fatal: any DB error is swallowed so an outage
never breaks the inference path.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session

log = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://politicheck:secret@localhost:5432/politicheck",
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,          # detect stale connections
    pool_size=5,
    max_overflow=10,
)


class Base(DeclarativeBase):
    pass


class Analysis(Base):
    __tablename__ = "analyses"

    id                = Column(String(32), primary_key=True)   # MD5 hex
    verdict           = Column(String(16), nullable=False)
    credibility_score = Column(Float,      nullable=False)
    confidence        = Column(Float,      nullable=False)
    source_domain     = Column(String(255))
    signals_json      = Column(Text)
    explanation_json  = Column(Text)
    created_at        = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


# Create table on import (idempotent — skipped if it already exists)
try:
    Base.metadata.create_all(engine)
except Exception as _e:
    log.warning("Could not create analyses table: %s", _e)


def save_to_postgres(result: dict) -> None:
    """
    Upsert *result* into the analyses table.

    Expects the dict returned by run_pipeline():
      {verdict, credibility_score, confidence, document: {source}, signals, explanation, input_text}
    """
    try:
        input_text = result.get("input_text", "")
        row_id = hashlib.md5(
            input_text.strip().encode("utf-8"), usedforsecurity=False
        ).hexdigest()

        row = Analysis(
            id=row_id,
            verdict=result.get("verdict", ""),
            credibility_score=float(result.get("credibility_score", 0.5)),
            confidence=float(result.get("confidence", 0.0)),
            source_domain=result.get("document", {}).get("source", ""),
            signals_json=json.dumps(result.get("signals", {})),
            explanation_json=json.dumps(result.get("explanation", {})),
        )
        with Session(engine) as session:
            session.merge(row)   # INSERT … ON CONFLICT DO UPDATE
            session.commit()
    except Exception as exc:
        log.warning("save_to_postgres failed (non-fatal): %s", exc)
