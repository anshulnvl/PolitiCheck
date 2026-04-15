from sqlalchemy import (
    Column, Integer, String, Float, Text, Boolean,
    DateTime, ForeignKey, JSON
)
from sqlalchemy.sql import func
from backend.pipeline.db import Base

class User(Base):
    __tablename__ = "users"
    id             = Column(Integer, primary_key=True)
    email          = Column(String, unique=True, nullable=False)
    hashed_password= Column(String, nullable=False)
    created_at     = Column(DateTime, server_default=func.now())
    plan           = Column(String, default="free")

class Analysis(Base):
    __tablename__ = "analyses"
    id              = Column(Integer, primary_key=True)
    content_hash    = Column(String, unique=True, index=True)
    input_text      = Column(Text, nullable=False)
    input_type      = Column(String)          # e.g. "text", "url", "image"
    verdict         = Column(String)          # e.g. "likely_false", "unverified"
    credibility_score = Column(Float)
    confidence      = Column(Float)
    ml_score        = Column(Float)
    linguistic_score= Column(Float)
    external_score  = Column(Float)
    full_result_json= Column(JSON)
    created_at      = Column(DateTime, server_default=func.now())
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=True)

class Feedback(Base):
    __tablename__ = "feedback"
    id               = Column(Integer, primary_key=True)
    analysis_id      = Column(Integer, ForeignKey("analyses.id"))
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=True)
    disagreed_verdict= Column(String)
    correct_verdict  = Column(String)
    notes            = Column(Text)
    reviewed         = Column(Boolean, default=False)
    created_at       = Column(DateTime, server_default=func.now())

class FactCheckCache(Base):
    __tablename__ = "fact_check_cache"
    id         = Column(Integer, primary_key=True)
    claim_text = Column(Text)
    claim_hash = Column(String, unique=True, index=True)
    source_api = Column(String)
    verdict    = Column(String)
    url        = Column(String)
    fetched_at = Column(DateTime, server_default=func.now())
    expires_at = Column(DateTime)

class SourceReputation(Base):
    __tablename__ = "source_reputation"
    id              = Column(Integer, primary_key=True)
    domain          = Column(String, unique=True, nullable=False, index=True)
    newsguard_score = Column(Integer)        # 0–100
    mbfc_rating     = Column(String)         # HIGH, MIXED, LOW, SATIRE
    domain_age_days = Column(Integer)
    alexa_rank      = Column(Integer)
    last_updated    = Column(DateTime, server_default=func.now())