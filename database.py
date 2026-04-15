import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://politicheck:secret@localhost:5432/politicheck"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,     # Detect stale connections
    pool_size=5,            # Keep 5 connections in pool
    max_overflow=10,        # Allow 10 overflow connections max
    pool_recycle=3600,      # Recycle connections after 1 hour
)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()