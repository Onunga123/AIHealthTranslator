# app/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os

# ----------------------------
# Database setup
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'translator.db')}"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ----------------------------
# Dependency: Get DB session
# ----------------------------
def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ----------------------------
# Initialize DB (create tables)
# ----------------------------
def init_db():
    import app.models.user  # noqa: F401 (ensure models are registered)
    import app.models.dictionary_term  # noqa: F401
    Base.metadata.create_all(bind=engine)


# ----------------------------
# Helper: Get dictionary entry
# ----------------------------
def get_dictionary_entry(term_en: str):
    """Quick dictionary lookup without circular imports"""
    from app.models.dictionary_term import DictionaryTerm  # local import
    db = SessionLocal()
    try:
        return (
            db.query(DictionaryTerm)
            .filter(DictionaryTerm.term_en == term_en.lower())
            .first()
        )
    finally:
        db.close()
