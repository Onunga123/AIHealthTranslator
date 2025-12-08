from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DATABASE_URL = "sqlite:///./health_translator.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    # Import models so they are registered with Base before create_all
    from .models import user, dictionary_term, translation_log  # noqa: F401
    Base.metadata.create_all(bind=engine)

    # Seed a default admin if not present
    from sqlalchemy.orm import Session
    from .models.user import User
    from .security import get_password_hash

    with Session(engine) as db:
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            admin = User(
                username="admin",
                email="admin@example.com",
                hashed_password=get_password_hash("admin123"),
                role="admin",
            )
            db.add(admin)
            db.commit()


# -----------------------------
# NEW: Dictionary lookup helper
# -----------------------------
from .models.dictionary_term import DictionaryTerm  # adjust if your model name is different

def get_dictionary_entry(text: str, language: str) -> DictionaryTerm | None:
    """
    Look up a term in the dictionary table for a specific language.
    Returns None if not found.
    """
    db = SessionLocal()
    try:
        entry = db.query(DictionaryTerm).filter(
            DictionaryTerm.text.ilike(text),
            DictionaryTerm.language.ilike(language)
        ).first()
        return entry
    finally:
        db.close()
