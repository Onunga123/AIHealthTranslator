from fastapi import APIRouter, HTTPException, Body, Depends
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session

DATABASE_URL = "sqlite:///./health_translator.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy model
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class Word(Base):
    __tablename__ = "words"
    id = Column(Integer, primary_key=True, index=True)
    term_en = Column(String, index=True, nullable=False)
    term_sw = Column(String, index=True, nullable=False)
    term_luo = Column(String, index=True, nullable=False)

Base.metadata.create_all(bind=engine)

# Pydantic schemas
class WordCreate(BaseModel):
    term_en: str
    term_sw: str
    term_luo: str

class WordUpdate(BaseModel):
    term_en: Optional[str] = None
    term_sw: Optional[str] = None
    term_luo: Optional[str] = None

class WordOut(BaseModel):
    id: int
    term_en: str
    term_sw: str
    term_luo: str

    class Config:
        orm_mode = True

router = APIRouter()

@router.post("/add-word/", response_model=WordOut)
def add_word(payload: WordCreate = Body(...), db: Session = Depends(get_db)):
    word = Word(term_en=payload.term_en, term_sw=payload.term_sw, term_luo=payload.term_luo)
    db.add(word)
    db.commit()
    db.refresh(word)
    return word

@router.get("/lookup/{term}", response_model=WordOut)
def lookup_word(term: str, db: Session = Depends(get_db)):
    word = db.query(Word).filter(
        (Word.term_en == term) |
        (Word.term_sw == term) |
        (Word.term_luo == term)
    ).first()
    if not word:
        raise HTTPException(status_code=404, detail="Word not found")
    return word

@router.put("/update-word/{id}", response_model=WordOut)
def update_word(id: int, payload: WordUpdate = Body(...), db: Session = Depends(get_db)):
    word = db.query(Word).filter(Word.id == id).first()
    if not word:
        raise HTTPException(status_code=404, detail="Word not found")
    if payload.term_en is not None:
        word.term_en = payload.term_en
    if payload.term_sw is not None:
        word.term_sw = payload.term_sw
    if payload.term_luo is not None:
        word.term_luo = payload.term_luo
    db.commit()
    db.refresh(word)
    return word
