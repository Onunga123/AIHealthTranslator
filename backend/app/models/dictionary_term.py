from sqlalchemy import Column, Integer, String, UniqueConstraint
from ..database import Base

class DictionaryTerm(Base):
    __tablename__ = "dictionary_terms"

    id = Column(Integer, primary_key=True, index=True)
    term_en = Column(String, index=True, nullable=False)
    term_sw = Column(String, nullable=True)   # Kiswahili
    term_luo = Column(String, nullable=True)  # Luo (Dholuo)

    __table_args__ = (
        UniqueConstraint("term_en", name="uq_term_en"),
    )
