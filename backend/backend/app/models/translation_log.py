from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from ..database import Base

class TranslationLog(Base):
    __tablename__ = "translation_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    source_text = Column(String, nullable=False)
    translated_text = Column(String, nullable=False)
    source_lang = Column(String, nullable=False)
    target_lang = Column(String, nullable=False)

    user = relationship("User")
