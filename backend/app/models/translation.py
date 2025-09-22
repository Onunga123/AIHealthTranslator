from sqlalchemy import Column, Integer, String, Text
from ..database import Base


class Translation(Base):
    __tablename__ = "translations"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    target_language = Column(String, nullable=False)
    translated_text = Column(Text, nullable=True)
