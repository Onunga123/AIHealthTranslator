from sqlalchemy import Column, Integer, String, Text
from ..database import Base

class Speech(Base):
    __tablename__ = "speech_logs"

    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(Text, nullable=True)
    output_text = Column(Text, nullable=True)
    audio_file = Column(Text, nullable=True)  # base64 encoded
