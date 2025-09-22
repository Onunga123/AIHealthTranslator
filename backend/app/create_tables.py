from .database import Base, engine
from .models import User, Language, Translation, Speech, DictionaryTerm

Base.metadata.create_all(bind=engine)
print("✅ All tables created successfully!")
