from .database import Base, engine
from .models import User, Language, Translation, Speech

Base.metadata.create_all(bind=engine)
print("✅ All tables created successfully!")
