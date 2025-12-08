from dotenv import load_dotenv
import os

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "fallback_secret_key")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ai_health_translator.db")
