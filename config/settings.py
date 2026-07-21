from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

HARDCODED_POLICY_NO = os.getenv("HARDCODED_POLICY_NO")
HARDCODED_ACCESS_TOKEN = os.getenv("HARDCODED_ACCESS_TOKEN")
POLICY_DOCUMENT_API_URL = os.getenv("POLICY_DOCUMENT_API_URL")

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./vector_store/chroma_data")

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 8000))
DEBUG = os.getenv("DEBUG", "True") == "True"