import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
DB_PATH = os.getenv("DB_PATH", "data/jobpilot.db")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
