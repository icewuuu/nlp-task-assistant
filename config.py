import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DB_PATH = "tasks.db"
SPACY_MODEL_PL = "pl_core_news_sm"
DEFAULT_LANG = "pl"

INDIVIDUAL_NGRAM = 6
