import spacy
import nltk
from pystempel import Stemmer


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

print("[NLP] Ładowanie modelu spaCy (pl_core_news_sm)...")
try:
    nlp_spacy = spacy.load("pl_core_news_sm")
except OSError:
    raise OSError(
        "Nie znaleziono modelu spaCy. Uruchom w terminalu:\n"
        "uv pip install https://github.com/explosion/spacy-models/releases/download/pl_core_news_sm-3.8.0/pl_core_news_sm-3.8.0-py3-none-any.whl"
    )

print("[NLP] Wyodrębnianie polskiej listy stopwords...")
# Wbudowana w spaCy lista ponad 300 polskich słów stopu (np. "aby", "bez", "czy")
POLISH_STOPWORDS = nlp_spacy.Defaults.stop_words

print("[NLP] Ładowanie polskiego stemmera (Pystempel)...")
# Tworzy obiekt stemmera na podstawie domyślnego słownika
polish_stemmer = Stemmer.default()

print("[NLP] Modele załadowane pomyślnie!")
