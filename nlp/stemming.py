from nltk.tokenize import word_tokenize
from nlp.loader import polish_stemmer


def stem_text(text: str) -> list[tuple[str, str]]:
    """
    Wykonuje stemming tekstu.
    Wykorzystuje NLTK do podziału na słowa oraz Pystempel do analizy rdzeni.
    """
    try:
        tokens = word_tokenize(text, language="polish")
    except Exception:
        tokens = word_tokenize(text)

    stemmed_words = []

    for token in tokens:
        if token.isalnum():
            stem = polish_stemmer.stem(token) or token
            stemmed_words.append((token, stem))

    return stemmed_words
