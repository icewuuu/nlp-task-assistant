from nltk.tokenize import word_tokenize
from nltk.util import ngrams as nltk_ngrams


def generate_ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    """
    Generuje n-gramy dla podanego tekstu.
    """

    try:
        tokens = word_tokenize(text, language="polish")
    except Exception:
        tokens = word_tokenize(text)

    words = [word.lower() for word in tokens if word.isalnum()]

    # Zabezpieczenie przed zbyt krótkim tekstem
    if len(words) < n:
        return []

    # Wygenerowanie n-gramów
    return list(nltk_ngrams(words, n))
