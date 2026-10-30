from nltk.tokenize import word_tokenize
from nlp.loader import nlp_spacy


def tokenize_compare(text: str) -> dict[str, list[str]]:
    """
    Przeprowadza tokenizację tekstu dwoma metodami (NLTK i spaCy)
    i zwraca słownik pozwalający na łatwe porównanie wyników w bocie.
    """

    try:
        nltk_tokens = word_tokenize(text, language="polish")
    except Exception:
        nltk_tokens = word_tokenize(text)

    doc = nlp_spacy(text)
    spacy_tokens = [token.text for token in doc]

    return {"NLTK": nltk_tokens, "spaCy": spacy_tokens}
