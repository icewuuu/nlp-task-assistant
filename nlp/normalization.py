import unicodedata
import re


def normalize_text(text: str) -> str:
    """
    Normalizacja tekstu.
    """
    # Konwersja do małych liter
    text = text.lower()

    # Normalizacja Unicode do postaci NFKD
    # (Rozbija znaki z ogonkami na znak podstawowy + modyfikator, np. 'ą' -> 'a' + '˛')
    text = unicodedata.normalize("NFKD", text)

    # Usuwanie znaków diakrytycznych (modyfikatorów)
    text = "".join([c for c in text if not unicodedata.combining(c)])

    # Ujednolicenie i normalizacja białych znaków
    text = re.sub(r"\s+", " ", text).strip()

    return text
