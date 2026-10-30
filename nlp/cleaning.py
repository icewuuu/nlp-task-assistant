import re


def clean_text(text: str) -> str:
    """
    Czyszczenie tekstu.
    """
    # Zamiana na małe litery
    text = text.lower()

    # Usuwanie liczb
    text = re.sub(r"\d+", "", text)

    # Usuwanie interpunkcji i znaków specjalnych
    text = re.sub(r"[^\w\s]", " ", text)

    # Usuwanie pojedynczych znaków
    text = re.sub(r"\b\w\b", "", text)

    # Usuwanie nadmiarowych białych znaków na samym końcu
    text = re.sub(r"\s+", " ", text).strip()

    return text
