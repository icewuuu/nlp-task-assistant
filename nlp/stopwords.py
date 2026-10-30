from nlp.loader import nlp_spacy, POLISH_STOPWORDS


def remove_stopwords(text: str) -> dict[str, str]:
    """
    Usuwa stopwords z tekstu przy użyciu polskiej listy ze spaCy.
    Zwraca słownik z oryginalnym tekstem i tekstem po filtracji.
    """

    doc = nlp_spacy(text)

    filtered_tokens = [
        token.text for token in doc if token.text.lower() not in POLISH_STOPWORDS
    ]

    filtered_text = " ".join(filtered_tokens)

    return {"original": text, "filtered": filtered_text}
