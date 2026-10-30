from nlp.loader import nlp_spacy


def lemmatize_text(text: str) -> list[tuple[str, str]]:
    """
    Przeprowadza lematyzację tekstu.
    Zwraca listę krotek w formacie (oryginalne_słowo, lemat).
    """
    doc = nlp_spacy(text)

    lemmas = [
        (token.text, token.lemma_)
        for token in doc
        if not token.is_space and not token.is_punct
    ]

    return lemmas
