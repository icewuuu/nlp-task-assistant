from sklearn.feature_extraction.text import CountVectorizer


def get_bow(task_text: str, corpus: list[str]) -> dict:
    """
    Tworzy reprezentację Bag of Words.
    Korpus to lista wszystkich zadań z bazy danych.
    """

    if not corpus:
        corpus = [task_text]
    if task_text not in corpus:
        corpus.append(task_text)

    vectorizer = CountVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=100,
        binary=False,
    )

    vectorizer.fit(corpus)
    vector = vectorizer.transform([task_text])

    feature_names = vectorizer.get_feature_names_out()
    dense_vector = vector.toarray()[0]

    result_vector = {
        feature_names[i]: int(dense_vector[i])
        for i in range(len(feature_names))
        if dense_vector[i] > 0
    }

    # Gotowe wyjaśnienie do wysłania w wiadomości
    explanation = (
        "Parametry Bag of Words:\n"
        "• lowercase=True: sprowadza tekst do małych liter\n"
        "• ngram_range=(1, 2): zlicza pojedyncze słowa oraz zbitki 2 słów\n"
        "• max_features=100: słownik ograniczony do 100 najczęstszych fraz w bazie\n"
        "• binary=False: zlicza, ILE razy słowo wystąpiło, a nie tylko CZY wystąpiło"
    )

    return {"vector": result_vector, "explanation": explanation}
