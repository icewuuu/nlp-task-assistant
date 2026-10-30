from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf(task_text: str, corpus: list[str]) -> dict:
    """
    Tworzy reprezentację TF-IDF.
    Korpus to lista wszystkich zadań z bazy danych.
    """
    if not corpus:
        corpus = [task_text]
    if task_text not in corpus:
        corpus.append(task_text)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=100,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    )

    # 2. Fit i transform
    vectorizer.fit(corpus)
    vector = vectorizer.transform([task_text])

    feature_names = vectorizer.get_feature_names_out()
    dense_vector = vector.toarray()[0]

    result_vector = {
        feature_names[i]: round(float(dense_vector[i]), 4)
        for i in range(len(feature_names))
        if dense_vector[i] > 0
    }

    explanation = (
        "Parametry TF-IDF:\n"
        "• use_idf=True: osłabia wagę słów pospolitych, a podbija rzadkich\n"
        "• smooth_idf=True: dodaje wygładzanie zapobiegające błędom przy rzadkich słowach\n"
        "• sublinear_tf=False: klasyczne, liniowe zliczanie term frequency\n"
        "• (oraz lowercase=True i ngram_range=(1, 2) jak w BoW)"
    )

    return {"vector": result_vector, "explanation": explanation}
