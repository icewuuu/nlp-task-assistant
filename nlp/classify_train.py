from __future__ import annotations

from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from config import MODELS_DIR, RANDOM_STATE
from nlp.cleaning import clean_text
from nlp.classify_metrics import (
    compute_metrics,
    format_all_matrices,
    format_comparison_table,
)

REPR_METHODS = ("bow", "tfidf", "embedding")
CLASS_METHODS = ("logreg", "mlp")

# krótkie opisy parametrów
REPR_EXPLANATIONS = {
    "bow": (
        "Bag of Words (CountVectorizer):\n"
        "• ngram_range=(1, 2): unigramy i bigramy\n"
        "• max_features=5000: limit słownika\n"
        "• min_df=2: pomijam rzadkie tokeny"
    ),
    "tfidf": (
        "TF-IDF (TfidfVectorizer):\n"
        "• ngramy / max_features / min_df jak w BoW\n"
        "• use_idf=True, smooth_idf=True\n"
        "• sublinear_tf=False"
    ),
    "embedding": (
        "Embedding (all-MiniLM-L6-v2):\n"
        "• gęsty wektor zdania (384 dim)\n"
        "• MultinomialNB odpada (ujemne wartości) -> logreg/mlp"
    ),
}

_embedding_model = None


def _ensure_models_dir() -> Path:
    path = Path(MODELS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _model_path(method_repr: str, method_class: str) -> Path:
    return _ensure_models_dir() / f"{method_repr}_{method_class}.joblib"


def _get_embedding_model():
    # ładuję raz, bo model jest ciężki
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _make_classifier(method_class: str):
    if method_class == "logreg":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    if method_class == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=300,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Nieznana metoda: {method_class}. Dostępne: {CLASS_METHODS}")


def _fit_transform_repr(method_repr: str, texts_train: list[str], texts_test: list[str]):
    """Fit na train, transform na train+test. Zwraca X_train, X_test, vectorizer."""
    if method_repr == "bow":
        vectorizer = CountVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=5000,
            min_df=2,
        )
        X_train = vectorizer.fit_transform(texts_train)
        X_test = vectorizer.transform(texts_test)
        return X_train, X_test, vectorizer

    if method_repr == "tfidf":
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=5000,
            min_df=2,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
        )
        X_train = vectorizer.fit_transform(texts_train)
        X_test = vectorizer.transform(texts_test)
        return X_train, X_test, vectorizer

    if method_repr == "embedding":
        model = _get_embedding_model()
        X_train = np.asarray(model.encode(texts_train, show_progress_bar=False))
        X_test = np.asarray(model.encode(texts_test, show_progress_bar=False))
        return X_train, X_test, None

    raise ValueError(f"Nieznana reprezentacja: {method_repr}. Użyj: {REPR_METHODS}")


def _transform_one(method_repr: str, text: str, vectorizer):
    """Jeden tekst -> wektor (ten sam sposób co przy treningu)."""
    if method_repr in ("bow", "tfidf"):
        return vectorizer.transform([text])
    if method_repr == "embedding":
        model = _get_embedding_model()
        return np.asarray(model.encode([text], show_progress_bar=False))
    raise ValueError(f"Nieznana reprezentacja: {method_repr}")


def train_model(
    texts: list[str],
    labels: list[str],
    method_repr: str,
    method_class: str,
    clean: bool = True,
) -> dict:
    """Trenuje jedną parę (repr, method), zapisuje do models/ i zwraca metryki."""
    method_repr = method_repr.lower()
    method_class = method_class.lower()

    if method_repr not in REPR_METHODS:
        raise ValueError(f"repr musi być jednym z: {REPR_METHODS}")
    if method_class not in CLASS_METHODS:
        raise ValueError(f"method musi być jednym z: {CLASS_METHODS}")
    if len(texts) < 4:
        raise ValueError("Za mało danych do treningu.")

    prepared = [clean_text(t) if clean else t for t in texts]

    # stratify wywala się, gdy klasa ma 1 przykład (np. małe TODO)
    counts = Counter(labels)
    min_count = min(counts.values()) if counts else 0
    stratify_labels = labels if min_count >= 2 else None

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        prepared,
        labels,
        test_size=0.2,
        stratify=stratify_labels,
        random_state=RANDOM_STATE,
    )

    X_train, X_test, vectorizer = _fit_transform_repr(
        method_repr, X_train_txt, X_test_txt
    )

    clf = _make_classifier(method_class)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = compute_metrics(y_test, y_pred.tolist())

    bundle = {
        "method_repr": method_repr,
        "method_class": method_class,
        "vectorizer": vectorizer,
        "classifier": clf,
        "classes": list(clf.classes_),
        "metrics": metrics,
        "clean": clean,
        "explanation": REPR_EXPLANATIONS[method_repr],
    }
    joblib.dump(bundle, _model_path(method_repr, method_class))
    return metrics


def train_all_configs(texts: list[str], labels: list[str], clean: bool = True) -> dict:
    """Odpalam wszystkie 6 wariantów (3 repr x 2 metody)."""
    results = []
    detailed = []
    for method_repr in REPR_METHODS:
        for method_class in CLASS_METHODS:
            metrics = train_model(texts, labels, method_repr, method_class, clean=clean)
            results.append(
                {
                    "repr": method_repr,
                    "method": method_class,
                    "ACC": metrics["ACC"],
                    "ERR": metrics["ERR"],
                    "F1_macro": metrics["macro"]["f1"],
                }
            )
            detailed.append(
                {
                    "repr": method_repr,
                    "method": method_class,
                    "metrics": metrics,
                }
            )

    table = format_comparison_table(results)
    matrices = format_all_matrices(detailed)

    models_dir = _ensure_models_dir()
    (models_dir / "comparison.txt").write_text(table, encoding="utf-8")
    (models_dir / "comparison_matrices.txt").write_text(matrices, encoding="utf-8")

    return {"table": table, "matrices": matrices}


def build_all_matrices_report() -> str:
    """Bierze metryki z już zapisanych .joblib (bez trenowania od zera)."""
    detailed = []
    missing = []
    for method_repr in REPR_METHODS:
        for method_class in CLASS_METHODS:
            path = _model_path(method_repr, method_class)
            if not path.exists():
                missing.append(f"{method_repr}_{method_class}")
                continue
            bundle = joblib.load(path)
            detailed.append(
                {
                    "repr": method_repr,
                    "method": method_class,
                    "metrics": bundle["metrics"],
                }
            )

    if missing and not detailed:
        raise FileNotFoundError(
            "Nie ma modeli. Najpierw /prepare_data_for_classification agnews"
        )

    report = format_all_matrices(detailed)
    if missing:
        report += "\n\nBrakuje: " + ", ".join(missing)
    return report


def load_model(method_repr: str, method_class: str) -> dict:
    path = _model_path(method_repr.lower(), method_class.lower())
    if not path.exists():
        raise FileNotFoundError(
            f"Brak pliku {path.name}. Odpal najpierw /prepare_data_for_classification."
        )
    return joblib.load(path)


def predict_text(
    text: str,
    method_repr: str,
    method_class: str,
    use_cleaning: bool = True,
) -> dict:
    """Klasa + prawdopodobieństwo dla jednego tekstu."""
    bundle = load_model(method_repr, method_class)
    prepared = clean_text(text) if use_cleaning else text
    X = _transform_one(bundle["method_repr"], prepared, bundle["vectorizer"])
    clf = bundle["classifier"]
    pred = str(clf.predict(X)[0])

    proba = None
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[0]
        classes = [str(c) for c in clf.classes_]
        class_index = classes.index(pred)
        proba = float(probs[class_index])

    return {
        "label": pred,
        "probability": proba,
        "method_repr": bundle["method_repr"],
        "method_class": bundle["method_class"],
        "explanation": bundle.get("explanation", ""),
    }


def evaluate_model(method_repr: str, method_class: str) -> dict:
    """Metryki zapisane przy ostatnim treningu."""
    bundle = load_model(method_repr, method_class)
    return bundle["metrics"]


def model_exists(method_repr: str, method_class: str) -> bool:
    return _model_path(method_repr.lower(), method_class.lower()).exists()
