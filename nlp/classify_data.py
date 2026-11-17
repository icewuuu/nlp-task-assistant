from collections import defaultdict

import numpy as np
from datasets import load_dataset

from database import db_manager
from config import AG_NEWS_PER_CLASS, RANDOM_STATE

# w sh0416/ag_news label jest od 1 do 4
AG_NEWS_LABELS = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}

# proste słowa kluczowe pod 3 klasy TODO
TODO_KEYWORDS = {
    "praca": [
        "raport",
        "prezentacj",
        "spotkani",
        "mail",
        "projekt",
        "klient",
        "biur",
        "deadline",
        "meeting",
        "praca",
    ],
    "dom": [
        "zakupy",
        "sprzątan",
        "pranie",
        "gotowan",
        "dom",
        "rodzin",
        "remont",
        "obiad",
        "kuchn",
    ],
    "nauka": [
        "nauka",
        "laborator",
        "wykład",
        "egzamin",
        "kolokwium",
        "nlp",
        "studia",
        "czytać",
        "czytac",
        "notatk",
        "kurs",
        "zadani",
    ],
}


def load_ag_news(n_per_class: int = AG_NEWS_PER_CLASS) -> tuple[list[str], list[str]]:
    """Bierze po n_per_class z każdej klasy. Tekst = title + description."""
    ds = load_dataset("sh0416/ag_news", split="train")

    by_label: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(ds):
        by_label[int(row["label"])].append(i)

    texts: list[str] = []
    labels: list[str] = []
    rng = np.random.RandomState(RANDOM_STATE)

    for label_id in sorted(by_label.keys()):
        indices = np.array(by_label[label_id])
        rng.shuffle(indices)
        chosen = indices[:n_per_class]
        class_name = AG_NEWS_LABELS.get(label_id, str(label_id))
        for idx in chosen:
            row = ds[int(idx)]
            text = f"{row['title']} {row['description']}".strip()
            texts.append(text)
            labels.append(class_name)

    return texts, labels


def guess_todo_label(description: str) -> str | None:
    """Proste liczenie trafień słów kluczowych."""
    text = description.lower()
    scores = {name: 0 for name in TODO_KEYWORDS}
    for name, words in TODO_KEYWORDS.items():
        for w in words:
            if w in text:
                scores[name] += 1

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return None
    return best


def label_todo_tasks(overwrite: bool = True) -> dict:
    """Ustawia label praca/dom/nauka na zadaniach w bazie."""
    tasks = db_manager.get_all_tasks()
    labeled = 0
    classes_used: set[str] = set()

    for task in tasks:
        if task.label and not overwrite:
            classes_used.add(task.label)
            continue

        label = guess_todo_label(task.description)
        if label is None:
            # jak nic nie pasuje, wrzucam do nauki
            label = "nauka"

        db_manager.set_task_label(task.id, label)
        labeled += 1
        classes_used.add(label)

    return {
        "labeled": labeled,
        "classes": sorted(classes_used),
        "total_tasks": len(tasks),
    }


def get_labeled_corpus() -> tuple[list[str], list[str]]:
    """Teksty + etykiety z SQLite."""
    tasks = db_manager.get_labeled_tasks()
    texts = [t.description for t in tasks]
    labels = [t.label for t in tasks]
    return texts, labels
