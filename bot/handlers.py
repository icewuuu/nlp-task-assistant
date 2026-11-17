from telegram import Update
from telegram.ext import ContextTypes

from database import db_manager
from database.models import TaskStatus
from bot import formatters
from config import INDIVIDUAL_NGRAM

from nlp import (
    clean_text,
    normalize_text,
    tokenize_compare,
    remove_stopwords,
    lemmatize_text,
    stem_text,
    generate_ngrams,
    get_bow,
    get_tfidf,
)
from nlp.classify_data import (
    get_labeled_corpus,
    label_todo_tasks,
    load_ag_news,
)
from nlp.classify_train import (
    CLASS_METHODS,
    REPR_METHODS,
    build_all_matrices_report,
    evaluate_model,
    predict_text,
    train_all_configs,
)


def _chunk_text(text: str, max_len: int = 3500) -> list[str]:
    """Telegram ucina długie wiadomości, więc dzielę na kawałki."""
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.splitlines(keepends=True):
        if current_len + len(line) > max_len and current:
            chunks.append("".join(current))
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += len(line)
    if current:
        chunks.append("".join(current))
    return chunks

CLEAR_RELATION_VALUES = {"none", "null", "clear", "-", "usunąć", "usuń"}

VALIDATION_SENTENCES = [
    ("Olympic athletes win gold medals in swimming finals", "Sports"),
    ("Central bank raises interest rates amid inflation fears", "Business"),
    ("Astronomers discover a new exoplanet near a red dwarf", "Sci/Tech"),
    ("Diplomats meet to discuss ceasefire in the region", "World"),
]


def _parse_add_args(args: list[str]) -> tuple[str, str | None]:
    """Parsuje /add <opis> [--relation <id>]."""
    if "--relation" not in args:
        return " ".join(args), None

    idx = args.index("--relation")
    if idx + 1 >= len(args):
        raise ValueError("Po --relation podaj ID zadania nadrzędnego.")

    relation = args[idx + 1]
    description_parts = args[:idx] + args[idx + 2 :]
    if not description_parts:
        raise ValueError("Podaj treść zadania oraz opcjonalnie --relation <id>.")

    return " ".join(description_parts), relation


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Obsługa komendy /start"""

    help_text = (
        "Witaj w systemie TODO z silnikiem NLP.\n\n"
        "Komendy TODO:\n"
        "/add <opis> [--relation <id>] - dodaje nowe zadanie\n"
        "/tasks - wyświetla listę zadań\n"
        "/update <id> <nowy_opis> - aktualizuje treść zadania\n"
        "/update <id> status <TODO|IN_PROGRESS|DONE> - zmienia status\n"
        "/update <id> relation <id|none> - ustawia/czyści relację\n"
        "/delete <id> - usuwa zadanie\n"
        f"/task <id> <operacja> - uruchamia analizę NLP na zadaniu\n\n"
        "Operacje NLP: clean, normalize, tokenize, stopwords, "
        f"lemmatize, stem, unigram, bigram, trigram, 6gram, bow, tfidf\n\n"
        "Klasyfikacja:\n"
        "/prepare_data_for_classification [agnews|todo]\n"
        "/classifier cleaning \"tekst\" <bow|tfidf|embedding> <logreg|mlp>\n"
        "/evaluate <repr> <method>  (albo: all)\n"
        "/validate [repr] [method]\n"
        "/label <id> <klasa>"
    )
    await update.message.reply_text(help_text)


async def add_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Obsługa komendy /add <opis> [--relation <id>]"""

    if not context.args:
        await update.message.reply_text(
            "Podaj treść zadania, np.:\n"
            "/add Przeanalizować dane\n"
            "/add Napisać raport --relation 550e8400-e29b-41d4-a716-446655440000"
        )
        return

    try:
        description, relation = _parse_add_args(context.args)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return

    if relation is not None:
        parent = db_manager.get_task_by_id(relation)
        if not parent:
            await update.message.reply_text(
                f"Nie znaleziono zadania nadrzędnego o ID: {relation}"
            )
            return

    task = db_manager.add_task(description=description, relation=relation)
    await update.message.reply_html(formatters.format_task(task))


async def tasks_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Obsługa komendy /tasks"""

    tasks = db_manager.get_all_tasks()
    await update.message.reply_html(formatters.format_task_list(tasks))


async def update_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /update <id> <nowy_opis>
    /update <id> status <TODO|IN_PROGRESS|DONE>
    /update <id> relation <id|none>
    """

    usage = (
        "Niepoprawny format. Użyj:\n"
        "/update <id> <nowy_opis>\n"
        "/update <id> status <TODO|IN_PROGRESS|DONE>\n"
        "/update <id> relation <id|none>"
    )

    if len(context.args) < 2:
        await update.message.reply_text(usage)
        return

    task_id = context.args[0]
    field = context.args[1].lower()

    updated_task = None

    if field == "status":
        if len(context.args) < 3:
            await update.message.reply_text(
                "Podaj status, np.: /update <id> status DONE\n"
                "Dostępne: TODO, IN_PROGRESS, DONE"
            )
            return
        try:
            status_enum = TaskStatus(context.args[2].upper())
        except ValueError:
            await update.message.reply_text(
                "Nieprawidłowy status. Dostępne: TODO, IN_PROGRESS, DONE"
            )
            return
        updated_task = db_manager.update_task(task_id, status=status_enum)

    elif field == "relation":
        if len(context.args) < 3:
            await update.message.reply_text(
                "Podaj ID zadania nadrzędnego lub none, np.:\n"
                "/update <id> relation <parent_id>\n"
                "/update <id> relation none"
            )
            return

        relation_raw = context.args[2]
        if relation_raw.lower() in CLEAR_RELATION_VALUES:
            relation_value = ""
        else:
            if relation_raw == task_id:
                await update.message.reply_text(
                    "Zadanie nie może być powiązane samo ze sobą."
                )
                return
            parent = db_manager.get_task_by_id(relation_raw)
            if not parent:
                await update.message.reply_text(
                    f"Nie znaleziono zadania nadrzędnego o ID: {relation_raw}"
                )
                return
            relation_value = relation_raw

        updated_task = db_manager.update_task(task_id, relation=relation_value)

    else:
        # Domyślnie: aktualizacja opisu (również kompatybilnie ze starym
        # /update <id> <STATUS>, gdy podano wyłącznie status).
        if len(context.args) == 2:
            try:
                status_enum = TaskStatus(context.args[1].upper())
            except ValueError:
                status_enum = None

            if status_enum is not None:
                updated_task = db_manager.update_task(task_id, status=status_enum)
            else:
                updated_task = db_manager.update_task(
                    task_id, description=" ".join(context.args[1:])
                )
        else:
            updated_task = db_manager.update_task(
                task_id, description=" ".join(context.args[1:])
            )

    if updated_task:
        await update.message.reply_html(
            f"<b>Zaktualizowano zadanie:</b>\n{formatters.format_task(updated_task)}"
        )
    else:
        await update.message.reply_text("Nie znaleziono zadania o podanym ID.")


async def delete_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Obsługa komendy /delete <id>"""

    if not context.args:
        await update.message.reply_text(
            "Podaj ID zadania do usunięcia, np.: /delete 1234-abcd"
        )
        return

    task_id = context.args[0]
    success = db_manager.delete_task(task_id)

    if success:
        await update.message.reply_text(f"Zadanie {task_id} zostało usunięte.")
    else:
        await update.message.reply_text("Nie znaleziono zadania o podanym ID.")


async def task_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Obsługa komendy /task <id> <operacja>"""
    if len(context.args) < 2:
        await update.message.reply_text(
            "Niepoprawny format. Użyj: /task <id_zadania> <operacja>\n"
            f"Np. /task 1234-abcd tokens  lub  /task 1234-abcd 6gram"
        )
        return

    task_id = context.args[0]
    operation = context.args[1].lower()

    task = db_manager.get_task_by_id(task_id)
    if not task:
        await update.message.reply_text("Nie znaleziono zadania o podanym ID.")
        return

    original_text = task.description
    result = None

    if operation == "clean":
        result = clean_text(original_text)
    elif operation == "normalize":
        result = normalize_text(original_text)
    elif operation in ["tokenize", "tokens"]:
        result = tokenize_compare(original_text)
    elif operation == "stopwords":
        result = remove_stopwords(original_text)
    elif operation == "lemmatize":
        result = lemmatize_text(original_text)
    elif operation == "stem":
        result = stem_text(original_text)
    elif operation == "unigram":
        result = generate_ngrams(original_text, 1)
    elif operation == "bigram":
        result = generate_ngrams(original_text, 2)
    elif operation == "trigram":
        result = generate_ngrams(original_text, 3)
    elif operation in ["xgram", "6gram"]:
        result = generate_ngrams(original_text, INDIVIDUAL_NGRAM)
    elif operation in ["bow", "tfidf"]:
        all_tasks = db_manager.get_all_tasks()
        corpus = [t.description for t in all_tasks]

        if operation == "bow":
            result = get_bow(original_text, corpus)
        elif operation == "tfidf":
            result = get_tfidf(original_text, corpus)
    else:
        await update.message.reply_text(f"Nieznana operacja NLP: {operation}")
        return

    response_msg = formatters.format_nlp_result(operation, result, original_text)
    await update.message.reply_html(response_msg)


def _parse_classifier_args(args: list[str]) -> tuple[str, str, str]:
    """
    /classifier cleaning <tekst...> <method_repr> <method_class>
    Tekst może być w cudzysłowach Telegram i tak dzieli po spacjach.
    """
    if len(args) < 4:
        raise ValueError(
            "Użyj: /classifier cleaning \"tekst\" <bow|tfidf|embedding> <logreg|mlp>"
        )

    if args[0].lower() != "cleaning":
        raise ValueError(
            "Parametr cleaning jest obowiązkowy, np.:\n"
            "/classifier cleaning \"Stock markets rose\" tfidf mlp"
        )

    method_repr = args[-2].lower()
    method_class = args[-1].lower()
    text = " ".join(args[1:-2]).strip().strip('"').strip("'")

    if not text:
        raise ValueError("Podaj niepusty tekst do sklasyfikowania.")
    if method_repr not in REPR_METHODS:
        raise ValueError(f"Nieznana reprezentacja. Dostępne: {', '.join(REPR_METHODS)}")
    if method_class not in CLASS_METHODS:
        raise ValueError(f"Nieznana metoda. Dostępne: {', '.join(CLASS_METHODS)}")

    return text, method_repr, method_class


async def prepare_data_for_classification_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """/prepare_data_for_classification [agnews|todo]"""
    source = "agnews"
    if context.args:
        source = context.args[0].lower().replace("-", "").replace("_", "")
        if source in {"agnews", "ag", "agnew"}:
            source = "agnews"
        elif source in {"todo", "tasks", "local"}:
            source = "todo"
        else:
            await update.message.reply_text(
                "Nie znam tego źródła. Użyj agnews albo todo."
            )
            return

    await update.message.reply_text(
        f"Biorę dane ({source}) i trenuję 6 modeli, chwila..."
    )

    try:
        if source == "agnews":
            texts, labels = load_ag_news()
            classes = sorted(set(labels))
            info = (
                f"AG News: {len(texts)} przykładów. "
                f"Classes: {', '.join(classes)}"
            )
        else:
            summary = label_todo_tasks(overwrite=True)
            texts, labels = get_labeled_corpus()
            if len(texts) < 4:
                await update.message.reply_text(
                    "Za mało oznaczonych zadań. Dodaj więcej (/add) "
                    "albo odpal agnews."
                )
                return
            info = (
                f"Labels assigned to {summary['labeled']} tasks. "
                f"Classes: {', '.join(summary['classes'])}"
            )

        report = train_all_configs(texts, labels, clean=True)
        await update.message.reply_text(info)
        await update.message.reply_text(report["table"])
        # telegram ma limit długości wiadomości
        for i, chunk in enumerate(_chunk_text(report["matrices"]), start=1):
            header = f"(macierze cz. {i})\n" if i > 1 else ""
            await update.message.reply_text(header + chunk)
    except Exception as exc:
        await update.message.reply_text(f"Błąd przygotowania/treningu: {exc}")


async def classifier_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/classifier cleaning \"tekst\" <repr> <method>"""
    try:
        text, method_repr, method_class = _parse_classifier_args(context.args or [])
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return

    try:
        result = predict_text(
            text, method_repr, method_class, use_cleaning=True
        )
        await update.message.reply_text(formatters.format_prediction(result))
    except FileNotFoundError as exc:
        await update.message.reply_text(str(exc))
    except Exception as exc:
        await update.message.reply_text(f"Błąd klasyfikacji: {exc}")


async def evaluate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/evaluate <repr> <method>  albo  /evaluate all"""
    if not context.args:
        await update.message.reply_text(
            "Użyj:\n"
            "/evaluate <bow|tfidf|embedding> <logreg|mlp>\n"
            "/evaluate all"
        )
        return

    if context.args[0].lower() == "all":
        try:
            report = build_all_matrices_report()
            for i, chunk in enumerate(_chunk_text(report), start=1):
                header = f"(macierze cz. {i})\n" if i > 1 else ""
                await update.message.reply_text(header + chunk)
        except FileNotFoundError as exc:
            await update.message.reply_text(str(exc))
        except Exception as exc:
            await update.message.reply_text(f"Błąd ewaluacji: {exc}")
        return

    if len(context.args) < 2:
        await update.message.reply_text(
            "Użyj: /evaluate <bow|tfidf|embedding> <logreg|mlp>\n"
            "lub: /evaluate all"
        )
        return

    method_repr = context.args[0].lower()
    method_class = context.args[1].lower()

    if method_repr not in REPR_METHODS or method_class not in CLASS_METHODS:
        await update.message.reply_text(
            f"Dostępne repr: {', '.join(REPR_METHODS)}\n"
            f"Dostępne method: {', '.join(CLASS_METHODS)}\n"
            "Zbiorczo: /evaluate all"
        )
        return

    try:
        metrics = evaluate_model(method_repr, method_class)
        await update.message.reply_html(
            formatters.format_evaluation(method_repr, method_class, metrics)
        )
    except FileNotFoundError as exc:
        await update.message.reply_text(str(exc))
    except Exception as exc:
        await update.message.reply_text(f"Błąd ewaluacji: {exc}")


async def validate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/validate [repr] [method] – test na moich zdaniach spoza zbioru"""
    method_repr = "tfidf"
    method_class = "logreg"
    if len(context.args) >= 2:
        method_repr = context.args[0].lower()
        method_class = context.args[1].lower()

    lines = [
        f"Walidacja spoza zbioru | {method_repr} + {method_class}",
        "",
    ]
    ok = 0
    try:
        for sentence, expected in VALIDATION_SENTENCES:
            result = predict_text(
                sentence, method_repr, method_class, use_cleaning=True
            )
            predicted = result["label"]
            hit = predicted == expected
            if hit:
                ok += 1
            mark = "OK" if hit else "MISS"
            p = result.get("probability")
            p_txt = f" p={p:.2f}" if p is not None else ""
            lines.append(
                f"[{mark}] expected={expected} pred={predicted}{p_txt}\n"
                f"  \"{sentence}\""
            )

        lines.append("")
        lines.append(
            f"Trafienia: {ok}/{len(VALIDATION_SENTENCES)} "
            "(zdania własne, nie z train)."
        )
        if ok == len(VALIDATION_SENTENCES):
            lines.append("Komentarz: na tych przykładach trafiliśmy wszystko.")
        elif ok >= 2:
            lines.append(
                "Komentarz: kilka pomyłek – przy krótkich zdaniach klasy "
                "się trochę mieszają (np. World vs Sci/Tech)."
            )
        else:
            lines.append(
                "Komentarz: słabo. Lepiej spróbować embedding albo inny classifier."
            )
        await update.message.reply_text("\n".join(lines))
    except FileNotFoundError as exc:
        await update.message.reply_text(str(exc))
    except Exception as exc:
        await update.message.reply_text(f"Błąd walidacji: {exc}")


async def label_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/label <id> <klasa> — ręczna etykieta zadania TODO."""
    if len(context.args) < 2:
        await update.message.reply_text(
            "Użyj: /label <id_zadania> <klasa>\n"
            "Np. /label 1234-abcd nauka"
        )
        return

    task_id = context.args[0]
    label = " ".join(context.args[1:]).strip()
    task = db_manager.set_task_label(task_id, label)
    if not task:
        await update.message.reply_text("Nie znaleziono zadania o podanym ID.")
        return

    await update.message.reply_text(
        f"Ustawiono label=\"{label}\" dla zadania:\n{task.description}"
    )
