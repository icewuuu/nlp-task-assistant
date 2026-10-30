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

CLEAR_RELATION_VALUES = {"none", "null", "clear", "-", "usunąć", "usuń"}


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
        "Dostępne komendy:\n"
        "/add <opis> [--relation <id>] - dodaje nowe zadanie\n"
        "/tasks - wyświetla listę zadań\n"
        "/update <id> <nowy_opis> - aktualizuje treść zadania\n"
        "/update <id> status <TODO|IN_PROGRESS|DONE> - zmienia status\n"
        "/update <id> relation <id|none> - ustawia/czyści relację\n"
        "/delete <id> - usuwa zadanie\n"
        f"/task <id> <operacja> - uruchamia analizę NLP na zadaniu\n\n"
        "Dostępne operacje NLP: clean, normalize, tokenize, stopwords, "
        f"lemmatize, stem, unigram, bigram, trigram, 6gram, bow, tfidf"
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
