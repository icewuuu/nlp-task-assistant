from database.models import Task, TaskStatus, TaskPriority

STATUS_TEXT = {
    TaskStatus.TODO: "DO ZROBIENIA",
    TaskStatus.IN_PROGRESS: "W TRAKCIE",
    TaskStatus.DONE: "ZAKOŃCZONE",
}

PRIORITY_TEXT = {
    TaskPriority.LOW: "Niski",
    TaskPriority.NORMAL: "Normalny",
    TaskPriority.HIGH: "Wysoki",
}


def format_task(task: Task) -> str:
    """Formatuje pojedyncze zadanie do wysłania na Telegramie (HTML)."""

    status_text = STATUS_TEXT.get(task.status, "NIEZNANY")
    priority_text = PRIORITY_TEXT.get(task.priority, "NIEZNANY")

    msg = (
        f"<b>Zadanie:</b> {task.description}\n"
        f"<b>Status:</b> {status_text}\n"
        f"<b>Priorytet:</b> {priority_text}\n"
        f"<b>ID:</b> <code>{task.id}</code>\n"
        f"<b>Utworzono:</b> {task.created_at}"
    )
    if task.relation:
        msg += f"\n<b>Powiązane z:</b> <code>{task.relation}</code>"

    return msg


def format_task_list(tasks: list[Task]) -> str:
    """Formatuje całą listę zadań."""

    if not tasks:
        return "Brak zadań w bazie danych."

    lines = ["<b>Zestawienie zadań:</b>\n"]
    for i, task in enumerate(tasks, 1):
        status_text = STATUS_TEXT.get(task.status, "NIEZNANY")
        lines.append(
            f"{i}. [{status_text}] <b>{task.description}</b>\n   ID: <code>{task.id}</code>"
        )

    return "\n".join(lines)


def format_nlp_result(operation_name: str, result, original_text: str) -> str:
    """Uniwersalna funkcja do formatowania wyników z modułu NLP."""

    msg = f"<b>Operacja:</b> {operation_name}\n"
    msg += f"<b>Oryginał:</b> {original_text}\n\n"

    if isinstance(result, str):
        msg += f"<b>Wynik:</b> {result}"

    elif isinstance(result, dict) and "explanation" in result:
        msg += f"<b>Wektor:</b>\n<code>{result['vector']}</code>\n\n"
        msg += f"{result['explanation']}"

    elif isinstance(result, dict):
        for key, val in result.items():
            msg += f"<b>{key}:</b> {val}\n"

    elif isinstance(result, list):
        msg += "<b>Wyniki analizy:</b>\n"
        for item in result:
            msg += f"- {item}\n"

    else:
        msg += f"<b>Wynik:</b> {result}"

    return msg
