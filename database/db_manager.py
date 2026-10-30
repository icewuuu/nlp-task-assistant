import sqlite3
from datetime import datetime
from typing import Optional

from config import DB_PATH
from database.models import Task, TaskPriority, TaskStatus, TaskLang


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _get_conn() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS tasks (
                                                    id TEXT PRIMARY KEY,
                                                    description TEXT NOT NULL,
                                                    relation TEXT,
                                                    status TEXT NOT NULL,
                                                    priority TEXT NOT NULL,
                                                    created_at TEXT NOT NULL,
                                                    updated_at TEXT NOT NULL,
                                                    lang TEXT NOT NULL
               )"""
        )


def add_task(
    description: str,
    relation: Optional[str] = None,
    priority: TaskPriority = TaskPriority.NORMAL,
    lang: TaskLang = TaskLang.PL,
) -> Task:
    task = Task(
        description=description, relation=relation, priority=priority, lang=lang
    )

    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task.id,
                task.description,
                task.relation,
                task.status,
                task.priority,
                task.created_at,
                task.updated_at,
                task.lang,
            ),
        )
    return task


def get_all_tasks() -> list[Task]:
    with _get_conn() as conn:
        cursor = conn.execute("SELECT * FROM tasks ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [Task(**dict(row)) for row in rows]


def get_task_by_id(task_id: str) -> Optional[Task]:
    with _get_conn() as conn:
        cursor = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        if row:
            return Task(**dict(row))
        return None


def update_task(
    task_id: str,
    description: Optional[str] = None,
    status: Optional[TaskStatus] = None,
    relation: Optional[str] = None,
) -> Optional[Task]:
    task = get_task_by_id(task_id)
    if not task:
        return None

    if description is not None:
        task.description = description
    if status is not None:
        task.status = status
    if relation is not None:
        # pusty string / "none" czyści relację
        task.relation = relation or None

    task.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with _get_conn() as conn:
        conn.execute(
            "UPDATE tasks SET description = ?, status = ?, relation = ?, updated_at = ? WHERE id = ?",
            (task.description, task.status, task.relation, task.updated_at, task.id),
        )
    return task


def delete_task(task_id: str) -> bool:
    with _get_conn() as conn:
        cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        return cursor.rowcount > 0


if __name__ == "__main__":
    # Test bazy danych (uruchom przez `uv run python -m database.db_manager`)
    init_db()
    nowe_zadanie = add_task("Przetestować bazę danych")
    print(f"Dodano zadanie: {nowe_zadanie.id}")
    print(f"Liczba zadań w bazie: {len(get_all_tasks())}")
