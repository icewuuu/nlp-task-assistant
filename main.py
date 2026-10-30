import logging
from telegram.ext import ApplicationBuilder, CommandHandler

from config import TELEGRAM_TOKEN
from database.db_manager import init_db
from bot.handlers import (
    start_command,
    add_command,
    tasks_command,
    update_command,
    delete_command,
    task_command,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    """Główna funkcja uruchamiająca bota."""

    logger.info("Inicjalizacja bazy danych...")
    init_db()

    if not TELEGRAM_TOKEN:
        logger.error("Brak tokenu Telegrama! Ustaw TELEGRAM_TOKEN w pliku .env.")
        return

    logger.info("Budowanie aplikacji Telegram...")
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Rejestracja komend
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("add", add_command))
    application.add_handler(CommandHandler("tasks", tasks_command))
    application.add_handler(CommandHandler("update", update_command))
    application.add_handler(CommandHandler("delete", delete_command))
    application.add_handler(CommandHandler("task", task_command))

    logger.info("Uruchamianie bota...")
    application.run_polling()

if __name__ == "__main__":
    main()
