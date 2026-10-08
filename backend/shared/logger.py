import sys
from loguru import logger
from backend.shared.config import get_settings

settings = get_settings()


def setup_logger(service_name: str):

    # Supprimer le handler par défaut de loguru
    logger.remove()

    # ─── Console ────────────────────────────────
    logger.add(
        sys.stdout,
        level=settings.log_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>" + service_name + "</cyan> | "
            "<white>{message}</white>"
        ),
    )

    # ─── Fichier ────────────────────────────────
    logger.add(
        f"logs/{service_name}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )

    return logger