import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("data/logs")
LOG_FILE = LOG_DIR / "backend.log"

def setup_logging(log_level: str = "INFO") -> None:
    """Configures both console and file logging with a readable format."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # 1. Console handler (readable colored output)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-25s | %(message)s",
        datefmt="%H:%M:%S"
    ))
    console.setLevel(log_level.upper())
    root.addHandler(console)

    # 2. File handler (persistent, rotating)
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-25s | %(message)s"
    ))
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    root.setLevel(logging.DEBUG)

    # Suppress noisy third-party logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("llama_index").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.info(f"Logging initialized. Writing to {LOG_FILE}")