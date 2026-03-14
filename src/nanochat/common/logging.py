"""Logging configuration and colored formatter."""

import logging
import re


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        message = super().format(record)
        if levelname == "INFO":
            message = re.sub(r"(\d+\.?\d*\s*(?:GB|MB|%|docs))", rf"{self.BOLD}\1{self.RESET}", message)
            message = re.sub(r"(Shard \d+)", rf"{self.COLORS['INFO']}{self.BOLD}\1{self.RESET}", message)
        return message


_logging_initialized = False


def setup_default_logging():
    global _logging_initialized
    if _logging_initialized:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    _logging_initialized = True
