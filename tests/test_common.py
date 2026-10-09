import io
import logging

import pytest

from nanochat.common import ColoredFormatter


@pytest.mark.parametrize("level", [logging.INFO, logging.WARNING])
@pytest.mark.parametrize("colored_first", [True, False])
def test_colored_formatter_keeps_file_logs_plain(level, colored_first, tmp_path):
    console_output = io.StringIO()
    console_handler = logging.StreamHandler(console_output)
    console_handler.setFormatter(ColoredFormatter("%(levelname)s: %(message)s"))
    log_path = tmp_path / "training.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger = logging.Logger("test", level=level)
    handlers = [console_handler, file_handler]
    if not colored_first:
        handlers.reverse()
    for handler in handlers:
        logger.addHandler(handler)

    try:
        logger.log(level, "Processed %d docs", 123)
    finally:
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)

    level_name = logging.getLevelName(level)
    assert log_path.read_text() == f"{level_name}: Processed 123 docs\n"
    assert console_output.getvalue().startswith(
        f"{ColoredFormatter.COLORS[level_name]}{ColoredFormatter.BOLD}{level_name}{ColoredFormatter.RESET}: "
    )
    if level == logging.INFO:
        assert f"{ColoredFormatter.BOLD}123 docs{ColoredFormatter.RESET}" in console_output.getvalue()
