"""Tests for logging configuration."""

import logging

import nanochat.common.logging as logging_mod
from nanochat.common.logging import setup_default_logging


def setup_function():
    logging_mod._logging_initialized = False
    logging.root.handlers.clear()


def test_setup_adds_handler():
    setup_default_logging()
    assert len(logging.root.handlers) >= 1


def test_setup_idempotent():
    setup_default_logging()
    count_after_first = len(logging.root.handlers)
    setup_default_logging()
    assert len(logging.root.handlers) == count_after_first
