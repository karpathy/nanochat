
import pytest
from unittest.mock import patch
from nanochat.configurator import get_config

def test_get_config_defaults():
    # Provide defaults
    defaults = {"a": 1, "b": 2}
    with patch("sys.argv", ["script.py"]):
        config = get_config(defaults)
        assert config == {}

def test_get_config_cli_override():
    defaults = {"batch_size": 16}
    # Pass args explicitly or patch argv
    # get_config takes explicit argv list
    argv = ["--batch_size=32"]
    config = get_config(defaults, argv=argv)
    assert config["batch_size"] == 32

def test_get_config_cli_boolean():
    defaults = {"use_cache": False, "debug": True}
    argv = ["--use_cache=True", "--debug=False"]
    config = get_config(defaults, argv=argv)
    assert config["use_cache"] is True
    assert config["debug"] is False

def test_get_config_cli_float():
    defaults = {"lr": 0.01}
    argv = ["--lr=0.001"]
    config = get_config(defaults, argv=argv)
    assert config["lr"] == 0.001

def test_get_config_load_file(tmp_path):
    defaults = {"a": 1, "b": 2}
    config_file = tmp_path / "config.py"
    config_file.write_text("a = 10\nb = 20")

    argv = [str(config_file)]
    config = get_config(defaults, argv=argv)
    assert config["a"] == 10
    assert config["b"] == 20

def test_get_config_load_json(tmp_path):
    defaults = {"a": 1, "b": 2}
    config_file = tmp_path / "config.json"
    config_file.write_text('{"a": 100, "b": 200}')

    argv = [str(config_file)]
    config = get_config(defaults, argv=argv)
    assert config["a"] == 100
    assert config["b"] == 200

def test_get_config_tune_system_json_format(tmp_path):
    defaults = {"a": 1}
    config_file = tmp_path / "tune.json"
    config_file.write_text('{"parameters": {"a": 999}}')

    argv = [str(config_file)]
    config = get_config(defaults, argv=argv)
    assert config["a"] == 999

def test_get_config_unknown_key():
    defaults = {"a": 1}
    argv = ["--z=100"]
    with pytest.raises(ValueError, match="Unknown config key: z"):
        get_config(defaults, argv=argv)

def test_get_config_type_mismatch():
    defaults = {"a": 1} # int
    argv = ["--a=1.5"] # float
    with pytest.raises(AssertionError, match="Type mismatch for a"):
        get_config(defaults, argv=argv)
