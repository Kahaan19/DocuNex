"""Tests for the centralized configuration module."""
import importlib


def test_defaults_present():
    import config

    assert config.OLLAMA_BASE_URL.startswith("http")
    assert config.OLLAMA_MODEL  # non-empty
    assert config.NEO4J_URI.startswith("bolt://")
    assert config.DEVICE in ("cpu", "cuda")


def test_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    monkeypatch.setenv("NEO4J_URI", "bolt://example:1234")
    import config

    importlib.reload(config)
    assert config.OLLAMA_MODEL == "llama3"
    assert config.NEO4J_URI == "bolt://example:1234"


def test_detect_device_falls_back_to_cpu_without_torch(monkeypatch):
    import config

    # Simulate torch import failure -> must not raise, must return 'cpu'.
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    assert config._detect_device() == "cpu"
