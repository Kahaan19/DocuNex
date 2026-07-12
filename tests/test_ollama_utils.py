"""Tests for the Ollama helpers, using fake HTTP responses (no server needed)."""
import json

import requests

import ollama_utils


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, lines=None):
        self.status_code = status_code
        self._payload = payload or {}
        self._lines = lines or []

    def json(self):
        return self._payload

    def iter_lines(self):
        for line in self._lines:
            yield line


def test_connection_ok_when_model_present(monkeypatch):
    def fake_get(url, timeout):
        return _FakeResponse(payload={"models": [{"name": "gemma:2b"}]})

    monkeypatch.setattr(requests, "get", fake_get)
    ok, msg = ollama_utils.check_ollama_connection(model="gemma:2b")
    assert ok is True
    assert "connected" in msg.lower()


def test_connection_reports_missing_model(monkeypatch):
    def fake_get(url, timeout):
        return _FakeResponse(payload={"models": [{"name": "other"}]})

    monkeypatch.setattr(requests, "get", fake_get)
    ok, msg = ollama_utils.check_ollama_connection(model="gemma:2b")
    assert ok is False
    assert "not found" in msg.lower()


def test_connection_handles_connection_error(monkeypatch):
    def fake_get(url, timeout):
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr(requests, "get", fake_get)
    ok, msg = ollama_utils.check_ollama_connection()
    assert ok is False
    assert "cannot connect" in msg.lower()


def test_stream_accumulates_chunks(monkeypatch):
    lines = [
        json.dumps({"response": "Hello"}).encode(),
        json.dumps({"response": " world"}).encode(),
        json.dumps({"response": "", "done": True}).encode(),
    ]

    def fake_post(url, json, stream, timeout):
        return _FakeResponse(lines=lines)

    monkeypatch.setattr(requests, "post", fake_post)
    outputs = list(ollama_utils.stream_ollama_response("hi"))
    # Streaming yields the progressively-accumulated text.
    assert outputs[-1] == "Hello world"


def test_stream_reports_bad_status(monkeypatch):
    def fake_post(url, json, stream, timeout):
        return _FakeResponse(status_code=500)

    monkeypatch.setattr(requests, "post", fake_post)
    outputs = list(ollama_utils.stream_ollama_response("hi"))
    assert any("500" in o for o in outputs)
