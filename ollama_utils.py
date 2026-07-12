"""Helpers for talking to a local Ollama server.

Shared by the DocuNex entry points so the connection check and streaming
logic live in exactly one place.
"""
import json

import requests

import config


def check_ollama_connection(base_url: str = None, model: str = None):
    """Check that Ollama is reachable and the target model is available.

    Returns a ``(ok: bool, message: str)`` tuple.
    """
    base_url = base_url or config.OLLAMA_BASE_URL
    model = model or config.OLLAMA_MODEL
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            if model in model_names or any(model in name for name in model_names):
                return True, f"✅ Ollama connected. Available models: {', '.join(model_names[:3])}"
            return (
                False,
                f"❌ Model '{model}' not found. Available: {', '.join(model_names[:3])}\n"
                f"Run: ollama pull {model}",
            )
        return False, "❌ Ollama not responding"
    except requests.exceptions.ConnectionError:
        return (
            False,
            f"❌ Cannot connect to Ollama at {base_url}\n"
            "Make sure Ollama is running: ollama serve",
        )
    except requests.exceptions.Timeout:
        return False, "❌ Ollama connection timeout"
    except Exception as e:  # noqa: BLE001 - surface any failure to the UI
        return False, f"❌ Ollama connection failed: {str(e)}"


def stream_ollama_response(prompt: str, base_url: str = None, model: str = None):
    """Yield the incrementally-accumulated response from Ollama's generate API."""
    base_url = base_url or config.OLLAMA_BASE_URL
    model = model or config.OLLAMA_MODEL
    try:
        url = f"{base_url}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": True}
        response = requests.post(url, json=payload, stream=True, timeout=30)

        if response.status_code != 200:
            yield f"❌ Error: Ollama responded with status {response.status_code}"
            return

        full_response = ""
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "response" in data:
                full_response += data["response"]
                yield full_response
            if data.get("done", False):
                break
    except requests.exceptions.RequestException as e:
        yield f"❌ Connection error: {str(e)}"
    except Exception as e:  # noqa: BLE001
        yield f"❌ Unexpected error: {str(e)}"
