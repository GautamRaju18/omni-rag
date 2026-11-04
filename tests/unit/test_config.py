from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnirag.core.config import Settings


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIRAG_GENERATION_MODEL", "qwen3:1.7b")
    monkeypatch.setenv("OMNIRAG_OLLAMA_TIMEOUT_S", "30")
    s = Settings(_env_file=None)
    assert s.generation_model == "qwen3:1.7b"
    assert s.ollama_timeout_s == 30.0


def test_password_is_never_rendered(settings: Settings) -> None:
    assert "test-password" not in repr(settings)
    assert "test-password" not in settings.model_dump_json()
    assert settings.neo4j_password.get_secret_value() == "test-password"


def test_required_models_follow_configured_models(settings: Settings) -> None:
    assert settings.required_ollama_models == ["qwen3:4b-instruct", "bge-m3"]


def test_invalid_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ollama_timeout_s=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_level="VERBOSE")
