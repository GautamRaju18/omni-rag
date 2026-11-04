"""Shared fixtures."""

from __future__ import annotations

import pytest

from omnirag.core.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Deterministic settings that ignore the developer's local .env file."""
    return Settings(_env_file=None, env="test", log_json=False, neo4j_password="test-password")
