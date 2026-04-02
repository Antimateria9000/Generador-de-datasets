from __future__ import annotations

from pathlib import Path

from dataset_core import settings


def test_load_local_env_returns_none_when_env_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("EODHD_API_KEY", raising=False)
    settings.reset_local_env_cache()

    assert settings.load_local_env(project_root=tmp_path) is None
    assert settings.get_default_eodhd_api_key(project_root=tmp_path) is None


def test_get_default_eodhd_api_key_loads_env_from_project_root(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("EODHD_API_KEY=env-secret\n", encoding="utf-8")
    monkeypatch.delenv("EODHD_API_KEY", raising=False)
    settings.reset_local_env_cache()

    assert settings.load_local_env(project_root=tmp_path) == env_path
    assert settings.get_default_eodhd_api_key(project_root=tmp_path) == "env-secret"


def test_load_local_env_is_idempotent_per_project_root(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("EODHD_API_KEY=env-secret\n", encoding="utf-8")
    monkeypatch.delenv("EODHD_API_KEY", raising=False)
    settings.reset_local_env_cache()

    calls = {"fallback": 0}

    monkeypatch.setattr(settings, "_load_env_with_python_dotenv", lambda path: False)

    original_loader = settings._load_env_without_python_dotenv

    def _counting_loader(path: Path) -> None:
        calls["fallback"] += 1
        original_loader(path)

    monkeypatch.setattr(settings, "_load_env_without_python_dotenv", _counting_loader)

    settings.load_local_env(project_root=tmp_path)
    settings.load_local_env(project_root=tmp_path)

    assert calls["fallback"] == 1


def test_get_default_eodhd_api_key_falls_back_cleanly_without_python_dotenv(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("EODHD_API_KEY=fallback-secret\n", encoding="utf-8")
    monkeypatch.delenv("EODHD_API_KEY", raising=False)
    settings.reset_local_env_cache()

    monkeypatch.setattr(settings, "_load_env_with_python_dotenv", lambda path: False)

    assert settings.load_local_env(project_root=tmp_path) == env_path
    assert settings.get_default_eodhd_api_key(project_root=tmp_path) == "fallback-secret"
