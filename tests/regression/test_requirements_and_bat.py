from __future__ import annotations

from pathlib import Path


def test_requirements_are_plain_pip_files():
    repo_root = Path(__file__).resolve().parents[2]
    requirements = (repo_root / "requirements.txt").read_text(encoding="utf-8")
    requirements_dev = (repo_root / "requirements-dev.txt").read_text(encoding="utf-8")

    assert "```" not in requirements
    assert "streamlit" in requirements
    assert requirements_dev.startswith("-r requirements.txt")


def test_launch_streamlit_bat_uses_project_venv():
    repo_root = Path(__file__).resolve().parents[2]
    bat_content = (repo_root / "scripts" / "launch_streamlit.bat").read_text(encoding="utf-8")
    legacy_bat_content = (repo_root / "crear_dataset_desde_ventana.bat").read_text(encoding="utf-8")

    assert '".venv\\Scripts\\python.exe" -m streamlit run app\\streamlit_app.py' in bat_content
    assert 'call "scripts\\launch_streamlit.bat"' in legacy_bat_content
