from __future__ import annotations

from pathlib import Path

from scripts import scrub_personal_data
from tests.fixtures.sample_data import make_provider_frame


def test_repo_root_no_longer_contains_legacy_trees_or_mirror_modules():
    repo_root = Path(__file__).resolve().parents[2]
    forbidden = [
        "Dataset",
        "Datasets",
        "Versiones antiguas",
        "exporters",
        "data_quality.py",
        "market_context.py",
        "yfinance_provider.py",
        "lanzador_exportador.py",
        "coverage.xml",
    ]

    for name in forbidden:
        assert not (repo_root / name).exists(), name


def test_gitignore_blocks_workspace_and_local_garbage():
    repo_root = Path(__file__).resolve().parents[2]
    content = (repo_root / ".gitignore").read_text(encoding="utf-8")

    for required in (
        ".venv/",
        "workspace/",
        "verification_output/",
        "*.tmp",
        "Dataset/",
        "Datasets/",
        "Versiones antiguas/",
        "**/.git/",
    ):
        assert required in content


def test_scrub_personal_data_detects_local_paths(tmp_path, monkeypatch):
    repo_file = tmp_path / "sample.txt"
    local_path = "C:\\Users\\alice\\secret.txt"
    repo_file.write_text(f"source={local_path}\n", encoding="utf-8")
    monkeypatch.setattr(scrub_personal_data, "REPO_ROOT", tmp_path)

    findings = scrub_personal_data._scan_file(repo_file, {"alice"})

    assert any(item["kind"] == "absolute_user_path" for item in findings)
    assert any(item["kind"] == "personal_token" for item in findings)


def test_scrub_personal_data_detects_unc_paths(tmp_path, monkeypatch):
    repo_file = tmp_path / "sample.txt"
    repo_file.write_text(r"source=\\corp-fs\home\alice\secret.txt" + "\n", encoding="utf-8")
    monkeypatch.setattr(scrub_personal_data, "REPO_ROOT", tmp_path)

    findings = scrub_personal_data._scan_file(repo_file, {"alice"})

    assert any(item["kind"] == "absolute_user_path" for item in findings)


def test_clean_workspace_script_preserves_directory_tree(tmp_path, monkeypatch):
    monkeypatch.setattr("dataset_core.settings.WORKSPACE_ROOT", tmp_path / "workspace")
    from scripts.clean_workspace import main

    workspace_root = tmp_path / "workspace"
    nested_file = workspace_root / "exports" / "run-1" / "sample.csv"
    nested_file.parent.mkdir(parents=True, exist_ok=True)
    nested_file.write_text(make_provider_frame().to_csv(index=False), encoding="utf-8")

    exit_code = main(
        [
            "--workspace-root",
            str(workspace_root),
            "--all",
            "--confirm-all",
            "DELETE",
        ]
    )

    assert exit_code == 0
    assert (workspace_root / "exports").exists()
    assert list((workspace_root / "exports").iterdir()) == []
