from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_core.settings import ensure_workspace_tree

REPO_ROOT = PROJECT_ROOT
TEXT_EXTENSIONS = {
    ".bat",
    ".cfg",
    ".csv",
    ".env",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".hypothesis",
    ".pytest_cache",
    "workspace",
    "Dataset",
    "Datasets",
    "Versiones antiguas",
}
EXCLUDED_FILE_PATHS = {
    Path("scripts/scrub_personal_data.py"),
}
SENSITIVE_PATTERNS = {
    "absolute_user_path": re.compile(r"([A-Za-z]:\\\\Users\\\\|/Users/)"),
    "api_key": re.compile(r"\bapi[_-]?key\b\s*[:=]", re.IGNORECASE),
    "token": re.compile(r"\b(access_token|refresh_token|bearer_token|token)\b\s*[:=]", re.IGNORECASE),
    "secret": re.compile(r"\b(client_secret|secret)\b\s*[:=]", re.IGNORECASE),
    "password": re.compile(r"\bpassword\b\s*[:=]", re.IGNORECASE),
}
FILE_NAME_PATTERNS = {
    "coverage_user_pid": re.compile(r"\.coverage\..+\.pid\d+", re.IGNORECASE),
    "streamlit_secrets": re.compile(r"\.streamlit[\\/]+secrets\.toml$", re.IGNORECASE),
    "dotenv": re.compile(r"(^|[\\/])\.env(\..+)?$", re.IGNORECASE),
}


def _candidate_personal_tokens() -> set[str]:
    tokens: set[str] = set()
    home = Path.home().name.strip().lower()
    if len(home) >= 3:
        tokens.add(home)
    userprofile = os.getenv("USERPROFILE", "").strip()
    if userprofile:
        tokens.add(Path(userprofile).name.strip().lower())
    return {token for token in tokens if token}


def _iter_repo_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        if path.relative_to(REPO_ROOT) in EXCLUDED_FILE_PATHS:
            continue
        files.append(path)
    return files


def _scan_file(path: Path, personal_tokens: set[str]) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    relative_path = path.relative_to(REPO_ROOT)
    for label, pattern in FILE_NAME_PATTERNS.items():
        if pattern.search(str(relative_path)):
            findings.append({"file": str(relative_path), "kind": label, "line": None, "snippet": str(relative_path)})

    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return findings

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = path.read_text(encoding="latin-1")
        except UnicodeDecodeError:
            return findings

    for line_number, line in enumerate(content.splitlines(), start=1):
        lowered = line.lower()
        for token in personal_tokens:
            if token in lowered:
                findings.append(
                    {"file": str(relative_path), "kind": "personal_token", "line": line_number, "snippet": line.strip()}
                )
        for label, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(line):
                findings.append({"file": str(relative_path), "kind": label, "line": line_number, "snippet": line.strip()})

    return findings


def main() -> int:
    personal_tokens = _candidate_personal_tokens()
    findings: list[dict[str, object]] = []
    for path in _iter_repo_files():
        findings.extend(_scan_file(path, personal_tokens))

    workspace = ensure_workspace_tree()
    report_path = workspace["audits"] / "scrub_personal_data.json"
    report_path.write_text(json.dumps({"findings": findings}, ensure_ascii=False, indent=2), encoding="utf-8")

    if findings:
        for finding in findings:
            print(f"{finding['kind']}: {finding['file']}:{finding['line']} :: {finding['snippet']}")
        return 1

    print("No personal data or secrets detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
