from __future__ import annotations

import pytest

from dataset_core.path_safety import normalize_filename_override


@pytest.mark.parametrize(
    ("raw_filename", "expected"),
    [
        ("../nested/..\\escape?.csv", "escape_.csv"),
        ("..\\..\\unsafe folder\\report?.csv", "report_.csv"),
        (".hidden", "hidden.csv"),
        ("custom_name", "custom_name.csv"),
        ("archive.tar.gz", "archive.tar.csv"),
    ],
)
def test_normalize_filename_override_is_cross_platform_and_canonical(raw_filename: str, expected: str):
    assert normalize_filename_override(raw_filename) == expected


@pytest.mark.parametrize(
    "raw_filename",
    ["", "   ", ".", "..", "////", "\\\\\\", "..."],
)
def test_normalize_filename_override_rejects_empty_or_unsafe_basenames(raw_filename: str):
    with pytest.raises(ValueError, match="filename_override"):
        normalize_filename_override(raw_filename)
