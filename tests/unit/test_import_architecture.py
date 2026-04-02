from __future__ import annotations

import subprocess
import sys


def test_reference_and_validation_modules_import_cleanly_without_circular_dependency(tmp_path):
    command = [
        sys.executable,
        "-c",
        "import dataset_core.reference_adapters; import dataset_core.validation_external",
    ]

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
