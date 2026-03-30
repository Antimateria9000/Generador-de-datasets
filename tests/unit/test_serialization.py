from __future__ import annotations

import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from dataset_core.serialization import cleanup_orphan_temp_files, write_json


def test_write_json_emits_strict_portable_payload(tmp_path):
    target = tmp_path / "payload.json"
    payload = {
        "nan_value": float("nan"),
        "pos_inf": float("inf"),
        "neg_inf": float("-inf"),
        "nested": {
            "np_float": np.float64(1.25),
            "np_array": np.array([1, np.nan, np.float64("inf")]),
            "dates": np.array([np.datetime64("2024-01-01"), np.datetime64("2024-01-02")]),
            "timestamp": pd.Timestamp("2024-01-03T04:05:06Z"),
            "date": date(2024, 1, 4),
            "datetime": datetime(2024, 1, 5, 6, 7, 8, tzinfo=timezone.utc),
        },
    }

    write_json(target, payload)
    loaded = json.loads(target.read_text(encoding="utf-8"))

    assert loaded["nan_value"] is None
    assert loaded["pos_inf"] is None
    assert loaded["neg_inf"] is None
    assert loaded["nested"]["np_float"] == 1.25
    assert loaded["nested"]["np_array"] == [1, None, None]
    assert loaded["nested"]["dates"] == ["2024-01-01", "2024-01-02"]
    assert loaded["nested"]["timestamp"] == "2024-01-03T04:05:06+00:00"
    assert loaded["nested"]["date"] == "2024-01-04"
    assert loaded["nested"]["datetime"] == "2024-01-05T06:07:08+00:00"


def test_cleanup_orphan_temp_files_removes_stray_tmp_files(tmp_path):
    stray = tmp_path / "reports" / "orphan.json.abcd.tmp"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_text("{}", encoding="utf-8")

    removed = cleanup_orphan_temp_files(tmp_path)

    assert stray in removed
    assert not stray.exists()
