from __future__ import annotations

import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from dataset_core.contracts import DatasetRequest, EODHDExternalValidationConfig, ExternalValidationConfig, TemporalRange
from dataset_core.manifest_service import build_batch_manifest, render_batch_manifest_text
from dataset_core.result_models import BatchResult, TickerResult
from dataset_core.serialization import cleanup_orphan_temp_files, write_json, write_text
from dataset_core.settings import REDACTED_SECRET, register_secret


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


def test_write_json_redacts_registered_secrets_and_api_token_urls(tmp_path):
    secret = "serialization-secret"
    register_secret(secret)
    target = tmp_path / "payload.json"

    write_json(
        target,
        {
            "message": f"boom {secret}",
            "url": f"https://eodhd.test/api/eod/MSFT.US?api_token={secret}&fmt=json",
        },
    )
    content = target.read_text(encoding="utf-8")

    assert secret not in content
    assert REDACTED_SECRET in content
    assert "api_token=***redacted***" in content


def test_batch_manifest_and_text_report_do_not_expose_eodhd_api_key(tmp_path):
    secret = "manifest-secret"
    register_secret(secret)
    request = DatasetRequest(
        tickers=["MSFT"],
        time_range=TemporalRange.from_inputs(years=5, start=None, end=None),
        output_dir=tmp_path,
        external_validation=ExternalValidationConfig(
            provider="eodhd",
            enabled=True,
            eodhd=EODHDExternalValidationConfig(api_key=secret),
        ),
    )
    batch_result = BatchResult(
        batch_id="batch_test",
        run_id="run_test",
        output_root=tmp_path / "runs" / "run_test",
        csv_dir=tmp_path / "exports" / "run_test" / "csv",
        meta_dir=tmp_path / "manifests" / "run_test",
        report_dir=tmp_path / "reports" / "run_test",
        manifest_json_path=tmp_path / "runs" / "run_test" / "manifest_batch.json",
        manifest_txt_path=tmp_path / "runs" / "run_test" / "manifest_batch.txt",
        results=[
            TickerResult(
                ticker="MSFT",
                requested_ticker="MSFT",
                resolved_ticker="MSFT",
                status="error",
                qlib_compatible=False,
                errors=[f"EODHD failed with api_token={secret}"],
            )
        ],
    )

    manifest = build_batch_manifest(request=request, batch_result=batch_result)
    manifest_json_path = tmp_path / "manifest_batch.json"
    manifest_txt_path = tmp_path / "manifest_batch.txt"
    write_json(manifest_json_path, manifest)
    write_text(manifest_txt_path, render_batch_manifest_text(manifest))

    manifest_json = manifest_json_path.read_text(encoding="utf-8")
    manifest_txt = manifest_txt_path.read_text(encoding="utf-8")

    assert secret not in manifest_json
    assert secret not in manifest_txt
    assert REDACTED_SECRET in manifest_json
    assert REDACTED_SECRET in manifest_txt
