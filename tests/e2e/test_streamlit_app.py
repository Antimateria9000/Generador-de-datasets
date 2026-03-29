from __future__ import annotations

from datetime import date

import pytest

from dataset_core.result_models import ArtifactPaths, BatchResult, TickerResult

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

import app.streamlit_app as streamlit_app


class FakeOrchestrator:
    def __init__(self, batch_result):
        self.batch_result = batch_result
        self.last_request = None

    def run(self, request):
        self.last_request = request
        return self.batch_result


def _run_streamlit_app():
    import app.streamlit_app as app_module

    app_module.main()


def test_streamlit_app_renders_main_form(monkeypatch, tmp_path):
    batch_result = BatchResult(
        batch_id="batch-test",
        run_id="20260328_000000_000000_deadbeef",
        output_root=tmp_path,
        csv_dir=tmp_path / "csv",
        meta_dir=tmp_path / "meta",
        report_dir=tmp_path / "reports",
        manifest_json_path=tmp_path / "manifest_batch.json",
        manifest_txt_path=tmp_path / "manifest_batch.txt",
        results=[
            TickerResult(
                ticker="MSFT",
                requested_ticker="MSFT",
                resolved_ticker="MSFT",
                status="success",
                qlib_compatible=True,
                columns=["date", "open", "high", "low", "close", "volume", "factor"],
                artifacts=ArtifactPaths(csv=tmp_path / "csv" / "MSFT.csv"),
            )
        ],
    )
    orchestrator = FakeOrchestrator(batch_result)
    monkeypatch.setattr(streamlit_app, "get_orchestrator", lambda: orchestrator)

    at = AppTest.from_function(_run_streamlit_app)
    at.run()
    assert len(at.text_area) == 1
    assert len(at.radio) == 1
    assert len(at.selectbox) == 4
    assert len(at.button) >= 1


def test_streamlit_helper_builds_request_for_qlib(monkeypatch, tmp_path):
    monkeypatch.setitem(streamlit_app.st.session_state, "extra_adj_close", False)
    monkeypatch.setitem(streamlit_app.st.session_state, "extra_dividends", False)
    monkeypatch.setitem(streamlit_app.st.session_state, "extra_stock_splits", False)
    monkeypatch.setitem(streamlit_app.st.session_state, "extra_factor", True)

    request = streamlit_app._build_request_from_form(
        tickers_text="MSFT",
        range_mode="Anos moviles",
        start_date=None,
        end_date=None,
        years=5,
        interval="1d",
        mode="qlib",
        listing_preference="exact_symbol",
        dq_mode="report",
        qlib_sanitization=True,
        output_dir=str(tmp_path),
        reference_dir="",
        manual_events_file="",
    )

    assert request.mode == "qlib"
    assert request.tickers == ["MSFT"]
    assert request.requires_factor is True
    assert request.qlib_sanitization is True


def test_streamlit_exact_range_helper_uses_shared_end_exclusive_policy():
    now_utc = streamlit_app.pd.Timestamp("2026-03-29T15:45:00")

    time_range = streamlit_app._build_exact_temporal_range(
        start_date=date(2026, 3, 20),
        end_date=date(2026, 3, 29),
        interval="1d",
        now_utc=now_utc,
    )

    assert time_range.start == streamlit_app.pd.Timestamp("2026-03-20T00:00:00")
    assert time_range.end == streamlit_app.pd.Timestamp("2026-03-30T00:00:00")


class _FakeColumn:
    def metric(self, *_args, **_kwargs):
        return None


class _FakeExpander:
    def __init__(self, sink, label):
        self._sink = sink
        self._sink["expanders"].append(label)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeStreamlit:
    def __init__(self):
        self.calls = {
            "success": [],
            "writes": [],
            "dataframes": [],
            "json": [],
            "expanders": [],
        }

    def success(self, value):
        self.calls["success"].append(value)

    def columns(self, count):
        return [_FakeColumn() for _ in range(count)]

    def write(self, value):
        self.calls["writes"].append(value)

    def dataframe(self, frame, width=None):
        self.calls["dataframes"].append((frame, width))

    def expander(self, label, expanded=False):
        return _FakeExpander(self.calls, f"{label}|expanded={expanded}")

    def json(self, payload):
        self.calls["json"].append(payload)


def test_render_results_exposes_warnings_errors_provider_warnings_and_run_log(monkeypatch, tmp_path):
    fake_streamlit = _FakeStreamlit()
    monkeypatch.setattr(streamlit_app, "st", fake_streamlit)
    batch_result = BatchResult(
        batch_id="batch-test",
        run_id="20260329_000000_000000_deadbeef",
        output_root=tmp_path,
        csv_dir=tmp_path / "csv",
        meta_dir=tmp_path / "meta",
        report_dir=tmp_path / "reports",
        manifest_json_path=tmp_path / "manifest_batch.json",
        manifest_txt_path=tmp_path / "manifest_batch.txt",
        run_log_path=tmp_path / "logs" / "dataset_factory.log",
        results=[
            TickerResult(
                ticker="MSFT",
                requested_ticker="MSFT",
                resolved_ticker="MSFT",
                status="warning",
                qlib_compatible=False,
                warnings=["general warning"],
                errors=["acquisition: RuntimeError: boom"],
                provider_warnings=["provider warning"],
                artifacts=ArtifactPaths(
                    csv=tmp_path / "csv" / "MSFT.csv",
                    meta=tmp_path / "reports" / "MSFT.meta.json",
                ),
            )
        ],
    )

    streamlit_app._render_results(batch_result)

    summary, width = fake_streamlit.calls["dataframes"][0]
    assert width == "stretch"
    assert list(summary.columns) == [
        "ticker",
        "status",
        "qlib_compatible",
        "warnings",
        "errors",
        "provider_warnings",
        "csv_path",
        "qlib_csv_path",
        "meta_path",
        "dq_path",
        "external_validation_json_path",
    ]
    assert any("Run log:" in str(item) for item in fake_streamlit.calls["writes"])
    assert len(fake_streamlit.calls["expanders"]) == 1
