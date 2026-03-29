from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from dataset_core import (
    BatchOrchestrator,
    DatasetRequest,
    ExternalValidationConfig,
    ProviderConfig,
    TemporalRange,
    parse_tickers_text,
)
from dataset_core.date_windows import build_ui_exact_end_exclusive
from dataset_core.settings import DEFAULT_OUTPUT_ROOT, DQ_MODES, LISTING_PREFERENCES, PRESET_NAMES, SUPPORTED_INTERVALS

PRESET_DEFAULTS = {
    "base": set(),
    "extended": {"adj_close", "dividends", "stock_splits"},
    "qlib": {"factor"},
}

PRESET_LABELS = {
    "base": "Base",
    "extended": "Extended",
    "qlib": "Qlib",
}

LISTING_LABELS = {
    "exact_symbol": "Exact symbol",
    "home_market": "Home market",
    "prefer_europe": "Prefer Europe",
    "prefer_usa": "Prefer USA",
}


def get_orchestrator() -> BatchOrchestrator:
    return BatchOrchestrator()


def _sync_extras_from_preset(preset: str) -> None:
    state_key = "_last_preset"
    if st.session_state.get(state_key) == preset:
        return

    defaults = PRESET_DEFAULTS[preset]
    for extra in ("adj_close", "dividends", "stock_splits", "factor"):
        st.session_state[f"extra_{extra}"] = extra in defaults
    st.session_state[state_key] = preset


def _build_request_from_form(
    tickers_text: str,
    range_mode: str,
    start_date: date | None,
    end_date: date | None,
    years: int,
    interval: str,
    mode: str,
    listing_preference: str,
    dq_mode: str,
    qlib_sanitization: bool,
    output_dir: str,
    reference_dir: str,
    manual_events_file: str,
) -> DatasetRequest:
    extras: list[str] = []
    for extra in ("adj_close", "dividends", "stock_splits", "factor"):
        if st.session_state.get(f"extra_{extra}", False):
            extras.append(extra)

    tickers = parse_tickers_text(tickers_text)
    if not tickers:
        raise ValueError("Debes introducir al menos un ticker valido.")

    if range_mode == "Rango exacto":
        if start_date is None or end_date is None:
            raise ValueError("Debes seleccionar fecha de inicio y fin.")
        time_range = _build_exact_temporal_range(start_date=start_date, end_date=end_date, interval=interval)
    else:
        time_range = TemporalRange.from_inputs(years=int(years), start=None, end=None)

    return DatasetRequest(
        tickers=tickers,
        time_range=time_range,
        output_dir=Path(output_dir).expanduser().resolve(),
        interval=interval,
        mode=mode,
        extras=extras,
        listing_preference=listing_preference,
        dq_mode=dq_mode,
        dq_market="AUTO",
        auto_adjust=False,
        actions=True,
        qlib_sanitization=qlib_sanitization or mode == "qlib",
        provider=ProviderConfig(),
        external_validation=ExternalValidationConfig(
            reference_dir=None if not reference_dir.strip() else Path(reference_dir).expanduser().resolve(),
            manual_events_file=None
            if not manual_events_file.strip()
            else Path(manual_events_file).expanduser().resolve(),
        ),
    )


def _build_exact_temporal_range(
    start_date: date,
    end_date: date,
    interval: str,
    now_utc: pd.Timestamp | None = None,
) -> TemporalRange:
    end_exclusive = build_ui_exact_end_exclusive(
        end_date=end_date,
        interval=interval,
        now_utc=now_utc,
    )
    return TemporalRange.from_inputs(
        years=None,
        start=start_date.isoformat(),
        end=end_exclusive.isoformat(),
    )


def _render_column_block(preset: str, qlib_sanitization: bool) -> None:
    st.markdown("### Columnas del dataset")
    fixed_columns = [
        ("date", True, True),
        ("open", True, True),
        ("high", True, True),
        ("low", True, True),
        ("close", True, True),
        ("volume", True, True),
    ]
    for label, value, disabled in fixed_columns:
        st.checkbox(label, value=value, disabled=disabled, key=f"fixed_{label}")

    st.caption("`volume` forma parte del contrato OHLCV y queda siempre activo.")

    st.checkbox("adj_close", key="extra_adj_close")
    st.checkbox("dividends", key="extra_dividends", disabled=preset == "qlib")
    st.checkbox("stock_splits", key="extra_stock_splits", disabled=preset == "qlib")
    st.checkbox("factor", key="extra_factor", disabled=preset == "qlib")

    if qlib_sanitization and preset != "qlib":
        st.caption(
            "El artefacto Qlib paralelo usara solo columnas compatibles con Qlib. "
            "Las columnas extra incompatibles se conservaran en la salida general."
        )


def _render_results(batch_result) -> None:
    counts = batch_result.status_counts
    st.success("Dataset generado")
    col1, col2, col3 = st.columns(3)
    col1.metric("Success", counts.get("success", 0))
    col2.metric("Warning", counts.get("warning", 0))
    col3.metric("Error", counts.get("error", 0))

    st.write(f"Output root: `{batch_result.output_root}`")
    st.write(f"CSV dir: `{batch_result.csv_dir}`")
    st.write(f"Report dir: `{batch_result.report_dir}`")
    st.write(f"Manifest: `{batch_result.manifest_json_path}`")
    if getattr(batch_result, "run_log_path", None):
        st.write(f"Run log: `{batch_result.run_log_path}`")

    result_rows = [result.to_dict() for result in batch_result.results]
    if result_rows:
        summary = pd.DataFrame(result_rows)[
            [
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
        ]
        st.dataframe(summary, width="stretch")

    for result in batch_result.results:
        with st.expander(f"{result.ticker} | status={result.status}", expanded=False):
            st.write(f"Requested ticker: `{result.requested_ticker}`")
            st.write(f"Resolved ticker: `{result.resolved_ticker}`")
            st.write(f"Provider symbol: `{result.provider_symbol}`")
            if getattr(batch_result, "run_log_path", None):
                st.write(f"Run log: `{batch_result.run_log_path}`")
            if result.warnings:
                st.write("Warnings")
                st.json(result.warnings)
            if result.errors:
                st.write("Errors")
                st.json(result.errors)
            if result.provider_warnings:
                st.write("Provider warnings")
                st.json(result.provider_warnings)
            if getattr(result, "error_context", None):
                st.write("Error context")
                st.json(result.error_context)

            artifact_payload = {key: value for key, value in result.artifacts.to_dict().items() if value}
            if artifact_payload:
                st.write("Artifacts")
                st.json(artifact_payload)


def main() -> None:
    st.set_page_config(page_title="Dataset Factory", layout="wide")
    st.title("Dataset Factory")
    st.caption("Generacion single y multi-ticker con presets Base, Extended y Qlib.")

    today = pd.Timestamp.utcnow().date()
    default_start = date(today.year - 5, today.month, min(today.day, 28))
    default_end = today

    with st.form("dataset_form"):
        tickers_text = st.text_area(
            "Tickers",
            height=140,
            placeholder="MSFT, AAPL, NVDA",
            help="Puedes separar tickers por coma, espacio o salto de linea.",
        )

        time_col, range_col = st.columns(2)
        with time_col:
            range_mode = st.radio("Modo temporal", ("Rango exacto", "Anos moviles"), horizontal=True)
        with range_col:
            mode = st.selectbox(
                "Preset de salida",
                PRESET_NAMES,
                format_func=lambda value: PRESET_LABELS[value],
            )

        _sync_extras_from_preset(mode)
        if "qlib_sanitization" not in st.session_state:
            st.session_state["qlib_sanitization"] = False
        if mode == "qlib":
            st.session_state["qlib_sanitization"] = True

        if range_mode == "Rango exacto":
            date_col_1, date_col_2 = st.columns(2)
            with date_col_1:
                start_date = st.date_input("Fecha inicio", value=default_start)
            with date_col_2:
                end_date = st.date_input(
                    "Fecha fin",
                    value=default_end,
                    max_value=today,
                    help="La UI interpreta esta fecha como fin inclusivo y la traduce a fin exclusivo internamente.",
                )
            years = 5
        else:
            years = st.number_input("Anos moviles", min_value=1, max_value=50, value=5, step=1)
            start_date = None
            end_date = None

        select_col_1, select_col_2, select_col_3 = st.columns(3)
        with select_col_1:
            interval = st.selectbox("Intervalo", SUPPORTED_INTERVALS, index=SUPPORTED_INTERVALS.index("1d"))
        with select_col_2:
            listing_preference = st.selectbox(
                "Listing preference",
                LISTING_PREFERENCES,
                format_func=lambda value: LISTING_LABELS[value],
            )
        with select_col_3:
            dq_mode = st.selectbox("Modo DQ", DQ_MODES, index=DQ_MODES.index("report"))

        qlib_sanitization = st.checkbox(
            "Saneamiento Qlib",
            key="qlib_sanitization",
            disabled=mode == "qlib",
            help="Prepara un artefacto listo para dump_bin.py y check_data_health.py.",
        )
        if mode == "qlib":
            st.caption("El preset qlib activa y bloquea el saneamiento Qlib para evitar estados ambiguos.")
        elif qlib_sanitization:
            st.caption("Se emitira la salida general saneada y, ademas, un artefacto Qlib-ready en paralelo.")
        else:
            st.caption("Solo se ejecutara el saneamiento general.")

        output_dir = st.text_input("Workspace de salida", value=str(DEFAULT_OUTPUT_ROOT))

        with st.expander("Validacion externa opcional"):
            reference_dir = st.text_input("Directorio con CSVs de referencia", value="")
            manual_events_file = st.text_input("Fichero de eventos manuales", value="")

        _render_column_block(mode, qlib_sanitization)
        submitted = st.form_submit_button("Generar dataset", width="stretch")

    if submitted:
        try:
            request = _build_request_from_form(
                tickers_text=tickers_text,
                range_mode=range_mode,
                start_date=start_date,
                end_date=end_date,
                years=years,
                interval=interval,
                mode=mode,
                listing_preference=listing_preference,
                dq_mode=dq_mode,
                qlib_sanitization=qlib_sanitization,
                output_dir=output_dir,
                reference_dir=reference_dir,
                manual_events_file=manual_events_file,
            )
            with st.spinner("Generando dataset..."):
                batch_result = get_orchestrator().run(request)
            st.session_state["last_batch_result"] = batch_result
        except Exception as exc:
            st.error(f"Generacion fallida: {exc}")
            st.exception(exc)

    if "last_batch_result" in st.session_state:
        _render_results(st.session_state["last_batch_result"])


if __name__ == "__main__":
    main()
