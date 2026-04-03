from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from dataset_core import (
    BatchOrchestrator,
    DatasetRequest,
    ProviderConfig,
    TemporalRange,
    parse_tickers_text,
)
from dataset_core.external_validation_runtime import build_external_validation_config
from dataset_core.presets import resolve_preset
from dataset_core.settings import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_YFINANCE_CACHE_MODE,
    DQ_MODES,
    EXTERNAL_VALIDATION_DISABLED_REASON,
    LISTING_PREFERENCES,
    PRESET_NAMES,
    SUPPORTED_INTERVALS,
    YFINANCE_CACHE_MODES,
    is_external_validation_runtime_enabled,
    resolve_eodhd_api_key,
    sanitize_secret_text,
)
from dataset_core.workspace_inventory import filter_workspace_runs, list_workspace_runs

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


def resolve_requested_eodhd_api_key(
    manual_api_key: str,
    *,
    external_validation_provider: str,
) -> str | None:
    return resolve_eodhd_api_key(
        manual_api_key,
        allow_env_fallback=str(external_validation_provider or "").strip().lower() == "eodhd",
    )


def _external_validation_disabled_ui_copy() -> tuple[str, str]:
    return (
        "Módulo de validación externa desactivado",
        "Desactivado temporalmente. El pipeline actual utiliza únicamente validaciones internas; la UI no expone controles operativos y cualquier configuración residual se ignora de forma explícita.",
    )


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
    external_validation_provider: str = "",
    external_validation_enabled: bool = False,
    eodhd_api_key: str = "",
    eodhd_base_url: str = "",
    eodhd_timeout_seconds: str = "",
    eodhd_use_cache: bool = True,
    eodhd_cache_dir: str = "",
    eodhd_cache_ttl_seconds: str = "",
    eodhd_allow_partial_coverage: bool = False,
    eodhd_max_retries: str = "",
    eodhd_backoff_seconds: str = "",
    eodhd_price_lookback_days: str = "",
    provider_cache_mode: str = DEFAULT_YFINANCE_CACHE_MODE,
    provider_metadata_timeout: str = "",
    provider_metadata_candidate_limit: str = "",
    provider_context_cache_ttl_seconds: str = "",
    provider_batch_max_workers: str = "",
    provider_batch_chunk_size: str = "",
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
        time_range = TemporalRange.from_inputs(
            years=int(years),
            start=None,
            end=None,
            interval=interval,
        )

    def _parse_optional_positive_int(raw: str, label: str) -> int | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"{label} debe ser un entero positivo.") from exc
        if value < 1:
            raise ValueError(f"{label} debe ser >= 1.")
        return value

    def _parse_optional_non_negative_int(raw: str, label: str) -> int | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"{label} debe ser un entero mayor o igual que 0.") from exc
        if value < 0:
            raise ValueError(f"{label} debe ser >= 0.")
        return value

    def _parse_optional_positive_float(raw: str, label: str) -> float | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"{label} debe ser un numero positivo.") from exc
        if value <= 0:
            raise ValueError(f"{label} debe ser > 0.")
        return value

    def _parse_optional_non_negative_float(raw: str, label: str) -> float | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"{label} debe ser un numero mayor o igual que 0.") from exc
        if value < 0:
            raise ValueError(f"{label} debe ser >= 0.")
        return value

    effective_eodhd_api_key = None
    if is_external_validation_runtime_enabled():
        effective_eodhd_api_key = resolve_requested_eodhd_api_key(
            eodhd_api_key,
            external_validation_provider=external_validation_provider,
        )
        if str(external_validation_provider or "").strip().lower() == "eodhd" and effective_eodhd_api_key is None:
            raise ValueError(
                "EODHD requiere una API key. Define EODHD_API_KEY en el .env de la raiz del proyecto o introduce una key manual para esta sesion."
            )
    external_validation = build_external_validation_config(
        enabled=external_validation_enabled or None,
        provider=external_validation_provider,
        reference_dir=reference_dir,
        manual_events_file=manual_events_file,
        eodhd_api_key=effective_eodhd_api_key,
        eodhd_base_url=None if not eodhd_base_url.strip() else eodhd_base_url.strip(),
        eodhd_timeout_seconds=_parse_optional_positive_float(
            eodhd_timeout_seconds,
            "EODHD timeout",
        ),
        eodhd_use_cache=bool(eodhd_use_cache),
        eodhd_cache_dir=eodhd_cache_dir,
        eodhd_cache_ttl_seconds=_parse_optional_non_negative_int(
            eodhd_cache_ttl_seconds,
            "EODHD cache TTL",
        ),
        eodhd_allow_partial_coverage=bool(eodhd_allow_partial_coverage),
        eodhd_max_retries=_parse_optional_positive_int(
            eodhd_max_retries,
            "EODHD max retries",
        ),
        eodhd_backoff_seconds=_parse_optional_non_negative_float(
            eodhd_backoff_seconds,
            "EODHD backoff",
        ),
        eodhd_price_lookback_days=_parse_optional_positive_int(
            eodhd_price_lookback_days,
            "EODHD price lookback",
        ),
    )

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
        provider=ProviderConfig(
            cache_mode=provider_cache_mode,
            metadata_timeout=_parse_optional_positive_float(
                provider_metadata_timeout,
                "Metadata timeout",
            ),
            metadata_candidate_limit=_parse_optional_positive_int(
                provider_metadata_candidate_limit,
                "Metadata candidate limit",
            ),
            context_cache_ttl_seconds=_parse_optional_non_negative_int(
                provider_context_cache_ttl_seconds,
                "Context cache TTL",
            ),
            batch_max_workers=_parse_optional_positive_int(
                provider_batch_max_workers,
                "Batch max workers",
            ),
            batch_chunk_size=_parse_optional_positive_int(
                provider_batch_chunk_size,
                "Batch chunk size",
            ),
        ),
        external_validation=external_validation,
    )


def _build_exact_temporal_range(
    start_date: date,
    end_date: date,
    interval: str,
    now_utc: pd.Timestamp | None = None,
) -> TemporalRange:
    return TemporalRange.from_inputs(
        years=None,
        start=start_date,
        end=end_date,
        interval=interval,
        now_utc=now_utc,
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

    st.checkbox("adj_close", key="extra_adj_close", disabled=preset == "qlib")
    st.checkbox("dividends", key="extra_dividends", disabled=preset == "qlib")
    st.checkbox("stock_splits", key="extra_stock_splits", disabled=preset == "qlib")
    st.checkbox("factor", key="extra_factor", disabled=preset == "qlib")

    selected_extras = [
        extra
        for extra in ("adj_close", "dividends", "stock_splits", "factor")
        if st.session_state.get(f"extra_{extra}", False)
    ]
    if preset == "qlib":
        selected_extras = ["factor"]
    resolved_general = resolve_preset(preset, selected_extras)
    st.caption(f"Salida principal: `{', '.join(resolved_general.output_columns)}`")

    if qlib_sanitization and preset != "qlib":
        resolved_qlib = resolve_preset("qlib", [])
        st.caption(
            "El artefacto Qlib paralelo usara solo columnas compatibles con Qlib. "
            "Las columnas extra incompatibles se conservaran en la salida general."
        )
        st.caption(f"Artefacto Qlib paralelo: `{', '.join(resolved_qlib.output_columns)}`")
    elif preset == "qlib":
        st.caption("Preset Qlib cerrado: el backend forzara factor y validacion estricta aunque la UI sea manipulada.")


def _render_results(batch_result) -> None:
    counts = batch_result.status_counts
    st.success("Dataset generado")
    col1, col2, col3 = st.columns(3)
    col1.metric("Success", counts.get("success", 0))
    col2.metric("Warning", counts.get("warning", 0))
    col3.metric("Error", counts.get("error", 0))
    validation_counts = getattr(batch_result, "validation_outcome_counts", {})
    st.caption(
        "Validated success="
        f"{validation_counts.get('success_validated', 0)} | "
        f"Partial validation={validation_counts.get('success_partial_validation', 0)} | "
        f"Validation failure={validation_counts.get('failure', 0)}"
    )

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
                "validation_outcome",
                "status_reasons",
                "neutral_notes",
                "qlib_compatible",
                "warnings",
                "errors",
                "provider_warnings",
                "csv_path",
                "qlib_csv_path",
                "meta_path",
                "dq_path",
                "external_validation_json_path",
                "qlib_report_path",
            ]
        ]
        st.dataframe(summary, width="stretch")

    for result in batch_result.results:
        with st.expander(f"{result.ticker} | status={result.status}", expanded=False):
            st.write(f"Requested ticker: `{result.requested_ticker}`")
            st.write(f"Resolved ticker: `{result.resolved_ticker}`")
            st.write(f"Provider symbol: `{result.provider_symbol}`")
            st.write(f"Validation outcome: `{result.validation_outcome}`")
            if getattr(batch_result, "run_log_path", None):
                st.write(f"Run log: `{batch_result.run_log_path}`")
            if result.status_reasons:
                st.write("Status reasons")
                st.json(result.status_reasons)
            if result.neutral_notes:
                st.write("Neutral notes")
                st.json(result.neutral_notes)
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


def _render_workspace_panel(workspace_root_value: str) -> None:
    st.markdown("## Workspace")
    workspace_root = Path(workspace_root_value).expanduser().resolve()

    with st.expander("Corridas anteriores", expanded=False):
        try:
            inventory = list_workspace_runs(workspace_root)
        except Exception as exc:
            st.error(f"No se pudo cargar el inventario del workspace: {exc}")
            return

        if not inventory:
            st.info("No hay corridas registradas todavia.")
            return

        filter_col_1, filter_col_2, filter_col_3 = st.columns(3)
        with filter_col_1:
            ticker_filter = st.text_input("Ticker", key="workspace_filter_ticker")
            created_from = st.date_input("Fecha desde", value=None, key="workspace_filter_from")
        with filter_col_2:
            preset_filter = st.selectbox(
                "Preset",
                ["", *PRESET_NAMES],
                format_func=lambda value: "Todos" if value == "" else PRESET_LABELS.get(value, value),
                key="workspace_filter_preset",
            )
            created_to = st.date_input("Fecha hasta", value=None, key="workspace_filter_to")
        with filter_col_3:
            interval_options = sorted({record.interval for record in inventory if record.interval})
            status_filter = st.selectbox(
                "Estado",
                ["", "success", "warning", "error", "orphan", "unknown"],
                format_func=lambda value: "Todos" if value == "" else value,
                key="workspace_filter_status",
            )
            interval_filter = st.selectbox(
                "Intervalo",
                ["", *interval_options],
                format_func=lambda value: "Todos" if value == "" else value,
                key="workspace_filter_interval",
            )

        age_filter = st.number_input(
            "Antiguedad minima (dias)",
            min_value=0,
            value=0,
            step=1,
            key="workspace_filter_age",
        )
        filtered = filter_workspace_runs(
            inventory,
            ticker=ticker_filter,
            preset=preset_filter or None,
            interval=interval_filter or None,
            status=status_filter or None,
            older_than_days=None if int(age_filter) <= 0 else int(age_filter),
            created_from=created_from,
            created_to=created_to,
        )

        rows = [record.to_dict() for record in filtered]
        if not rows:
            st.info("No hay corridas que coincidan con los filtros.")
            return

        summary = pd.DataFrame(rows)[
            [
                "run_id",
                "created_at_utc",
                "ticker_summary",
                "preset",
                "interval",
                "overall_status",
                "size_human",
                "age_days",
                "orphaned",
                "missing_components",
            ]
        ]
        st.dataframe(summary, width="stretch")

        for record in filtered[:20]:
            with st.expander(f"{record.run_id} | status={record.overall_status}", expanded=False):
                st.write(f"Tickers: `{', '.join(record.tickers) or '-'}`")
                st.write(f"Preset: `{record.preset}`")
                st.write(f"Intervalo: `{record.interval}`")
                st.write(f"Tamanio: `{record.size_human}`")
                st.write(f"Workspace root: `{record.workspace_root}`")
                st.json(record.to_dict())


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

        with st.expander("Runtime batch / contexto", expanded=False):
            runtime_col_1, runtime_col_2, runtime_col_3 = st.columns(3)
            with runtime_col_1:
                execution_mode = st.selectbox(
                    "Modo de ejecucion",
                    ("concurrent", "sequential"),
                    help="El modo secuencial sigue disponible como fallback de depuracion.",
                )
                provider_metadata_timeout = st.text_input(
                    "Metadata timeout (s)",
                    value="",
                    help="Timeout explicito para lookups de metadata/contexto.",
                )
                provider_cache_mode = st.selectbox(
                    "Modo cache yfinance",
                    YFINANCE_CACHE_MODES,
                    index=YFINANCE_CACHE_MODES.index(DEFAULT_YFINANCE_CACHE_MODE),
                    help="`run` aísla la caché por corrida y reduce contención multiproceso.",
                )
            with runtime_col_2:
                provider_metadata_candidate_limit = st.text_input(
                    "Metadata candidate limit",
                    value="",
                    help="Numero maximo de candidatos a probar por simbolo.",
                )
                provider_batch_max_workers = st.text_input(
                    "Batch max workers",
                    value="",
                    help="Workers para planificacion, adquisicion y finalizacion concurrente.",
                )
            with runtime_col_3:
                provider_context_cache_ttl_seconds = st.text_input(
                    "Context cache TTL (s)",
                    value="",
                    help="TTL de la cache persistente de contexto. Usa 0 para desactivarla.",
                )
                provider_batch_chunk_size = st.text_input(
                    "Batch chunk size",
                    value="",
                    help="Tamano del lote para adquisicion agrupada.",
                )

        with st.expander("Validación externa", expanded=False):
            disabled_title, disabled_caption = _external_validation_disabled_ui_copy()
            st.info(disabled_title)
            st.caption(disabled_caption)
            if not is_external_validation_runtime_enabled():
                st.caption(EXTERNAL_VALIDATION_DISABLED_REASON)
            external_validation_enabled = False
            external_validation_provider = ""
            reference_dir = ""
            manual_events_file = ""
            eodhd_api_key = ""
            eodhd_base_url = ""
            eodhd_timeout_seconds = ""
            eodhd_use_cache = False
            eodhd_cache_dir = ""
            eodhd_cache_ttl_seconds = ""
            eodhd_allow_partial_coverage = False
            eodhd_max_retries = ""
            eodhd_backoff_seconds = ""
            eodhd_price_lookback_days = ""

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
                external_validation_provider=external_validation_provider,
                external_validation_enabled=external_validation_enabled,
                eodhd_api_key=eodhd_api_key,
                eodhd_base_url=eodhd_base_url,
                eodhd_timeout_seconds=eodhd_timeout_seconds,
                eodhd_use_cache=eodhd_use_cache,
                eodhd_cache_dir=eodhd_cache_dir,
                eodhd_cache_ttl_seconds=eodhd_cache_ttl_seconds,
                eodhd_allow_partial_coverage=eodhd_allow_partial_coverage,
                eodhd_max_retries=eodhd_max_retries,
                eodhd_backoff_seconds=eodhd_backoff_seconds,
                eodhd_price_lookback_days=eodhd_price_lookback_days,
                provider_cache_mode=provider_cache_mode,
                provider_metadata_timeout=provider_metadata_timeout,
                provider_metadata_candidate_limit=provider_metadata_candidate_limit,
                provider_context_cache_ttl_seconds=provider_context_cache_ttl_seconds,
                provider_batch_max_workers=provider_batch_max_workers,
                provider_batch_chunk_size=provider_batch_chunk_size,
            )
            with st.spinner("Generando dataset..."):
                batch_result = get_orchestrator().run(request, execution_mode=execution_mode)
            st.session_state["last_batch_result"] = batch_result
        except Exception as exc:
            safe_message = sanitize_secret_text(str(exc)) or "Error no disponible."
            st.error(f"Generacion fallida: {safe_message}")

    if "last_batch_result" in st.session_state:
        _render_results(st.session_state["last_batch_result"])

    _render_workspace_panel(output_dir)


if __name__ == "__main__":
    main()
