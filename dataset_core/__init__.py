from __future__ import annotations

from importlib import import_module

__all__ = [
    "BatchOrchestrator",
    "DatasetExportService",
    "DatasetRequest",
    "DatasetSchemaBuilder",
    "ExternalValidationConfig",
    "ProviderConfig",
    "RequestContractError",
    "TemporalRange",
    "parse_extras",
    "parse_tickers_text",
    "resolve_ticker_inputs",
]

_EXPORT_MAP = {
    "BatchOrchestrator": ("dataset_core.batch_orchestrator", "BatchOrchestrator"),
    "DatasetExportService": ("dataset_core.export_service", "DatasetExportService"),
    "DatasetRequest": ("dataset_core.contracts", "DatasetRequest"),
    "DatasetSchemaBuilder": ("dataset_core.schema_builder", "DatasetSchemaBuilder"),
    "ExternalValidationConfig": ("dataset_core.contracts", "ExternalValidationConfig"),
    "ProviderConfig": ("dataset_core.contracts", "ProviderConfig"),
    "RequestContractError": ("dataset_core.contracts", "RequestContractError"),
    "TemporalRange": ("dataset_core.contracts", "TemporalRange"),
    "parse_extras": ("dataset_core.contracts", "parse_extras"),
    "parse_tickers_text": ("dataset_core.contracts", "parse_tickers_text"),
    "resolve_ticker_inputs": ("dataset_core.contracts", "resolve_ticker_inputs"),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORT_MAP[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
