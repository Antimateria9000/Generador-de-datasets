from __future__ import annotations

from typing import Protocol

from dataset_core.external_sources.base import (
    ExternalCorporateActionsReferenceSource,
    ExternalPriceReferenceSource,
    ValidationScope,
    adapter_validation_scope,
    filter_event_frame,
    filter_reference_frame,
    normalize_event_frame,
    normalize_reference_frame,
    normalize_reference_timestamp,
)
from dataset_core.external_sources.csv_source import CSVReferenceSource
from dataset_core.external_sources.manual_events import ManualEventReferenceSource


class ReferenceAdapter(ExternalPriceReferenceSource, Protocol):
    """Backward-compatible alias for price reference sources."""


class PriceReferenceAdapter(ExternalPriceReferenceSource, Protocol):
    """Backward-compatible alias for price reference sources."""


class EventReferenceAdapter(ExternalCorporateActionsReferenceSource, Protocol):
    """Backward-compatible alias for sparse corporate actions sources."""


CSVReferenceAdapter = CSVReferenceSource
ManualEventAdapter = ManualEventReferenceSource

__all__ = [
    "CSVReferenceAdapter",
    "EventReferenceAdapter",
    "ManualEventAdapter",
    "PriceReferenceAdapter",
    "ReferenceAdapter",
    "ValidationScope",
    "adapter_validation_scope",
    "filter_event_frame",
    "filter_reference_frame",
    "normalize_event_frame",
    "normalize_reference_frame",
    "normalize_reference_timestamp",
]
