from __future__ import annotations

from dataclasses import dataclass, field

_NEUTRAL_EXTERNAL_STATUSES = {None, "", "not_validated", "skipped"}
_NEUTRAL_INTERNAL_STATUSES = {None, "", "passed", "skipped"}


@dataclass(frozen=True)
class StatusResolution:
    status: str
    reasons: list[str] = field(default_factory=list)
    neutral_notes: list[str] = field(default_factory=list)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def resolve_ticker_status(
    *,
    warnings: list[str] | tuple[str, ...],
    internal_validation_status: str | None,
    external_validation_status: str | None,
    qlib_requested: bool,
    qlib_compatible: bool,
    qlib_errors: list[str] | tuple[str, ...],
) -> StatusResolution:
    reasons = _unique(list(warnings))
    neutral_notes: list[str] = []

    normalized_internal = None if internal_validation_status is None else str(internal_validation_status).strip().lower()
    normalized_external = None if external_validation_status is None else str(external_validation_status).strip().lower()

    if normalized_internal not in _NEUTRAL_INTERNAL_STATUSES:
        reasons.append(f"Internal validation status is {internal_validation_status}.")
    elif normalized_internal == "skipped":
        neutral_notes.append("Internal validation was skipped by configuration.")

    if normalized_external == "passed":
        pass
    elif normalized_external in _NEUTRAL_EXTERNAL_STATUSES:
        neutral_notes.append("External validation did not run.")
    else:
        reasons.append(f"External validation status is {external_validation_status}.")

    if qlib_requested and not qlib_compatible:
        if qlib_errors:
            reasons.extend(str(item) for item in qlib_errors if str(item).strip())
        else:
            reasons.append("The requested Qlib artifact is not compatible.")

    return StatusResolution(
        status="warning" if reasons else "success",
        reasons=_unique(reasons),
        neutral_notes=_unique(neutral_notes),
    )
