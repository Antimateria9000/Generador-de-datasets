from __future__ import annotations

from dataclasses import dataclass, field

_NEUTRAL_EXTERNAL_STATUSES = {None, "", "not_validated", "skipped"}
_NEUTRAL_INTERNAL_STATUSES = {None, "", "passed", "passed_with_warnings", "skipped", "unsupported"}
_EXTERNAL_STATUS_MESSAGES = {
    "adapter_error": "External validation adapter failed.",
    "failed": "External validation reported blocking differences.",
    "validation_error": "External validation crashed after loading the reference.",
}
_INTERNAL_STATUS_MESSAGES = {
    "execution_error": "Internal validation crashed.",
    "failed": "Internal validation reported blocking errors.",
}


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


def _clean_reason(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def resolve_ticker_status(
    *,
    warnings: list[str] | tuple[str, ...],
    neutral_notes: list[str] | tuple[str, ...] = (),
    internal_validation_status: str | None,
    internal_validation_reason: str | None = None,
    external_validation_status: str | None,
    external_validation_reason: str | None = None,
    qlib_requested: bool,
    qlib_compatible: bool,
    qlib_errors: list[str] | tuple[str, ...],
) -> StatusResolution:
    reasons = _unique(list(warnings))
    neutral_messages = _unique(list(neutral_notes))

    normalized_internal = None if internal_validation_status is None else str(internal_validation_status).strip().lower()
    normalized_external = None if external_validation_status is None else str(external_validation_status).strip().lower()

    if normalized_internal not in _NEUTRAL_INTERNAL_STATUSES:
        reasons.append(
            _clean_reason(internal_validation_reason)
            or _INTERNAL_STATUS_MESSAGES.get(
                normalized_internal,
                f"Internal validation status is {internal_validation_status}.",
            )
        )
    elif normalized_internal == "passed_with_warnings":
        neutral_messages.append("Internal validation passed with non-blocking warnings.")
    elif normalized_internal == "skipped":
        neutral_messages.append("Internal validation was skipped by configuration.")
    elif normalized_internal == "unsupported":
        neutral_messages.append(
            _clean_reason(internal_validation_reason)
            or "Internal validation is not applicable for the requested dataset profile."
        )

    if normalized_external == "passed":
        pass
    elif normalized_external in _NEUTRAL_EXTERNAL_STATUSES:
        neutral_messages.append(
            _clean_reason(external_validation_reason)
            or "External validation did not run."
        )
    else:
        reasons.append(
            _clean_reason(external_validation_reason)
            or _EXTERNAL_STATUS_MESSAGES.get(
                normalized_external,
                f"External validation status is {external_validation_status}.",
            )
        )

    if qlib_requested and not qlib_compatible:
        if qlib_errors:
            reasons.extend(str(item) for item in qlib_errors if str(item).strip())
        else:
            reasons.append("The requested Qlib artifact is not compatible.")

    return StatusResolution(
        status="warning" if reasons else "success",
        reasons=_unique(reasons),
        neutral_notes=_unique(neutral_messages),
    )
