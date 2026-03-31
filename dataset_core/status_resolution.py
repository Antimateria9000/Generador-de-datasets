from __future__ import annotations

from dataclasses import dataclass, field

_PARTIAL_EXTERNAL_STATUSES = {None, "", "not_validated", "skipped"}
_PARTIAL_INTERNAL_STATUSES = {None, "", "passed_with_warnings", "skipped", "unsupported"}
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
    validation_outcome: str = "success_validated"


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


def _prefix_reason(prefix: str, detail: str | None) -> str:
    sentence = prefix.rstrip(".:;!, ") + "."
    cleaned = _clean_reason(detail)
    if not cleaned:
        return sentence
    lowered = cleaned.lower()
    if prefix.lower() in lowered:
        return cleaned
    return f"{sentence} {cleaned}"


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
    partial_validation = False
    validation_failed = False

    normalized_internal = None if internal_validation_status is None else str(internal_validation_status).strip().lower()
    normalized_external = None if external_validation_status is None else str(external_validation_status).strip().lower()

    if normalized_internal == "passed":
        pass
    elif normalized_internal in _PARTIAL_INTERNAL_STATUSES:
        partial_validation = True
        if normalized_internal == "passed_with_warnings":
            reasons.append("Internal validation passed with warnings.")
        elif normalized_internal == "skipped":
            reasons.append(
                _clean_reason(internal_validation_reason)
                or "Internal validation was skipped."
            )
        elif normalized_internal == "unsupported":
            reasons.append(
                _clean_reason(internal_validation_reason)
                or "Internal validation is not applicable for the requested dataset profile."
            )
        else:
            reasons.append("Internal validation did not fully validate the dataset.")
    else:
        validation_failed = True
        reasons.append(
            _clean_reason(internal_validation_reason)
            or _INTERNAL_STATUS_MESSAGES.get(
                normalized_internal,
                f"Internal validation status is {internal_validation_status}.",
            )
        )

    if normalized_external == "passed":
        pass
    elif normalized_external in _PARTIAL_EXTERNAL_STATUSES:
        partial_validation = True
        reasons.append(_prefix_reason("External validation did not run", external_validation_reason))
    else:
        validation_failed = True
        reasons.append(
            _clean_reason(external_validation_reason)
            or _EXTERNAL_STATUS_MESSAGES.get(
                normalized_external,
                f"External validation status is {external_validation_status}.",
            )
        )

    if qlib_requested and not qlib_compatible:
        validation_failed = True
        if qlib_errors:
            reasons.extend(str(item) for item in qlib_errors if str(item).strip())
        else:
            reasons.append("The requested Qlib artifact is not compatible.")

    validation_outcome = "success_validated"
    if validation_failed:
        validation_outcome = "failure"
    elif partial_validation:
        validation_outcome = "success_partial_validation"

    return StatusResolution(
        status="error" if validation_failed else ("warning" if reasons else "success"),
        reasons=_unique(reasons),
        neutral_notes=_unique(neutral_messages),
        validation_outcome=validation_outcome,
    )
