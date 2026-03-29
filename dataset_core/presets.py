from __future__ import annotations

from dataclasses import dataclass

from dataset_core.settings import CORE_COLUMNS, OPTIONAL_COLUMNS, QLIB_OPTIONAL_COLUMNS, QLIB_REQUIRED_COLUMNS


@dataclass(frozen=True)
class OutputPreset:
    name: str
    description: str
    default_extras: tuple[str, ...]
    allowed_extras: tuple[str, ...]
    forced_extras: tuple[str, ...] = ()
    locked_extras: tuple[str, ...] = ()
    qlib_ready: bool = False


@dataclass(frozen=True)
class ResolvedPreset:
    preset: OutputPreset
    selected_extras: tuple[str, ...]
    ignored_extras: tuple[str, ...]
    output_columns: tuple[str, ...]


PRESETS: dict[str, OutputPreset] = {
    "base": OutputPreset(
        name="base",
        description="Date + OHLCV with optional extras for general inspection.",
        default_extras=(),
        allowed_extras=OPTIONAL_COLUMNS,
        qlib_ready=False,
    ),
    "extended": OutputPreset(
        name="extended",
        description="Date + OHLCV plus default market extras for research and QA.",
        default_extras=("adj_close", "dividends", "stock_splits"),
        allowed_extras=OPTIONAL_COLUMNS,
        qlib_ready=False,
    ),
    "qlib": OutputPreset(
        name="qlib",
        description="Closed Qlib contract: adjusted OHLCV, mandatory factor and hard validation.",
        default_extras=("factor",),
        allowed_extras=("factor",) + QLIB_OPTIONAL_COLUMNS,
        forced_extras=("factor",),
        locked_extras=("factor",),
        qlib_ready=True,
    ),
}

_EXTRA_ORDER: tuple[str, ...] = OPTIONAL_COLUMNS


def get_preset(name: str) -> OutputPreset:
    normalized = str(name or "base").strip().lower()
    if normalized not in PRESETS:
        raise ValueError(f"Unknown output preset: {name!r}")
    return PRESETS[normalized]


def resolve_preset(name: str, requested_extras: list[str] | tuple[str, ...]) -> ResolvedPreset:
    preset = get_preset(name)
    requested = [str(item).strip().lower() for item in requested_extras if str(item).strip()]
    selected: list[str] = list(preset.default_extras)
    ignored: list[str] = []

    for item in requested:
        if item not in preset.allowed_extras:
            ignored.append(item)
            continue
        if item not in selected:
            selected.append(item)

    for item in preset.forced_extras:
        if item not in selected:
            selected.append(item)

    ordered_selected = tuple(extra for extra in _EXTRA_ORDER if extra in selected)
    ignored_extras = tuple(dict.fromkeys(ignored))

    if preset.name == "qlib":
        columns = list(QLIB_REQUIRED_COLUMNS)
        if "adj_close" in ordered_selected:
            columns.append("adj_close")
    else:
        columns = list(CORE_COLUMNS)
        columns.extend(extra for extra in _EXTRA_ORDER if extra in ordered_selected)

    return ResolvedPreset(
        preset=preset,
        selected_extras=ordered_selected,
        ignored_extras=ignored_extras,
        output_columns=tuple(columns),
    )
