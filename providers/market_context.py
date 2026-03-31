from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple

import yfinance as yf

from dataset_core.serialization import write_json
from dataset_core.settings import DEFAULT_CONTEXT_CACHE_TTL_SECONDS, DEFAULT_METADATA_CANDIDATE_LIMIT

LOGGER_NAME = "AB3.MarketContext"
logger = logging.getLogger(LOGGER_NAME)
logger.addHandler(logging.NullHandler())


@dataclass
class InstrumentContext:
    requested_symbol: str
    preferred_symbol: str
    resolved_symbol: str
    listing_preference: str
    quote_type: str
    asset_type: str
    asset_family: str
    market: Optional[str]
    calendar: Optional[str]
    timezone: Optional[str]
    currency: Optional[str]
    exchange_name: Optional[str]
    exchange_code: Optional[str]
    region: str
    is_24_7: bool
    volume_expected: bool
    corporate_actions_expected: bool
    calendar_validation_supported: bool
    dq_profile: str
    confidence: str = "medium"
    inference_sources: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    structured_warnings: List[Dict[str, Any]] = field(default_factory=list)
    resolution_trace: List[Dict[str, Any]] = field(default_factory=list)
    resolver_metrics: Dict[str, Any] = field(default_factory=dict)
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RawMetadataResult:
    history_metadata: Dict[str, Any]
    fast_info: Dict[str, Any]
    info: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    structured_warnings: List[Dict[str, Any]] = field(default_factory=list)
    query_trace: List[Dict[str, Any]] = field(default_factory=list)


_SUFFIX_RULES: Dict[str, Dict[str, str]] = {
    "PA": {"market": "XPAR", "calendar": "XPAR", "timezone": "Europe/Paris", "exchange_name": "Euronext Paris", "region": "EUROPE"},
    "AS": {"market": "XAMS", "calendar": "XAMS", "timezone": "Europe/Amsterdam", "exchange_name": "Euronext Amsterdam", "region": "EUROPE"},
    "BR": {"market": "XBRU", "calendar": "XBRU", "timezone": "Europe/Brussels", "exchange_name": "Euronext Brussels", "region": "EUROPE"},
    "LS": {"market": "XLIS", "calendar": "XLIS", "timezone": "Europe/Lisbon", "exchange_name": "Euronext Lisbon", "region": "EUROPE"},
    "IR": {"market": "XDUB", "calendar": "XDUB", "timezone": "Europe/Dublin", "exchange_name": "Euronext Dublin", "region": "EUROPE"},
    "MI": {"market": "XMIL", "calendar": "XMIL", "timezone": "Europe/Rome", "exchange_name": "Borsa Italiana", "region": "EUROPE"},
    "MC": {"market": "XMAD", "calendar": "XMAD", "timezone": "Europe/Madrid", "exchange_name": "Bolsa de Madrid", "region": "EUROPE"},
    "DE": {"market": "XETR", "calendar": "XETR", "timezone": "Europe/Berlin", "exchange_name": "Xetra", "region": "EUROPE"},
    "F": {"market": "XFRA", "calendar": "XFRA", "timezone": "Europe/Berlin", "exchange_name": "Frankfurt", "region": "EUROPE"},
    "L": {"market": "XLON", "calendar": "XLON", "timezone": "Europe/London", "exchange_name": "London Stock Exchange", "region": "EUROPE"},
    "SW": {"market": "XSWX", "calendar": "XSWX", "timezone": "Europe/Zurich", "exchange_name": "SIX Swiss Exchange", "region": "EUROPE"},
    "ST": {"market": "XSTO", "calendar": "XSTO", "timezone": "Europe/Stockholm", "exchange_name": "Nasdaq Stockholm", "region": "EUROPE"},
    "HE": {"market": "XHEL", "calendar": "XHEL", "timezone": "Europe/Helsinki", "exchange_name": "Nasdaq Helsinki", "region": "EUROPE"},
    "CO": {"market": "XCSE", "calendar": "XCSE", "timezone": "Europe/Copenhagen", "exchange_name": "Nasdaq Copenhagen", "region": "EUROPE"},
    "OL": {"market": "XOSL", "calendar": "XOSL", "timezone": "Europe/Oslo", "exchange_name": "Oslo Børs", "region": "EUROPE"},
    "WA": {"market": "XWAR", "calendar": "XWAR", "timezone": "Europe/Warsaw", "exchange_name": "Warsaw Stock Exchange", "region": "EUROPE"},
    "PR": {"market": "XPRA", "calendar": "XPRA", "timezone": "Europe/Prague", "exchange_name": "Prague Stock Exchange", "region": "EUROPE"},
    "VI": {"market": "XWBO", "calendar": "XWBO", "timezone": "Europe/Vienna", "exchange_name": "Vienna Stock Exchange", "region": "EUROPE"},
    "TO": {"market": "XTSE", "calendar": "XTSE", "timezone": "America/Toronto", "exchange_name": "Toronto Stock Exchange", "region": "AMERICAS"},
    "AX": {"market": "XASX", "calendar": "XASX", "timezone": "Australia/Sydney", "exchange_name": "ASX", "region": "APAC"},
    "T": {"market": "XTKS", "calendar": "XTKS", "timezone": "Asia/Tokyo", "exchange_name": "Tokyo Stock Exchange", "region": "APAC"},
    "HK": {"market": "XHKG", "calendar": "XHKG", "timezone": "Asia/Hong_Kong", "exchange_name": "Hong Kong Exchanges", "region": "APAC"},
}


_EXCHANGE_RULES: Dict[str, Dict[str, str]] = {
    "NMS": {"market": "XNAS", "calendar": "XNAS", "timezone": "America/New_York", "exchange_name": "NasdaqGS", "region": "USA"},
    "NGM": {"market": "XNAS", "calendar": "XNAS", "timezone": "America/New_York", "exchange_name": "NasdaqGM", "region": "USA"},
    "NCM": {"market": "XNAS", "calendar": "XNAS", "timezone": "America/New_York", "exchange_name": "NasdaqCM", "region": "USA"},
    "NYQ": {"market": "XNYS", "calendar": "XNYS", "timezone": "America/New_York", "exchange_name": "NYSE", "region": "USA"},
    "ASE": {"market": "XNYS", "calendar": "XNYS", "timezone": "America/New_York", "exchange_name": "NYSE American", "region": "USA"},
    "PCX": {"market": "XNYS", "calendar": "XNYS", "timezone": "America/New_York", "exchange_name": "NYSE Arca", "region": "USA"},
    "BTS": {"market": "BATS", "calendar": "XNYS", "timezone": "America/New_York", "exchange_name": "Cboe BZX", "region": "USA"},
}


_QUOTE_TYPE_ALIASES: Dict[str, str] = {
    "EQUITY": "equity",
    "ETF": "etf",
    "MUTUALFUND": "fund",
    "INDEX": "index",
    "CRYPTOCURRENCY": "crypto",
    "CURRENCY": "fx",
    "FUTURE": "future",
    "OPTION": "option",
}


_COMMON_NAME_STOPWORDS = {
    "sa",
    "s.a",
    "plc",
    "inc",
    "corp",
    "corporation",
    "holdings",
    "holding",
    "group",
    "adr",
    "nv",
    "ag",
    "se",
    "spa",
    "s.p.a",
    "the",
}


_REGION_SUFFIX_GROUPS: Dict[str, List[str]] = {
    "EUROPE": ["MC", "PA", "AS", "BR", "MI", "DE", "F", "L", "SW", "ST", "HE", "CO", "OL", "WA", "PR", "VI"],
    "AMERICAS": ["TO"],
    "APAC": ["AX", "T", "HK"],
}

_ALL_KNOWN_SUFFIXES: List[str] = [suffix for suffix in _SUFFIX_RULES.keys()]


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_get_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        return dict(value)
    except Exception:
        return {}


def _extract_suffix(symbol: str) -> Optional[str]:
    if "." not in symbol:
        return None
    return symbol.rsplit(".", 1)[-1].upper()


def _base_symbol(symbol: str) -> str:
    return symbol.rsplit(".", 1)[0].upper()


def _normalize_company_name(name: Optional[str]) -> str:
    if not name:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", str(name).lower())
    tokens = [token for token in cleaned.split() if len(token) > 1 and token not in _COMMON_NAME_STOPWORDS]
    return " ".join(tokens)


def _name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _looks_like_fx(symbol: str, quote_type: Optional[str]) -> bool:
    s = symbol.upper()
    return bool(quote_type == "CURRENCY" or s.endswith("=X"))


def _looks_like_crypto(symbol: str, quote_type: Optional[str]) -> bool:
    s = symbol.upper()
    return bool(quote_type == "CRYPTOCURRENCY" or "-USD" in s or "-EUR" in s or s.endswith("-BTC"))


def _infer_asset_type(symbol: str, quote_type: Optional[str]) -> str:
    qt = (quote_type or "").upper()
    if _looks_like_crypto(symbol, qt):
        return "crypto"
    if _looks_like_fx(symbol, qt):
        return "fx"
    return _QUOTE_TYPE_ALIASES.get(qt, "equity")


def _build_asset_flags(asset_type: str) -> Dict[str, Any]:
    asset = asset_type.lower()
    if asset == "crypto":
        return {
            "asset_family": "digital_asset",
            "is_24_7": True,
            "volume_expected": True,
            "corporate_actions_expected": False,
            "calendar_validation_supported": False,
            "dq_profile": "crypto",
        }
    if asset == "fx":
        return {
            "asset_family": "macro",
            "is_24_7": False,
            "volume_expected": False,
            "corporate_actions_expected": False,
            "calendar_validation_supported": False,
            "dq_profile": "fx",
        }
    if asset == "index":
        return {
            "asset_family": "market_index",
            "is_24_7": False,
            "volume_expected": False,
            "corporate_actions_expected": False,
            "calendar_validation_supported": True,
            "dq_profile": "index",
        }
    if asset == "fund":
        return {
            "asset_family": "fund",
            "is_24_7": False,
            "volume_expected": False,
            "corporate_actions_expected": False,
            "calendar_validation_supported": True,
            "dq_profile": "fund",
        }
    if asset == "etf":
        return {
            "asset_family": "fund",
            "is_24_7": False,
            "volume_expected": True,
            "corporate_actions_expected": True,
            "calendar_validation_supported": True,
            "dq_profile": "etf",
        }
    if asset == "future":
        return {
            "asset_family": "derivative",
            "is_24_7": False,
            "volume_expected": True,
            "corporate_actions_expected": False,
            "calendar_validation_supported": False,
            "dq_profile": "future",
        }
    return {
        "asset_family": "cash_equity",
        "is_24_7": False,
        "volume_expected": True,
        "corporate_actions_expected": True,
        "calendar_validation_supported": True,
        "dq_profile": "equity",
    }


def _resolve_calendar_from_metadata(
    symbol: str,
    history_metadata: Dict[str, Any],
    info: Dict[str, Any],
    fast_info: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    suffix = _extract_suffix(symbol)
    if suffix and suffix in _SUFFIX_RULES:
        rule = _SUFFIX_RULES[suffix]
        return {
            "market": rule.get("market"),
            "calendar": rule.get("calendar"),
            "timezone": rule.get("timezone"),
            "exchange_name": rule.get("exchange_name"),
            "exchange_code": suffix,
            "region": rule.get("region"),
            "source": f"suffix:{suffix}",
        }

    exchange_code = (
        _normalize_text(info.get("exchange"))
        or _normalize_text(history_metadata.get("exchangeName"))
        or _normalize_text(fast_info.get("exchange"))
    )
    if exchange_code:
        code = exchange_code.upper()
        if code in _EXCHANGE_RULES:
            rule = _EXCHANGE_RULES[code]
            return {
                "market": rule.get("market"),
                "calendar": rule.get("calendar"),
                "timezone": rule.get("timezone"),
                "exchange_name": rule.get("exchange_name"),
                "exchange_code": code,
                "region": rule.get("region"),
                "source": f"exchange:{code}",
            }

    timezone = (
        _normalize_text(history_metadata.get("exchangeTimezoneName"))
        or _normalize_text(info.get("exchangeTimezoneName"))
        or _normalize_text(fast_info.get("timezone"))
    )
    return {
        "market": None,
        "calendar": None,
        "timezone": timezone,
        "exchange_name": _normalize_text(info.get("fullExchangeName")) or _normalize_text(info.get("exchange")),
        "exchange_code": exchange_code,
        "region": "UNKNOWN",
        "source": "metadata_only",
    }


def _run_with_timeout(
    supplier: Callable[[], Any],
    *,
    timeout_seconds: float | None,
) -> Any:
    if timeout_seconds is None:
        return supplier()

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}
    completed = Event()

    def _target() -> None:
        try:
            result["value"] = supplier()
        except BaseException as exc:  # pragma: no cover - defensive handoff from worker thread
            error["exc"] = exc
        finally:
            completed.set()

    worker = Thread(target=_target, daemon=True)
    worker.start()
    if not completed.wait(float(timeout_seconds)):
        raise TimeoutError(f"metadata lookup exceeded timeout={timeout_seconds}s")
    if "exc" in error:
        raise error["exc"]
    return result.get("value")


def _metadata_warning(
    *,
    symbol: str,
    query: str,
    timeout_seconds: float | None,
    reason: str,
) -> Dict[str, Any]:
    return {
        "code": "metadata_timeout" if reason == "timeout" else "metadata_lookup_error",
        "symbol": symbol,
        "query": query,
        "timeout_seconds": timeout_seconds,
        "reason": reason,
        "message": (
            f"Metadata query '{query}' for {symbol} exceeded timeout={timeout_seconds}s."
            if reason == "timeout"
            else f"Metadata query '{query}' for {symbol} failed."
        ),
    }


def _fetch_raw_metadata(symbol: str, metadata_timeout: float | None = None) -> RawMetadataResult:
    ticker = yf.Ticker(symbol)
    warnings: List[str] = []
    structured_warnings: List[Dict[str, Any]] = []
    query_trace: List[Dict[str, Any]] = []

    def _lookup(query: str, supplier: Callable[[], Any]) -> Dict[str, Any]:
        try:
            value = _run_with_timeout(supplier, timeout_seconds=metadata_timeout)
            payload = _safe_get_dict(value)
            query_trace.append({"query": query, "status": "ok", "from_cache": False, "keys": sorted(payload.keys())})
            return payload
        except TimeoutError:
            warning = _metadata_warning(
                symbol=symbol,
                query=query,
                timeout_seconds=metadata_timeout,
                reason="timeout",
            )
            warnings.append(warning["message"])
            structured_warnings.append(warning)
            query_trace.append({"query": query, "status": "timeout", "from_cache": False})
        except Exception as exc:
            logger.debug("%s lookup failed for %s: %s", query, symbol, exc)
            warning = _metadata_warning(
                symbol=symbol,
                query=query,
                timeout_seconds=metadata_timeout,
                reason="error",
            )
            warning["error"] = str(exc)
            structured_warnings.append(warning)
            query_trace.append({"query": query, "status": "error", "from_cache": False, "error": str(exc)})
        return {}

    history_metadata = _lookup("history_metadata", lambda: getattr(ticker, "history_metadata", {}))
    if not history_metadata and hasattr(ticker, "get_history_metadata"):
        history_metadata = _lookup("get_history_metadata", ticker.get_history_metadata)

    fast_info = _lookup("fast_info", lambda: getattr(ticker, "fast_info", {}))
    if not fast_info and hasattr(ticker, "get_fast_info"):
        fast_info = _lookup("get_fast_info", ticker.get_fast_info)

    info = _lookup("info", lambda: getattr(ticker, "info", {}))

    return RawMetadataResult(
        history_metadata=history_metadata,
        fast_info=fast_info,
        info=info,
        warnings=warnings,
        structured_warnings=structured_warnings,
        query_trace=query_trace,
    )


def _snapshot_symbol(symbol: str, metadata_timeout: float | None = None) -> Dict[str, Any]:
    raw = _fetch_raw_metadata(symbol, metadata_timeout=metadata_timeout)
    history_metadata = raw.history_metadata
    fast_info = raw.fast_info
    info = raw.info

    quote_type = (
        _normalize_text(info.get("quoteType"))
        or _normalize_text(history_metadata.get("instrumentType"))
        or "UNKNOWN"
    ).upper()
    asset_type = _infer_asset_type(symbol, quote_type)
    calendar_info = _resolve_calendar_from_metadata(symbol, history_metadata, info, fast_info)
    company_name = (
        _normalize_text(info.get("longName"))
        or _normalize_text(info.get("shortName"))
        or _normalize_text(history_metadata.get("instrumentType"))
        or symbol
    )

    return {
        "symbol": symbol.upper(),
        "resolved_symbol": (
            _normalize_text(info.get("symbol"))
            or _normalize_text(history_metadata.get("symbol"))
            or symbol.upper()
        ),
        "quote_type": quote_type,
        "asset_type": asset_type,
        "market": calendar_info.get("market"),
        "calendar": calendar_info.get("calendar"),
        "timezone": calendar_info.get("timezone"),
        "exchange_name": calendar_info.get("exchange_name") or _normalize_text(info.get("fullExchangeName")),
        "exchange_code": calendar_info.get("exchange_code"),
        "region": calendar_info.get("region") or "UNKNOWN",
        "currency": _normalize_text(fast_info.get("currency")) or _normalize_text(info.get("currency")),
        "company_name": company_name,
        "company_key": _normalize_company_name(company_name),
        "source": calendar_info.get("source", "metadata"),
        "raw_metadata": {
            "history_metadata": history_metadata,
            "fast_info": fast_info,
            "info": info,
        },
        "warnings": list(raw.warnings),
        "structured_warnings": list(raw.structured_warnings),
        "query_trace": list(raw.query_trace),
        "metadata_present": bool(history_metadata or fast_info or info),
    }


def _candidate_suffixes(listing_preference: str, base_region: Optional[str] = None) -> List[str]:
    normalized_region = str(base_region or "UNKNOWN").upper()

    if listing_preference == "prefer_europe":
        return list(_REGION_SUFFIX_GROUPS["EUROPE"])

    if listing_preference == "home_market":
        if normalized_region == "EUROPE":
            prioritized = list(_REGION_SUFFIX_GROUPS["EUROPE"])
        elif normalized_region in _REGION_SUFFIX_GROUPS:
            prioritized = list(_REGION_SUFFIX_GROUPS[normalized_region])
        else:
            prioritized = []

        remaining = [suffix for suffix in _ALL_KNOWN_SUFFIXES if suffix not in prioritized]
        return prioritized + remaining

    return []


def _ordered_unique(values: List[str], *, preserve_case: bool = False) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        text = str(value or "").strip()
        normalized = text.upper()
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(text if preserve_case else normalized)
    return ordered


_CONTEXT_CACHE_VERSION = 1


def _safe_symbol_cache_key(symbol: str) -> str:
    normalized = str(symbol or "").strip().upper()
    return re.sub(r"[^A-Z0-9._-]+", "_", normalized)


def _cache_file_path(cache_dir: Path, symbol: str) -> Path:
    return Path(cache_dir) / f"{_safe_symbol_cache_key(symbol)}.json"


def _parse_cache_timestamp(raw: Any) -> datetime | None:
    text = _normalize_text(raw)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_cache_expired(saved_at_utc: Any, ttl_seconds: int) -> bool:
    if ttl_seconds <= 0:
        return False
    saved_at = _parse_cache_timestamp(saved_at_utc)
    if saved_at is None:
        return True
    age_seconds = (datetime.now(timezone.utc) - saved_at).total_seconds()
    return age_seconds > float(ttl_seconds)


def _candidate_symbols(
    requested_symbol: str,
    listing_preference: str,
    base_region: Optional[str] = None,
) -> List[str]:
    requested = str(requested_symbol or "").strip().upper()
    base = _base_symbol(requested)
    normalized_region = str(base_region or "UNKNOWN").upper()

    candidates: List[str] = []

    if listing_preference == "prefer_usa":
        if base != requested:
            candidates.append(base)
        return _ordered_unique(candidates)

    suffixes = _candidate_suffixes(listing_preference, base_region=normalized_region)
    candidates.extend(f"{base}.{suffix}" for suffix in suffixes)

    if listing_preference == "home_market" and normalized_region == "UNKNOWN" and base != requested:
        candidates.append(base)

    return [symbol for symbol in _ordered_unique(candidates) if symbol != requested]


def _choose_preferred_snapshot(
    base_snapshot: Dict[str, Any],
    listing_preference: str,
    metadata_timeout: float | None = None,
    candidate_limit: int | None = None,
    snapshot_loader: Callable[[str], Dict[str, Any]] | None = None,
) -> Tuple[Dict[str, Any], List[str], str, List[Dict[str, Any]]]:
    warnings: List[str] = []
    requested_symbol = str(base_snapshot["symbol"] or "").upper()
    tested_snapshots: List[Dict[str, Any]] = []
    load_snapshot = snapshot_loader or (lambda symbol: _snapshot_symbol(symbol, metadata_timeout=metadata_timeout))

    if listing_preference == "exact_symbol":
        return base_snapshot, warnings, "high", tested_snapshots

    if base_snapshot["asset_type"] not in {"equity", "etf"}:
        return base_snapshot, warnings, "high", tested_snapshots

    original_region = str(base_snapshot.get("region") or "UNKNOWN").upper()
    base_name = base_snapshot.get("company_key", "")

    if listing_preference == "prefer_europe" and original_region == "EUROPE":
        return base_snapshot, warnings, "high", tested_snapshots
    if listing_preference == "prefer_usa" and original_region == "USA":
        return base_snapshot, warnings, "high", tested_snapshots
    if listing_preference == "home_market" and original_region not in {"USA", "UNKNOWN"}:
        return base_snapshot, warnings, "high", tested_snapshots

    target_region: Optional[str] = None
    if listing_preference == "prefer_europe":
        target_region = "EUROPE"
    elif listing_preference == "prefer_usa":
        target_region = "USA"

    exact_score = 10.0
    if target_region and original_region == target_region:
        exact_score += 5.0
    elif listing_preference == "home_market" and original_region not in {"USA", "UNKNOWN"}:
        exact_score += 5.0

    best_snapshot = base_snapshot
    best_score = exact_score

    candidate_symbols = _candidate_symbols(requested_symbol, listing_preference, base_region=original_region)
    if candidate_limit is not None:
        candidate_symbols = candidate_symbols[: max(0, int(candidate_limit))]

    for candidate_symbol in candidate_symbols:
        try:
            snapshot = load_snapshot(candidate_symbol)
        except Exception as exc:
            logger.debug("Candidate lookup failed for %s: %s", candidate_symbol, exc)
            continue
        tested_snapshots.append(snapshot)

        if not snapshot.get("metadata_present"):
            continue
        if snapshot.get("asset_type") != base_snapshot.get("asset_type"):
            continue

        candidate_region = str(snapshot.get("region") or "UNKNOWN").upper()
        if listing_preference == "prefer_europe" and candidate_region != "EUROPE":
            continue
        if listing_preference == "prefer_usa" and candidate_region != "USA":
            continue

        similarity = _name_similarity(base_name, snapshot.get("company_key", ""))
        region_bonus = 0.0
        if listing_preference == "prefer_europe" and candidate_region == "EUROPE":
            region_bonus = 20.0
        elif listing_preference == "prefer_usa" and candidate_region == "USA":
            region_bonus = 20.0
        elif listing_preference == "home_market":
            if original_region == "USA" and candidate_region not in {"USA", "UNKNOWN"}:
                region_bonus = 20.0
            elif original_region == "UNKNOWN" and candidate_region not in {"USA", "UNKNOWN"}:
                region_bonus = 12.0
            elif original_region not in {"USA", "UNKNOWN"} and candidate_region == original_region:
                region_bonus = 20.0

        market_bonus = 5.0 if snapshot.get("market") else 0.0
        score = similarity * 100.0 + region_bonus + market_bonus

        if similarity >= 0.55 and score > best_score + 10.0:
            best_snapshot = snapshot
            best_score = score

    if best_snapshot["symbol"] != base_snapshot["symbol"]:
        warnings.append(
            f"Se ha priorizado el listing {best_snapshot['symbol']} en lugar de {base_snapshot['symbol']} por la preferencia {listing_preference}."
        )
        return best_snapshot, warnings, "medium", tested_snapshots

    if listing_preference == "prefer_usa" and original_region != "USA":
        warnings.append(
            "No se encontró una alternativa USA con suficiente confianza. Se intentó priorizar el símbolo base sin sufijo, que es la forma habitual de los listings primarios USA en Yahoo Finance, pero no se pudo validar una coincidencia suficientemente fiable."
        )
    elif listing_preference != "exact_symbol":
        warnings.append(
            "No se encontró un listing alternativo con suficiente confianza; se mantiene el símbolo exacto solicitado."
        )
    return base_snapshot, warnings, "medium", tested_snapshots


class ContextResolver:
    def __init__(
        self,
        *,
        metadata_timeout: float | None = None,
        candidate_limit: int = DEFAULT_METADATA_CANDIDATE_LIMIT,
        cache_dir: Path | None = None,
        cache_ttl_seconds: int | None = DEFAULT_CONTEXT_CACHE_TTL_SECONDS,
    ) -> None:
        self.metadata_timeout = metadata_timeout
        self.candidate_limit = max(1, int(candidate_limit))
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser().resolve()
        self.cache_ttl_seconds = (
            DEFAULT_CONTEXT_CACHE_TTL_SECONDS
            if cache_ttl_seconds is None
            else max(0, int(cache_ttl_seconds))
        )
        self._snapshot_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = Lock()
        self._symbol_locks: Dict[str, Lock] = {}
        if self.cache_dir is not None and self.cache_ttl_seconds != 0:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "metadata_queries": 0,
            "resolutions": 0,
            "persistent_cache_hits": 0,
            "persistent_cache_misses": 0,
            "persistent_cache_writes": 0,
        }

    def _increment_metric(self, key: str, amount: int = 1) -> None:
        with self._cache_lock:
            self.metrics[key] = int(self.metrics.get(key, 0)) + int(amount)

    def _symbol_lock(self, symbol: str) -> Lock:
        with self._cache_lock:
            lock = self._symbol_locks.get(symbol)
            if lock is None:
                lock = Lock()
                self._symbol_locks[symbol] = lock
            return lock

    @staticmethod
    def _clone_snapshot(snapshot: Dict[str, Any], *, cache_source: str) -> Dict[str, Any]:
        cloned = dict(snapshot)
        query_trace = [dict(item) for item in cloned.get("query_trace", []) if isinstance(item, dict)]
        if cache_source != "live":
            query_trace = [{**item, "from_cache": True} for item in query_trace]
        cloned["query_trace"] = query_trace
        cloned["cache_hit"] = cache_source != "live"
        cloned["cache_source"] = cache_source
        return cloned

    def _load_persistent_snapshot(self, symbol: str) -> Dict[str, Any] | None:
        if self.cache_dir is None or self.cache_ttl_seconds == 0:
            return None

        cache_file = _cache_file_path(self.cache_dir, symbol)
        if not cache_file.exists():
            self._increment_metric("persistent_cache_misses")
            return None

        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("Persistent context cache could not be read for %s: %s", symbol, exc)
            self._increment_metric("persistent_cache_misses")
            return None

        if int(payload.get("version", 0) or 0) != _CONTEXT_CACHE_VERSION:
            self._increment_metric("persistent_cache_misses")
            return None
        if _is_cache_expired(payload.get("saved_at_utc"), self.cache_ttl_seconds):
            try:
                cache_file.unlink(missing_ok=True)
            except OSError:
                pass
            self._increment_metric("persistent_cache_misses")
            return None

        snapshot = payload.get("snapshot")
        if not isinstance(snapshot, dict):
            self._increment_metric("persistent_cache_misses")
            return None

        restored = dict(snapshot)
        restored["symbol"] = str(restored.get("symbol") or symbol).strip().upper()
        self._increment_metric("persistent_cache_hits")
        return restored

    def _store_persistent_snapshot(self, symbol: str, snapshot: Dict[str, Any]) -> None:
        if self.cache_dir is None or self.cache_ttl_seconds == 0:
            return

        cache_file = _cache_file_path(self.cache_dir, symbol)
        payload = {
            "version": _CONTEXT_CACHE_VERSION,
            "saved_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "symbol": str(symbol or "").strip().upper(),
            "snapshot": dict(snapshot),
        }
        try:
            write_json(cache_file, payload, temp_dir=self.cache_dir)
        except Exception as exc:
            logger.debug("Persistent context cache could not be written for %s: %s", symbol, exc)
            return
        self._increment_metric("persistent_cache_writes")

    def _load_snapshot(self, symbol: str) -> Dict[str, Any]:
        normalized_symbol = str(symbol or "").strip().upper()
        with self._cache_lock:
            cached = self._snapshot_cache.get(normalized_symbol)
        if cached is not None:
            self._increment_metric("cache_hits")
            return self._clone_snapshot(cached, cache_source="memory")

        symbol_lock = self._symbol_lock(normalized_symbol)
        with symbol_lock:
            with self._cache_lock:
                cached = self._snapshot_cache.get(normalized_symbol)
            if cached is not None:
                self._increment_metric("cache_hits")
                return self._clone_snapshot(cached, cache_source="memory")

            self._increment_metric("cache_misses")
            persistent_snapshot = self._load_persistent_snapshot(normalized_symbol)
            if persistent_snapshot is not None:
                with self._cache_lock:
                    self._snapshot_cache[normalized_symbol] = dict(persistent_snapshot)
                return self._clone_snapshot(persistent_snapshot, cache_source="persistent")

            snapshot = dict(_snapshot_symbol(normalized_symbol, metadata_timeout=self.metadata_timeout))
            with self._cache_lock:
                self._snapshot_cache[normalized_symbol] = dict(snapshot)
                self.metrics["metadata_queries"] += len(snapshot.get("query_trace", []))
            self._store_persistent_snapshot(normalized_symbol, snapshot)
            return self._clone_snapshot(snapshot, cache_source="live")

    def resolve(
        self,
        symbol: str,
        *,
        market_override: Optional[str] = None,
        listing_preference: str = "exact_symbol",
    ) -> InstrumentContext:
        normalized_symbol = str(symbol or "").strip().upper()
        if not normalized_symbol:
            raise ValueError("Ticker symbol cannot be empty for context resolution.")

        listing_preference = str(listing_preference or "exact_symbol").strip().lower()
        if listing_preference not in {"exact_symbol", "home_market", "prefer_europe", "prefer_usa"}:
            raise ValueError(f"Unsupported listing_preference: {listing_preference}")

        base_snapshot = self._load_snapshot(normalized_symbol)
        chosen_snapshot, preference_warnings, confidence, tested_snapshots = _choose_preferred_snapshot(
            base_snapshot,
            listing_preference,
            metadata_timeout=self.metadata_timeout,
            candidate_limit=self.candidate_limit,
            snapshot_loader=self._load_snapshot,
        )
        attempted_snapshots = [base_snapshot, *tested_snapshots]

        quote_type = str(chosen_snapshot.get("quote_type") or "UNKNOWN").upper()
        asset_type = str(chosen_snapshot.get("asset_type") or _infer_asset_type(normalized_symbol, quote_type)).lower()
        flags = _build_asset_flags(asset_type)

        market = _normalize_text(market_override) or chosen_snapshot.get("market")
        calendar = _normalize_text(market_override) or chosen_snapshot.get("calendar")
        timezone = chosen_snapshot.get("timezone")
        currency = chosen_snapshot.get("currency")
        region = str(chosen_snapshot.get("region") or "UNKNOWN").upper()

        warnings: List[str] = list(preference_warnings)
        structured_warnings: List[Dict[str, Any]] = []
        resolution_trace: List[Dict[str, Any]] = []
        for rank, snapshot in enumerate(attempted_snapshots):
            warnings.extend(list(snapshot.get("warnings", [])))
            structured_warnings.extend(list(snapshot.get("structured_warnings", [])))
            resolution_trace.append(
                {
                    "symbol": snapshot.get("symbol"),
                    "candidate_rank": rank,
                    "selected": str(snapshot.get("symbol") or "").upper()
                    == str(chosen_snapshot.get("symbol") or "").upper(),
                    "cache_hit": bool(snapshot.get("cache_hit")),
                    "cache_source": str(snapshot.get("cache_source") or "live"),
                    "metadata_present": bool(snapshot.get("metadata_present")),
                    "query_trace": list(snapshot.get("query_trace", [])),
                }
            )
        inference_sources = [str(chosen_snapshot.get("source", "unknown")), f"listing_preference:{listing_preference}", "quote_type"]

        if flags["calendar_validation_supported"] and not calendar:
            warnings.append(
                "No se pudo inferir un calendario de mercado exacto; la DQ usarÃ¡ validaciÃ³n sin calendario oficial."
            )

        if asset_type in {"equity", "etf"} and not timezone:
            warnings.append("No se pudo inferir la zona horaria del mercado.")
        if quote_type == "UNKNOWN":
            warnings.append("quoteType no disponible; se asumiÃ³ perfil por heurÃ­stica.")

        if not base_snapshot.get("metadata_present"):
            confidence = "low"
            warnings.append("No se obtuvo metadata de Yahoo para enriquecer el contexto del instrumento.")

        raw_metadata = {
            "requested": base_snapshot.get("raw_metadata", {}),
            "preferred": chosen_snapshot.get("raw_metadata", {}),
        }
        self._increment_metric("resolutions")
        resolver_metrics = {
            "metadata_timeout": self.metadata_timeout,
            "candidate_limit": self.candidate_limit,
            "candidates_tested": len(tested_snapshots),
            "cache_hits": self.metrics["cache_hits"],
            "cache_misses": self.metrics["cache_misses"],
            "metadata_queries": self.metrics["metadata_queries"],
            "resolutions": self.metrics["resolutions"],
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "persistent_cache_enabled": bool(self.cache_dir) and self.cache_ttl_seconds != 0,
            "persistent_cache_hits": self.metrics["persistent_cache_hits"],
            "persistent_cache_misses": self.metrics["persistent_cache_misses"],
            "persistent_cache_writes": self.metrics["persistent_cache_writes"],
        }

        return InstrumentContext(
            requested_symbol=normalized_symbol,
            preferred_symbol=str(chosen_snapshot.get("symbol") or normalized_symbol),
            resolved_symbol=str(chosen_snapshot.get("resolved_symbol") or normalized_symbol),
            listing_preference=listing_preference,
            quote_type=quote_type,
            asset_type=asset_type,
            asset_family=flags["asset_family"],
            market=market,
            calendar=calendar,
            timezone=timezone,
            currency=currency,
            exchange_name=chosen_snapshot.get("exchange_name"),
            exchange_code=chosen_snapshot.get("exchange_code"),
            region=region,
            is_24_7=bool(flags["is_24_7"]),
            volume_expected=bool(flags["volume_expected"]),
            corporate_actions_expected=bool(flags["corporate_actions_expected"]),
            calendar_validation_supported=bool(flags["calendar_validation_supported"] and bool(calendar)),
            dq_profile=str(flags["dq_profile"]),
            confidence=confidence,
            inference_sources=inference_sources,
            warnings=_ordered_unique(warnings, preserve_case=True),
            structured_warnings=structured_warnings,
            resolution_trace=resolution_trace,
            resolver_metrics=resolver_metrics,
            raw_metadata=raw_metadata,
        )


def resolve_instrument_context(
    symbol: str,
    market_override: Optional[str] = None,
    listing_preference: str = "exact_symbol",
    metadata_timeout: float | None = None,
    resolver: ContextResolver | None = None,
    candidate_limit: int = DEFAULT_METADATA_CANDIDATE_LIMIT,
    cache_dir: Path | None = None,
    cache_ttl_seconds: int | None = DEFAULT_CONTEXT_CACHE_TTL_SECONDS,
) -> InstrumentContext:
    active_resolver = resolver or ContextResolver(
        metadata_timeout=metadata_timeout,
        candidate_limit=candidate_limit,
        cache_dir=cache_dir,
        cache_ttl_seconds=cache_ttl_seconds,
    )
    return active_resolver.resolve(
        symbol,
        market_override=market_override,
        listing_preference=listing_preference,
    )


    if flags["calendar_validation_supported"] and not calendar:
        warnings.append(
            "No se pudo inferir un calendario de mercado exacto; la DQ usará validación sin calendario oficial."
        )

    if asset_type in {"equity", "etf"} and not timezone:
        warnings.append("No se pudo inferir la zona horaria del mercado.")
    if quote_type == "UNKNOWN":
        warnings.append("quoteType no disponible; se asumió perfil por heurística.")

    if not base_snapshot.get("metadata_present"):
        confidence = "low"
        warnings.append("No se obtuvo metadata de Yahoo para enriquecer el contexto del instrumento.")

    raw_metadata = {
        "requested": base_snapshot.get("raw_metadata", {}),
        "preferred": chosen_snapshot.get("raw_metadata", {}),
    }
    resolver_metrics = {
        "metadata_timeout": metadata_timeout,
        "metadata_queries": len(base_snapshot.get("query_trace", [])) + len(chosen_snapshot.get("query_trace", []))
        if chosen_snapshot is not base_snapshot
        else len(base_snapshot.get("query_trace", [])),
        "cache_hits": 0,
        "cache_misses": 0,
    }

    return InstrumentContext(
        requested_symbol=normalized_symbol,
        preferred_symbol=str(chosen_snapshot.get("symbol") or normalized_symbol),
        resolved_symbol=str(chosen_snapshot.get("resolved_symbol") or normalized_symbol),
        listing_preference=listing_preference,
        quote_type=quote_type,
        asset_type=asset_type,
        asset_family=flags["asset_family"],
        market=market,
        calendar=calendar,
        timezone=timezone,
        currency=currency,
        exchange_name=chosen_snapshot.get("exchange_name"),
        exchange_code=chosen_snapshot.get("exchange_code"),
        region=region,
        is_24_7=bool(flags["is_24_7"]),
        volume_expected=bool(flags["volume_expected"]),
        corporate_actions_expected=bool(flags["corporate_actions_expected"]),
        calendar_validation_supported=bool(flags["calendar_validation_supported"] and bool(calendar)),
        dq_profile=str(flags["dq_profile"]),
        confidence=confidence,
        inference_sources=inference_sources,
        warnings=warnings,
        structured_warnings=structured_warnings,
        resolution_trace=resolution_trace,
        resolver_metrics=resolver_metrics,
        raw_metadata=raw_metadata,
    )


def build_dq_context_payload(context: InstrumentContext) -> Dict[str, Any]:
    return {
        "asset_type": context.asset_type,
        "asset_family": context.asset_family,
        "quote_type": context.quote_type,
        "market": context.market,
        "calendar": context.calendar,
        "timezone": context.timezone,
        "currency": context.currency,
        "exchange_name": context.exchange_name,
        "exchange_code": context.exchange_code,
        "region": context.region,
        "requested_symbol": context.requested_symbol,
        "preferred_symbol": context.preferred_symbol,
        "listing_preference": context.listing_preference,
        "is_24_7": context.is_24_7,
        "volume_expected": context.volume_expected,
        "corporate_actions_expected": context.corporate_actions_expected,
        "calendar_validation_supported": context.calendar_validation_supported,
        "dq_profile": context.dq_profile,
        "confidence": context.confidence,
        "warnings": list(context.warnings),
        "structured_warnings": list(context.structured_warnings),
        "inference_sources": list(context.inference_sources),
        "resolution_trace": list(context.resolution_trace),
        "resolver_metrics": dict(context.resolver_metrics),
    }
