from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yfinance as yf

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
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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


def _fetch_raw_metadata(symbol: str) -> Dict[str, Dict[str, Any]]:
    ticker = yf.Ticker(symbol)

    history_metadata: Dict[str, Any] = {}
    try:
        history_metadata = _safe_get_dict(getattr(ticker, "history_metadata", {}))
        if not history_metadata and hasattr(ticker, "get_history_metadata"):
            history_metadata = _safe_get_dict(ticker.get_history_metadata())
    except Exception as exc:
        logger.debug("history_metadata lookup failed for %s: %s", symbol, exc)

    fast_info: Dict[str, Any] = {}
    try:
        fast_info = _safe_get_dict(getattr(ticker, "fast_info", {}))
        if not fast_info and hasattr(ticker, "get_fast_info"):
            fast_info = _safe_get_dict(ticker.get_fast_info())
    except Exception as exc:
        logger.debug("fast_info lookup failed for %s: %s", symbol, exc)

    info: Dict[str, Any] = {}
    try:
        info = _safe_get_dict(getattr(ticker, "info", {}))
    except Exception as exc:
        logger.debug("info lookup failed for %s: %s", symbol, exc)

    return {
        "history_metadata": history_metadata,
        "fast_info": fast_info,
        "info": info,
    }


def _snapshot_symbol(symbol: str) -> Dict[str, Any]:
    raw = _fetch_raw_metadata(symbol)
    history_metadata = raw["history_metadata"]
    fast_info = raw["fast_info"]
    info = raw["info"]

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
        "raw_metadata": raw,
        "metadata_present": bool(raw["history_metadata"] or raw["fast_info"] or raw["info"]),
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


def _ordered_unique(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        normalized = str(value or "").strip().upper()
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


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


def _choose_preferred_snapshot(base_snapshot: Dict[str, Any], listing_preference: str) -> Tuple[Dict[str, Any], List[str], str]:
    warnings: List[str] = []
    requested_symbol = str(base_snapshot["symbol"] or "").upper()

    if listing_preference == "exact_symbol":
        return base_snapshot, warnings, "high"

    if base_snapshot["asset_type"] not in {"equity", "etf"}:
        return base_snapshot, warnings, "high"

    original_region = str(base_snapshot.get("region") or "UNKNOWN").upper()
    base_name = base_snapshot.get("company_key", "")

    if listing_preference == "prefer_europe" and original_region == "EUROPE":
        return base_snapshot, warnings, "high"
    if listing_preference == "prefer_usa" and original_region == "USA":
        return base_snapshot, warnings, "high"
    if listing_preference == "home_market" and original_region not in {"USA", "UNKNOWN"}:
        return base_snapshot, warnings, "high"

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

    for candidate_symbol in _candidate_symbols(requested_symbol, listing_preference, base_region=original_region):
        try:
            snapshot = _snapshot_symbol(candidate_symbol)
        except Exception as exc:
            logger.debug("Candidate lookup failed for %s: %s", candidate_symbol, exc)
            continue

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
        return best_snapshot, warnings, "medium"

    if listing_preference == "prefer_usa" and original_region != "USA":
        warnings.append(
            "No se encontró una alternativa USA con suficiente confianza. Se intentó priorizar el símbolo base sin sufijo, que es la forma habitual de los listings primarios USA en Yahoo Finance, pero no se pudo validar una coincidencia suficientemente fiable."
        )
    elif listing_preference != "exact_symbol":
        warnings.append(
            "No se encontró un listing alternativo con suficiente confianza; se mantiene el símbolo exacto solicitado."
        )
    return base_snapshot, warnings, "medium"


def resolve_instrument_context(
    symbol: str,
    market_override: Optional[str] = None,
    listing_preference: str = "exact_symbol",
) -> InstrumentContext:
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        raise ValueError("Ticker symbol cannot be empty for context resolution.")

    listing_preference = str(listing_preference or "exact_symbol").strip().lower()
    if listing_preference not in {"exact_symbol", "home_market", "prefer_europe", "prefer_usa"}:
        raise ValueError(f"Unsupported listing_preference: {listing_preference}")

    base_snapshot = _snapshot_symbol(normalized_symbol)
    chosen_snapshot, preference_warnings, confidence = _choose_preferred_snapshot(base_snapshot, listing_preference)

    quote_type = str(chosen_snapshot.get("quote_type") or "UNKNOWN").upper()
    asset_type = str(chosen_snapshot.get("asset_type") or _infer_asset_type(normalized_symbol, quote_type)).lower()
    flags = _build_asset_flags(asset_type)

    market = _normalize_text(market_override) or chosen_snapshot.get("market")
    calendar = _normalize_text(market_override) or chosen_snapshot.get("calendar")
    timezone = chosen_snapshot.get("timezone")
    currency = chosen_snapshot.get("currency")
    region = str(chosen_snapshot.get("region") or "UNKNOWN").upper()

    warnings: List[str] = list(preference_warnings)
    inference_sources = [str(chosen_snapshot.get("source", "unknown")), f"listing_preference:{listing_preference}", "quote_type"]

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
        "inference_sources": list(context.inference_sources),
    }

