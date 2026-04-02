from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode

import pandas as pd
import requests

from dataset_core.external_sources.base import (
    ExternalSourceAuthError,
    ExternalSourceCoverageError,
    ExternalSourceNetworkError,
    ExternalSourceNotFoundError,
    ExternalSourcePayloadError,
    ExternalSourceRateLimitError,
    attach_source_metadata,
    filter_event_frame,
    filter_reference_frame,
)
from dataset_core.serialization import write_json
from dataset_core.settings import (
    DEFAULT_EODHD_BACKOFF_SECONDS,
    DEFAULT_EODHD_BASE_URL,
    DEFAULT_EODHD_CACHE_TTL_SECONDS,
    DEFAULT_EODHD_MAX_RETRIES,
    DEFAULT_EODHD_TIMEOUT_SECONDS,
    sanitize_secret_text,
)

_EMPTY_PRICE_FRAME = pd.DataFrame(
    columns=["date", "open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits"]
)
_EMPTY_EVENT_FRAME = pd.DataFrame(columns=["date", "dividends", "stock_splits"])


@dataclass(frozen=True)
class EODHDPayload:
    payload: object
    url: str
    cache_status: Literal["hit", "miss", "disabled"]
    endpoint: str


def _normalize_symbol_candidate(symbol: str) -> str:
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        raise ExternalSourceNotFoundError("EODHD could not resolve an empty symbol.")
    return normalized


def _load_symbol_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ExternalSourcePayloadError("EODHD symbol_map_file JSON must contain a simple object.")
        return {str(key).strip().upper(): str(value).strip().upper() for key, value in payload.items() if str(key).strip() and str(value).strip()}

    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    candidate_pairs = (
        ("symbol", "eodhd_symbol"),
        ("requested_symbol", "provider_symbol"),
        ("input_symbol", "mapped_symbol"),
    )
    for source_column, target_column in candidate_pairs:
        if source_column in frame.columns and target_column in frame.columns:
            mapping = {}
            for row in frame.itertuples(index=False):
                source = str(getattr(row, source_column) or "").strip().upper()
                target = str(getattr(row, target_column) or "").strip().upper()
                if source and target:
                    mapping[source] = target
            return mapping
    raise ExternalSourcePayloadError(
        "EODHD symbol_map_file CSV must define one of: symbol/eodhd_symbol, requested_symbol/provider_symbol or input_symbol/mapped_symbol."
    )


@dataclass(frozen=True)
class EODHDSymbolResolution:
    requested_symbol: str
    candidates: tuple[str, ...]
    strategy: str


class EODHDSymbolResolver:
    def __init__(self, *, exchange_hint: str | None = None, symbol_map_file: Path | None = None) -> None:
        self.exchange_hint = None if exchange_hint is None else str(exchange_hint).strip().upper()
        self.symbol_map = _load_symbol_map(symbol_map_file)

    def resolve_candidates(self, symbol: str) -> EODHDSymbolResolution:
        requested_symbol = _normalize_symbol_candidate(symbol)
        candidates: list[str] = []

        mapped_symbol = self.symbol_map.get(requested_symbol)
        if mapped_symbol:
            candidates.append(mapped_symbol)
            strategy = "explicit_symbol_map"
        else:
            candidates.append(requested_symbol)
            strategy = "exact_symbol"
            if "." not in requested_symbol and self.exchange_hint:
                candidates.append(f"{requested_symbol}.{self.exchange_hint}")
            if "." not in requested_symbol:
                candidates.append(f"{requested_symbol}.US")

        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = _normalize_symbol_candidate(candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)

        return EODHDSymbolResolution(
            requested_symbol=requested_symbol,
            candidates=tuple(deduped),
            strategy=strategy,
        )


def _build_eodhd_url(base_url: str, path: str, params: dict[str, object]) -> str:
    query = urlencode([(key, value) for key, value in params.items() if value is not None])
    return f"{base_url.rstrip('/')}{path}?{query}"


def _cache_key(path: str, params: dict[str, object]) -> str:
    payload = json.dumps({"path": path, "params": params}, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json_cache(path: Path, ttl_seconds: int) -> object | None:
    if not path.exists():
        return None
    if ttl_seconds == 0:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    cached_at = pd.to_datetime(payload.get("cached_at"), utc=True, errors="coerce")
    if pd.isna(cached_at):
        return None
    if ttl_seconds > 0:
        age_seconds = (pd.Timestamp.utcnow() - cached_at).total_seconds()
        if age_seconds > ttl_seconds:
            return None
    return payload.get("payload")


def _write_json_cache(path: Path, payload: object) -> None:
    cache_payload = {
        "cached_at": pd.Timestamp.utcnow().isoformat(),
        "payload": payload,
    }
    write_json(path, cache_payload)


def _extract_error_message(response: requests.Response) -> str:
    text = response.text.strip()
    try:
        payload = response.json()
    except ValueError:
        return sanitize_secret_text(text or f"HTTP {response.status_code}") or ""

    if isinstance(payload, dict):
        for key in ("error", "errors", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return sanitize_secret_text(value.strip()) or ""
            if isinstance(value, list) and value:
                return sanitize_secret_text("; ".join(str(item).strip() for item in value if str(item).strip())) or ""
    if isinstance(payload, str) and payload.strip():
        return sanitize_secret_text(payload.strip()) or ""
    return sanitize_secret_text(text or f"HTTP {response.status_code}") or ""


def _raise_http_error(response: requests.Response) -> None:
    message = _extract_error_message(response)
    status_code = int(response.status_code)
    lowered = message.lower()
    if status_code == 429 or "too many requests" in lowered:
        raise ExternalSourceRateLimitError(f"EODHD rate limit reached: {message}")
    if status_code == 404 or "not found" in lowered or "unknown" in lowered:
        raise ExternalSourceNotFoundError(f"EODHD could not resolve the requested symbol: {message}")
    if status_code == 403 and ("support" in lowered or "forbidden" in lowered):
        raise ExternalSourceCoverageError(f"EODHD rejected the request due to access or plan coverage: {message}")
    if status_code in {401, 403}:
        raise ExternalSourceAuthError(f"EODHD rejected the credentials or request: {message}")
    if status_code >= 500:
        raise ExternalSourceNetworkError(f"EODHD server error ({status_code}): {message}")
    raise ExternalSourcePayloadError(f"EODHD returned HTTP {status_code}: {message}")


def parse_eodhd_prices(payload: object) -> pd.DataFrame:
    if payload in (None, []):
        return _EMPTY_PRICE_FRAME.copy()
    if not isinstance(payload, list):
        raise ExternalSourcePayloadError("EODHD prices payload must be a JSON list.")

    frame = pd.DataFrame(payload)
    if frame.empty:
        return _EMPTY_PRICE_FRAME.copy()
    required_columns = {"date", "open", "high", "low", "close", "volume"}
    if not required_columns.issubset(frame.columns):
        missing = ", ".join(sorted(required_columns - set(frame.columns)))
        raise ExternalSourcePayloadError(f"EODHD prices payload is missing required columns: {missing}.")

    frame = frame.rename(columns={"adjusted_close": "adj_close"})
    for column in ("adj_close", "dividends", "stock_splits"):
        if column not in frame.columns:
            frame[column] = 0.0 if column != "adj_close" else pd.NA

    columns = ["date", "open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits"]
    frame = frame[columns].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"])
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["dividends"] = frame["dividends"].fillna(0.0)
    frame["stock_splits"] = frame["stock_splits"].fillna(0.0)
    return frame.sort_values("date").reset_index(drop=True)


def parse_eodhd_dividends(payload: object) -> pd.DataFrame:
    if payload in (None, []):
        return _EMPTY_EVENT_FRAME.copy()
    if not isinstance(payload, list):
        raise ExternalSourcePayloadError("EODHD dividends payload must be a JSON list.")

    frame = pd.DataFrame(payload)
    if frame.empty:
        return _EMPTY_EVENT_FRAME.copy()
    if "date" not in frame.columns:
        raise ExternalSourcePayloadError("EODHD dividends payload is missing 'date'.")
    amount_column = "value" if "value" in frame.columns else "unadjustedValue" if "unadjustedValue" in frame.columns else None
    if amount_column is None:
        raise ExternalSourcePayloadError("EODHD dividends payload is missing 'value'.")

    output = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["date"], utc=True, errors="coerce"),
            "dividends": pd.to_numeric(frame[amount_column], errors="coerce").fillna(0.0),
            "stock_splits": 0.0,
        }
    )
    return output.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _parse_split_ratio(value: object) -> float:
    if value is None:
        raise ExternalSourcePayloadError("EODHD split ratio is missing.")
    raw = str(value).strip()
    if not raw:
        raise ExternalSourcePayloadError("EODHD split ratio is empty.")
    if "/" in raw:
        numerator_raw, denominator_raw = raw.split("/", 1)
        numerator = float(numerator_raw)
        denominator = float(denominator_raw)
        if abs(denominator) <= 1e-12:
            raise ExternalSourcePayloadError("EODHD split ratio denominator cannot be zero.")
        return numerator / denominator
    return float(raw)


def parse_eodhd_splits(payload: object) -> pd.DataFrame:
    if payload in (None, []):
        return _EMPTY_EVENT_FRAME.copy()
    if not isinstance(payload, list):
        raise ExternalSourcePayloadError("EODHD splits payload must be a JSON list.")

    frame = pd.DataFrame(payload)
    if frame.empty:
        return _EMPTY_EVENT_FRAME.copy()
    if "date" not in frame.columns or "split" not in frame.columns:
        raise ExternalSourcePayloadError("EODHD splits payload is missing 'date' or 'split'.")

    output = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["date"], utc=True, errors="coerce"),
            "dividends": 0.0,
            "stock_splits": frame["split"].map(_parse_split_ratio),
        }
    )
    return output.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


class EODHDClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_EODHD_BASE_URL,
        timeout_seconds: float = DEFAULT_EODHD_TIMEOUT_SECONDS,
        use_cache: bool = True,
        cache_dir: Path | None = None,
        cache_ttl_seconds: int = DEFAULT_EODHD_CACHE_TTL_SECONDS,
        max_retries: int = DEFAULT_EODHD_MAX_RETRIES,
        backoff_seconds: float = DEFAULT_EODHD_BACKOFF_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        normalized_key = str(api_key or "").strip()
        if not normalized_key:
            raise ValueError("EODHD api_key is required.")
        self.api_key = normalized_key
        self.base_url = str(base_url or DEFAULT_EODHD_BASE_URL).strip().rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.use_cache = bool(use_cache)
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser().resolve()
        self.cache_ttl_seconds = int(cache_ttl_seconds)
        self.max_retries = int(max_retries)
        self.backoff_seconds = float(backoff_seconds)
        self.session = session or requests.Session()
        self.metrics = {
            "request_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        if self.timeout_seconds <= 0:
            raise ValueError("EODHD timeout_seconds must be > 0.")
        if self.cache_ttl_seconds < 0:
            raise ValueError("EODHD cache_ttl_seconds must be >= 0.")
        if self.max_retries < 1:
            raise ValueError("EODHD max_retries must be >= 1.")
        if self.backoff_seconds < 0:
            raise ValueError("EODHD backoff_seconds must be >= 0.")
        if self.use_cache and self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, endpoint: str, params: dict[str, object]) -> Path | None:
        if not self.use_cache or self.cache_dir is None:
            return None
        namespace = self.cache_dir / endpoint
        namespace.mkdir(parents=True, exist_ok=True)
        return namespace / f"{_cache_key(endpoint, params)}.json"

    def _request_json(self, endpoint: str, params: dict[str, object]) -> EODHDPayload:
        full_params = dict(params)
        full_params["api_token"] = self.api_key
        full_params["fmt"] = "json"
        url = _build_eodhd_url(self.base_url, endpoint, full_params)
        redacted_params = dict(full_params)
        if "api_token" in redacted_params:
            redacted_params["api_token"] = "***"
        redacted_url = _build_eodhd_url(self.base_url, endpoint, redacted_params)
        cache_path = self._cache_path(endpoint.strip("/").replace("/", "_"), full_params)
        if cache_path is not None:
            cached = _read_json_cache(cache_path, self.cache_ttl_seconds)
            if cached is not None:
                self.metrics["cache_hits"] += 1
                return EODHDPayload(payload=cached, url=redacted_url, cache_status="hit", endpoint=endpoint)
            self.metrics["cache_misses"] += 1

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self.metrics["request_count"] += 1
                response = self.session.get(url, timeout=self.timeout_seconds)
            except requests.Timeout as exc:
                last_error = ExternalSourceNetworkError(
                    f"EODHD request timed out after {self.timeout_seconds:.2f}s for {endpoint}."
                )
            except requests.RequestException as exc:
                safe_message = sanitize_secret_text(str(exc)) or "unknown transport error"
                last_error = ExternalSourceNetworkError(f"EODHD request failed for {endpoint}: {safe_message}")
            else:
                if response.status_code >= 400:
                    try:
                        _raise_http_error(response)
                    except (ExternalSourceRateLimitError, ExternalSourceNetworkError) as exc:
                        last_error = exc
                    else:
                        last_error = None
                    if last_error is None:
                        raise AssertionError("HTTP error classifier did not raise an error as expected.")
                else:
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        raise ExternalSourcePayloadError(
                            f"EODHD returned invalid JSON for {endpoint}."
                        ) from exc
                    if isinstance(payload, dict) and any(key in payload for key in ("error", "errors")):
                        message = _extract_error_message(response)
                        raise ExternalSourcePayloadError(f"EODHD returned an error payload: {message}")
                    if cache_path is not None:
                        _write_json_cache(cache_path, payload)
                    return EODHDPayload(
                        payload=payload,
                        url=redacted_url,
                        cache_status="disabled" if cache_path is None else "miss",
                        endpoint=endpoint,
                    )

            if isinstance(last_error, (ExternalSourceNotFoundError, ExternalSourceCoverageError, ExternalSourcePayloadError)):
                raise last_error
            if attempt < self.max_retries:
                time.sleep(self.backoff_seconds * attempt)

        if last_error is None:
            raise ExternalSourceNetworkError(f"EODHD request failed without a classified error for {endpoint}.")
        raise last_error

    def fetch_prices(self, symbol: str, start: str | None, end: str | None) -> EODHDPayload:
        provider_symbol = _normalize_symbol_candidate(symbol)
        return self._request_json(
            f"/api/eod/{provider_symbol}",
            {
                "from": start,
                "to": end,
                "period": "d",
            },
        )

    def fetch_dividends(self, symbol: str, start: str | None, end: str | None) -> EODHDPayload:
        provider_symbol = _normalize_symbol_candidate(symbol)
        return self._request_json(
            f"/api/div/{provider_symbol}",
            {
                "from": start,
                "to": end,
            },
        )

    def fetch_splits(self, symbol: str, start: str | None, end: str | None) -> EODHDPayload:
        provider_symbol = _normalize_symbol_candidate(symbol)
        return self._request_json(
            f"/api/splits/{provider_symbol}",
            {
                "from": start,
                "to": end,
            },
        )


class EODHDPriceReferenceSource:
    def __init__(
        self,
        client: EODHDClient,
        *,
        symbol_resolver: EODHDSymbolResolver | None = None,
        price_lookback_days: int = 365,
    ) -> None:
        self.client = client
        self.symbol_resolver = symbol_resolver or EODHDSymbolResolver()
        self.price_lookback_days = max(1, int(price_lookback_days))

    def name(self) -> str:
        return "eodhd_prices"

    def validation_scope(self) -> Literal["price"]:
        return "price"

    def _resolve_effective_window(self, start: str | None, end: str | None) -> tuple[str | None, str | None, bool]:
        end_ts = pd.to_datetime(end, utc=True, errors="coerce")
        if pd.isna(end_ts):
            end_ts = pd.Timestamp.now(tz="UTC")

        requested_start_ts = pd.to_datetime(start, utc=True, errors="coerce")
        limited_start_ts = end_ts - pd.Timedelta(days=self.price_lookback_days)

        if pd.isna(requested_start_ts):
            effective_start_ts = limited_start_ts
            partial_coverage = True
        else:
            effective_start_ts = max(requested_start_ts, limited_start_ts)
            partial_coverage = bool(requested_start_ts < effective_start_ts)

        effective_start = pd.Timestamp(effective_start_ts).tz_convert("UTC").strftime("%Y-%m-%d")
        effective_end = pd.Timestamp(end_ts).tz_convert("UTC").strftime("%Y-%m-%d")
        return effective_start, effective_end, partial_coverage

    def fetch_reference(self, symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
        resolution = self.symbol_resolver.resolve_candidates(symbol)
        errors: list[str] = []
        response: EODHDPayload | None = None
        frame = _EMPTY_PRICE_FRAME.copy()
        provider_symbol = resolution.requested_symbol
        effective_start, effective_end, partial_coverage = self._resolve_effective_window(start, end)
        for candidate in resolution.candidates:
            try:
                response = self.client.fetch_prices(candidate, effective_start, effective_end)
            except ExternalSourceNotFoundError as exc:
                errors.append(f"{candidate}: {exc}")
                continue
            frame = filter_reference_frame(
                parse_eodhd_prices(response.payload),
                start=effective_start,
                end=effective_end,
            )
            provider_symbol = candidate
            break

        if response is None:
            raise ExternalSourceNotFoundError(
                "EODHD could not resolve the requested symbol. Candidates tried: "
                + ", ".join(resolution.candidates)
                + ("" if not errors else ". Details: " + " | ".join(errors))
            )

        return attach_source_metadata(
            frame,
            {
                "provider": "eodhd",
                "source": "eodhd_prices",
                "scope": "price",
                "requested_symbol": symbol,
                "provider_symbol": provider_symbol,
                "symbol_mapping": resolution.strategy,
                "candidates_tried": list(resolution.candidates),
                "endpoint": response.endpoint,
                "url": response.url,
                "cache_status": response.cache_status,
                "requested_start": start,
                "requested_end": end,
                "effective_start": effective_start,
                "effective_end": effective_end,
                "price_lookback_days": int(self.price_lookback_days),
                "partial_coverage": partial_coverage,
                "client_metrics": dict(self.client.metrics),
            },
        )


class EODHDCorporateActionsReferenceSource:
    def __init__(
        self,
        client: EODHDClient,
        *,
        allow_partial_coverage: bool = False,
        symbol_resolver: EODHDSymbolResolver | None = None,
    ) -> None:
        self.client = client
        self.allow_partial_coverage = bool(allow_partial_coverage)
        self.symbol_resolver = symbol_resolver or EODHDSymbolResolver()

    def name(self) -> str:
        return "eodhd_corporate_actions"

    def validation_scope(self) -> Literal["event"]:
        return "event"

    def fetch_events(self, symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
        resolution = self.symbol_resolver.resolve_candidates(symbol)
        provider_symbol = resolution.requested_symbol
        last_not_found: ExternalSourceNotFoundError | None = None
        for candidate in resolution.candidates:
            try:
                return self._fetch_events_for_symbol(
                    symbol=symbol,
                    provider_symbol=candidate,
                    mapping_strategy=resolution.strategy,
                    candidates_tried=resolution.candidates,
                    start=start,
                    end=end,
                )
            except ExternalSourceNotFoundError as exc:
                last_not_found = exc
                continue

        raise ExternalSourceNotFoundError(
            "EODHD could not resolve the requested symbol for corporate actions. Candidates tried: "
            + ", ".join(resolution.candidates)
            + ("" if last_not_found is None else f". Last error: {last_not_found}")
        )

    def _fetch_events_for_symbol(
        self,
        *,
        symbol: str,
        provider_symbol: str,
        mapping_strategy: str,
        candidates_tried: tuple[str, ...],
        start: str | None,
        end: str | None,
    ) -> pd.DataFrame:
        dividends: pd.DataFrame | None = None
        splits: pd.DataFrame | None = None
        cache_statuses: list[str] = []
        urls: list[str] = []
        coverage_errors: list[str] = []

        try:
            dividends_response = self.client.fetch_dividends(provider_symbol, start, end)
            dividends = parse_eodhd_dividends(dividends_response.payload)
            cache_statuses.append(dividends_response.cache_status)
            urls.append(dividends_response.url)
        except ExternalSourceCoverageError as exc:
            coverage_errors.append(str(exc))
        try:
            splits_response = self.client.fetch_splits(provider_symbol, start, end)
            splits = parse_eodhd_splits(splits_response.payload)
            cache_statuses.append(splits_response.cache_status)
            urls.append(splits_response.url)
        except ExternalSourceCoverageError as exc:
            coverage_errors.append(str(exc))

        if coverage_errors and not self.allow_partial_coverage and (dividends is None or splits is None):
            raise ExternalSourceCoverageError(" | ".join(coverage_errors))

        if dividends is None:
            dividends = _EMPTY_EVENT_FRAME.copy()
        if splits is None:
            splits = _EMPTY_EVENT_FRAME.copy()

        frames_to_concat = [frame for frame in (dividends, splits) if not frame.empty]
        combined = (
            pd.concat(frames_to_concat, ignore_index=True)
            if frames_to_concat
            else _EMPTY_EVENT_FRAME.copy()
        )
        if combined.empty:
            return attach_source_metadata(
                filter_event_frame(combined, start=start, end=end),
                {
                    "provider": "eodhd",
                    "source": "eodhd_corporate_actions",
                    "scope": "event",
                    "requested_symbol": symbol,
                    "provider_symbol": provider_symbol,
                    "symbol_mapping": mapping_strategy,
                    "candidates_tried": list(candidates_tried),
                    "cache_statuses": list(cache_statuses),
                    "urls": urls,
                    "client_metrics": dict(self.client.metrics),
                    "partial_coverage": bool(coverage_errors),
                    "coverage_notes": coverage_errors,
                },
            )

        aggregated = (
            combined.groupby("date", as_index=False)[["dividends", "stock_splits"]]
            .sum()
            .sort_values("date")
            .reset_index(drop=True)
        )
        return attach_source_metadata(
            filter_event_frame(aggregated, start=start, end=end),
            {
                "provider": "eodhd",
                "source": "eodhd_corporate_actions",
                "scope": "event",
                "requested_symbol": symbol,
                "provider_symbol": provider_symbol,
                "symbol_mapping": mapping_strategy,
                "candidates_tried": list(candidates_tried),
                "cache_statuses": list(cache_statuses),
                "urls": urls,
                "client_metrics": dict(self.client.metrics),
                "partial_coverage": bool(coverage_errors),
                "coverage_notes": coverage_errors,
            },
        )

    def fetch_reference(self, symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
        return self.fetch_events(symbol, start, end)
