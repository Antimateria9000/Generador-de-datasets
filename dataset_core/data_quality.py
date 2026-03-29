from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import exchange_calendars as xc
except Exception:  # pragma: no cover
    xc = None

try:
    import pandas_market_calendars as pmc
except Exception:  # pragma: no cover
    pmc = None

SUITE_NAME = "AB3.DataQualitySuite"
LOGGER_NAME = SUITE_NAME

logger = logging.getLogger(LOGGER_NAME)
logger.addHandler(logging.NullHandler())


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class DQFinding:
    check: str
    severity: Severity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["severity"] = self.severity.value
        return payload


@dataclass
class DQCapabilities:
    calendar_provider: str
    calendar_exact: bool
    exchange_calendars_available: bool
    pandas_market_calendars_available: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DQContext:
    asset_type: str = "equity"
    quote_type: str = "EQUITY"
    market: Optional[str] = "XNYS"
    dq_profile: str = "equity"
    is_24_7: bool = False
    volume_expected: bool = True
    corporate_actions_expected: bool = True
    calendar_validation_supported: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AB3DataQualitySuite:
    SUITE_VERSION = "3.1.0"

    def __init__(
        self,
        jump_threshold: float = 0.40,
        split_match_tol: float = 0.03,
        volume_zero_ratio_err: float = 0.05,
        volume_window: int = 60,
        volume_zscore_thr: float = 5.0,
        price_move_bps_thr: float = 50.0,
        relative_low_volume_ratio: float = 0.15,
        static_run_days_err: int = 3,
        variance_eps: float = 1e-8,
        unique_ratio_thr: float = 0.03,
        long_window: int = 60,
        business_days_tol: int = 5,
        adj_ratio_std_tol: float = 0.02,
        div_explain_tol: float = 0.05,
        save_sidecar: bool = False,
        sidecar_outdir: str = ".",
        discard_on_error: bool = False,
        market: Optional[str] = "XNYS",
        missing_split_col_severity: Severity = Severity.WARNING,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.jump_threshold = float(jump_threshold)
        self.split_match_tol = float(split_match_tol)
        self.volume_zero_ratio_err = float(volume_zero_ratio_err)
        self.volume_window = int(volume_window)
        self.volume_zscore_thr = float(volume_zscore_thr)
        self.price_move_bps_thr = float(price_move_bps_thr)
        self.relative_low_volume_ratio = float(relative_low_volume_ratio)
        self.static_run_days_err = int(static_run_days_err)
        self.variance_eps = float(variance_eps)
        self.unique_ratio_thr = float(unique_ratio_thr)
        self.long_window = int(long_window)
        self.business_days_tol = int(business_days_tol)
        self.adj_ratio_std_tol = float(adj_ratio_std_tol)
        self.div_explain_tol = float(div_explain_tol)
        self.save_sidecar = bool(save_sidecar)
        self.sidecar_outdir = str(sidecar_outdir)
        self.discard_on_error = bool(discard_on_error)
        self.market = str(market) if market else None
        self.missing_split_col_severity = missing_split_col_severity

        self.findings: List[DQFinding] = []
        self._working_df: Optional[pd.DataFrame] = None
        self._calendar_provider_name = self._resolve_calendar_provider_name()
        self._calendar_exact = self._calendar_provider_name in {"exchange_calendars", "pandas_market_calendars"}
        self.context = self._resolve_context(context)

    def run(self, df: pd.DataFrame, symbol: str = "") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        self.findings = []
        self._working_df = None

        if df is None or df.empty:
            self._push("basic", Severity.ERROR, "DataFrame vacío")
            report = self._report(symbol=symbol, df=pd.DataFrame(), sidecar_path=None)
            empty = pd.DataFrame() if self.discard_on_error else df
            if hasattr(empty, "attrs"):
                empty.attrs["dq_report"] = report
            return empty, report

        try:
            work = self._prepare_frame(df)
        except Exception as exc:
            self._push("prepare_frame", Severity.ERROR, f"No se pudo preparar el DataFrame: {exc}")
            report = self._report(symbol=symbol, df=df, sidecar_path=None)
            if hasattr(df, "attrs"):
                df.attrs["dq_report"] = report
            return (pd.DataFrame() if self.discard_on_error else df), report

        self._working_df = work
        self._push_context_info()

        self._check_index(work)
        self._check_required_and_nans(work)
        self._check_numeric_dtypes(work)
        self._check_ohlc_geometry(work)

        self._check_splits(work)
        self._check_adjclose_consistency(work)
        self._check_ex_div(work)

        self._check_volume_zeros(work)
        self._check_volume_outliers(work)
        self._check_price_move_with_abnormal_low_vol(work)

        self._check_static_prices(work)
        self._check_low_variance(work)
        self._check_few_uniques(work)

        self._check_calendar(work, mkt=self.context.market)

        sidecar_path: Optional[str] = None
        report = self._report(symbol=symbol, df=work, sidecar_path=None)

        if self.save_sidecar:
            try:
                sidecar_path = self._save_sidecar(report, symbol=symbol, df=work)
            except Exception as exc:
                self._push(
                    "sidecar_write",
                    Severity.WARNING,
                    f"No se pudo escribir el sidecar de data quality: {exc}",
                )
                report = self._report(symbol=symbol, df=work, sidecar_path=None)
            else:
                report = self._report(symbol=symbol, df=work, sidecar_path=sidecar_path)

        work.attrs["dq_report"] = report
        if sidecar_path:
            work.attrs["dq_sidecar_path"] = sidecar_path

        if self.discard_on_error and report["n_errors"] > 0:
            empty = pd.DataFrame()
            empty.attrs["dq_report"] = report
            if sidecar_path:
                empty.attrs["dq_sidecar_path"] = sidecar_path
            return empty, report

        return work, report

    def has_errors(self) -> bool:
        return any(f.severity == Severity.ERROR for f in self.findings)

    def print_report(self, report: Dict[str, Any]) -> None:
        print("\n" + "=" * 72)
        print(f"Data Quality Report - {report.get('symbol', '')}")
        print(f"Status: {report['status']}")
        print(
            f"Errors: {report['n_errors']} | Warnings: {report['n_warnings']} | Infos: {report['n_infos']}"
        )
        print("=" * 72 + "\n")

        for section_name in ("errors", "warnings", "infos"):
            entries = report.get(section_name, [])
            if not entries:
                continue
            print(section_name.upper() + ":")
            for item in entries:
                print(f"  - [{item['check']}] {item['message']}")
                if item.get("details"):
                    print(f"    details={item['details']}")
            print()

        print("=" * 72 + "\n")

    def get_capabilities(self) -> Dict[str, Any]:
        return DQCapabilities(
            calendar_provider=self._calendar_provider_name,
            calendar_exact=self._calendar_exact,
            exchange_calendars_available=xc is not None,
            pandas_market_calendars_available=pmc is not None,
        ).to_dict()

    def _resolve_context(self, payload: Optional[Dict[str, Any]]) -> DQContext:
        data = dict(payload or {})
        market = data.get("market", self.market)
        return DQContext(
            asset_type=str(data.get("asset_type", "equity")).lower(),
            quote_type=str(data.get("quote_type", "EQUITY")).upper(),
            market=str(market) if market else None,
            dq_profile=str(data.get("dq_profile", "equity")).lower(),
            is_24_7=bool(data.get("is_24_7", False)),
            volume_expected=bool(data.get("volume_expected", True)),
            corporate_actions_expected=bool(data.get("corporate_actions_expected", True)),
            calendar_validation_supported=bool(data.get("calendar_validation_supported", True)),
        )

    def _push_context_info(self) -> None:
        self._push(
            "dq_context",
            Severity.INFO,
            "Contexto de validación aplicado.",
            self.context.to_dict(),
        )

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()

        if not isinstance(work.index, pd.DatetimeIndex):
            work.index = pd.to_datetime(work.index, errors="coerce")

        bad_index = int(pd.isna(work.index).sum())
        if bad_index > 0:
            self._push(
                "index_parse",
                Severity.ERROR,
                f"Se detectaron {bad_index} valores inválidos en el índice temporal.",
            )
            work = work[~pd.isna(work.index)].copy()

        if getattr(work.index, "tz", None) is not None:
            work.index = work.index.tz_convert("UTC").tz_localize(None)
            self._push(
                "index_timezone_normalized",
                Severity.INFO,
                "Índice timezone-aware normalizado a UTC naive para validación determinista.",
            )

        work = work.sort_index()
        return work

    def _check_index(self, df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            self._push("index_type", Severity.ERROR, "Index debe ser DatetimeIndex")
            return

        if not df.index.is_monotonic_increasing:
            self._push("index_monotonic", Severity.ERROR, "Índice no monótono creciente")

        dup = df.index.duplicated(keep=False)
        if dup.any():
            sample = df.index[dup][:10].astype(str).tolist()
            self._push(
                "index_duplicates",
                Severity.ERROR,
                f"Índice con duplicados: {int(dup.sum())} filas",
                {"sample": sample},
            )

    def _check_required_and_nans(self, df: pd.DataFrame) -> None:
        required = ["Open", "High", "Low", "Close"]
        missing = [column for column in required if column not in df.columns]
        if self.context.volume_expected:
            required.append("Volume")
        missing = [column for column in required if column not in df.columns]
        if missing:
            self._push("required_columns", Severity.ERROR, f"Faltan columnas requeridas: {missing}")
            return

        nan_counts = df[["Open", "High", "Low", "Close"]].isna().sum()
        bad = {key: int(value) for key, value in nan_counts.items() if int(value) > 0}
        if bad:
            self._push("nan_ohlc", Severity.ERROR, "NaNs en OHLC", {"counts": bad})

        if "Volume" not in df.columns:
            if not self.context.volume_expected:
                self._push(
                    "volume_not_expected",
                    Severity.INFO,
                    "El contexto del activo no exige columna Volume.",
                )
            return

        volume_numeric = pd.to_numeric(df["Volume"], errors="coerce")
        neg_vol = int((volume_numeric < 0).sum())
        if neg_vol:
            self._push("neg_volume", Severity.ERROR, f"{neg_vol} filas con Volume < 0")

        nan_vol_n = int(volume_numeric.isna().sum())
        if nan_vol_n:
            ratio = nan_vol_n / max(len(df), 1)
            sev = Severity.ERROR if self.context.volume_expected and ratio > self.volume_zero_ratio_err else Severity.WARNING
            self._push(
                "nan_volume_ratio",
                sev,
                f"{ratio * 100:.1f}% de filas con Volume=NaN",
                {"ratio": ratio, "n": nan_vol_n},
            )

    def _check_numeric_dtypes(self, df: pd.DataFrame) -> None:
        cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]
        bad: Dict[str, Dict[str, Any]] = {}

        for column in cols:
            if column not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(df[column]):
                continue

            converted = pd.to_numeric(df[column], errors="coerce")
            baseline_nans = int(df[column].isna().sum())
            converted_nans = int(converted.isna().sum())
            bad[column] = {
                "dtype": str(df[column].dtype),
                "coercion_new_nans": max(0, converted_nans - baseline_nans),
            }

        if bad:
            self._push(
                "non_numeric_dtype",
                Severity.ERROR,
                "Columnas financieras no numéricas o con coerción problemática.",
                bad,
            )

    def _check_ohlc_geometry(self, df: pd.DataFrame) -> None:
        needed = {"Open", "High", "Low", "Close"}
        if not needed.issubset(df.columns):
            return

        o = pd.to_numeric(df["Open"], errors="coerce")
        h = pd.to_numeric(df["High"], errors="coerce")
        l = pd.to_numeric(df["Low"], errors="coerce")
        c = pd.to_numeric(df["Close"], errors="coerce")

        neg_price = (pd.concat([o, h, l, c], axis=1) <= 0).any(axis=1)
        bad_hilo = h < l
        bad_open = (o < l) | (o > h)
        bad_close = (c < l) | (c > h)
        mask = neg_price | bad_hilo | bad_open | bad_close

        if mask.any():
            self._push(
                "ohlc_incoherence",
                Severity.ERROR,
                f"Violaciones OHLC intrínsecas: {int(mask.sum())}",
                {"sample": df.index[mask][:10].astype(str).tolist()},
            )

    def _parse_split(self, value: Any) -> float:
        if pd.isna(value) or value == 0:
            return 0.0
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, str):
            s = value.strip().lower().replace("for", "/").replace("×", "/").replace("x", "/")
            s = s.replace(" ", "")
            for sep in (":", "/"):
                if sep in s:
                    try:
                        left, right = map(float, s.split(sep))
                        return left / right if right else 0.0
                    except Exception:
                        return 0.0
            try:
                return float(s)
            except Exception:
                return 0.0
        return 0.0

    def _check_splits(self, df: pd.DataFrame) -> None:
        if not self.context.corporate_actions_expected:
            self._push(
                "splits_skipped",
                Severity.INFO,
                "Chequeo de splits omitido por el contexto del activo.",
            )
            return

        if "Close" not in df.columns or len(df) < 2:
            return

        ret = pd.to_numeric(df["Close"], errors="coerce").pct_change()
        big = ret.abs() > self.jump_threshold

        split_col = None
        for candidate in ("Stock Splits", "Splits", "stock_splits"):
            if candidate in df.columns:
                split_col = candidate
                break

        for date_value in df.index[big]:
            if "Dividends" in df.columns:
                try:
                    div_v = float(pd.to_numeric(df.at[date_value, "Dividends"], errors="coerce"))
                except Exception:
                    div_v = 0.0
                if div_v and abs(float(ret.loc[date_value])) < 0.60:
                    continue

            if split_col is None:
                self._push(
                    "missing_split_col",
                    self.missing_split_col_severity,
                    f"Salto {ret.loc[date_value]:.2%} sin columna de splits",
                    {"date": str(date_value)},
                )
                self._push(
                    "missing_split",
                    Severity.ERROR,
                    "Posible split no etiquetado (no hay columna de splits).",
                    {"date": str(date_value), "jump": float(ret.loc[date_value])},
                )
                continue

            candidates = [date_value]
            try:
                idx = df.index.get_loc(date_value)
                if isinstance(idx, (np.ndarray, list)):
                    idx = int(np.atleast_1d(idx)[0])
            except Exception:
                idx = None

            if idx is not None:
                if idx > 0:
                    candidates.append(df.index[idx - 1])
                if idx < len(df.index) - 1:
                    candidates.append(df.index[idx + 1])

            factors = [self._parse_split(df.loc[candidate, split_col]) for candidate in candidates if candidate in df.index]
            factors = [factor for factor in factors if factor and factor > 0]

            if not factors:
                self._push(
                    "missing_split",
                    Severity.ERROR,
                    "Posible split no etiquetado.",
                    {"date": str(date_value), "jump": float(ret.loc[date_value])},
                )
                continue

            jump = float(ret.loc[date_value])
            ok = False
            for factor in factors:
                if abs(jump - (1.0 / factor - 1.0)) <= self.split_match_tol:
                    ok = True
                    break
                if abs(jump - (factor - 1.0)) <= self.split_match_tol:
                    ok = True
                    break

            if not ok:
                self._push(
                    "split_coherence",
                    Severity.WARNING,
                    "Split no explica completamente el salto.",
                    {
                        "date": str(date_value),
                        "jump": jump,
                        "factors": factors,
                        "tol": self.split_match_tol,
                    },
                )

    def _event_mask(self, df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
        event_mask = pd.Series(False, index=idx)

        if self.context.corporate_actions_expected and "Dividends" in df.columns:
            event_mask |= pd.to_numeric(df["Dividends"], errors="coerce").reindex(idx).fillna(0) > 0

        if self.context.corporate_actions_expected:
            for candidate in ("Stock Splits", "Splits", "stock_splits"):
                if candidate in df.columns:
                    event_mask |= pd.Series(df[candidate].reindex(idx).map(self._parse_split)).fillna(0) > 0

        for shift in (-2, -1, 1, 2):
            event_mask |= event_mask.shift(shift, fill_value=False)

        return event_mask

    def _check_adjclose_consistency(self, df: pd.DataFrame) -> None:
        if not self.context.corporate_actions_expected:
            self._push(
                "adjclose_skipped",
                Severity.INFO,
                "Chequeo de Adj Close omitido por el contexto del activo.",
            )
            return

        if "Close" not in df.columns or "Adj Close" not in df.columns:
            return

        close = pd.to_numeric(df["Close"], errors="coerce")
        adj_close = pd.to_numeric(df["Adj Close"], errors="coerce")
        ratio = (close / adj_close).replace([np.inf, -np.inf], np.nan).dropna()
        if ratio.empty:
            return

        core = ratio[~self._event_mask(df, ratio.index)]
        if len(core) < 10:
            return

        ratio_std = float(core.std())
        jumps = core.pct_change().abs() > 0.05

        if ratio_std > self.adj_ratio_std_tol:
            self._push(
                "adj_ratio_instability",
                Severity.WARNING,
                f"Close/AdjClose ratio inestable (std={ratio_std:.4f})",
                {"std": ratio_std},
            )

        if jumps.any():
            self._push(
                "adj_ratio_jumps",
                Severity.WARNING,
                f"{int(jumps.sum())} saltos grandes del ratio fuera de eventos.",
            )

    def _check_ex_div(self, df: pd.DataFrame) -> None:
        if not self.context.corporate_actions_expected:
            self._push(
                "exdiv_skipped",
                Severity.INFO,
                "Chequeo ex-div omitido por el contexto del activo.",
            )
            return

        if "Dividends" not in df.columns or "Close" not in df.columns:
            return

        dividends = pd.to_numeric(df["Dividends"], errors="coerce").fillna(0.0)
        if not (dividends > 0).any():
            return

        close = pd.to_numeric(df["Close"], errors="coerce")
        prev_close = close.shift(1)
        expected_drop = dividends / prev_close
        actual_drop = (prev_close - close) / prev_close

        bad_close = (dividends > 0) & ((actual_drop - expected_drop).abs() > self.div_explain_tol)
        if bad_close.any():
            dates = df.index[bad_close][:10].astype(str).tolist()
            self._push(
                "exdiv_missing_in_close",
                Severity.WARNING,
                "Señal de ex-div no aparece en Close.",
                {"sample": dates, "tol": self.div_explain_tol},
            )

        if "Adj Close" in df.columns:
            adj_close = pd.to_numeric(df["Adj Close"], errors="coerce")
            adj_prev = adj_close.shift(1)
            adj_drop = (adj_prev - adj_close) / adj_prev
            bad_adj = (dividends > 0) & (adj_drop.abs() > self.div_explain_tol)
            if bad_adj.any():
                dates = df.index[bad_adj][:10].astype(str).tolist()
                self._push(
                    "exdiv_present_in_adjclose",
                    Severity.WARNING,
                    "Ex-div aparece en Adj Close (debería ser ~0).",
                    {"sample": dates, "tol": self.div_explain_tol},
                )

    def _check_volume_zeros(self, df: pd.DataFrame) -> None:
        if not self.context.volume_expected:
            self._push(
                "volume_zeros_skipped",
                Severity.INFO,
                "Chequeo de volumen omitido por el contexto del activo.",
            )
            return

        if "Volume" not in df.columns or len(df) == 0:
            return

        vol = pd.to_numeric(df["Volume"], errors="coerce")
        zero = vol.isna() | (vol == 0)
        ratio = float(zero.mean())
        if ratio > self.volume_zero_ratio_err:
            self._push(
                "volume_zero_ratio",
                Severity.ERROR,
                f"{ratio * 100:.1f}% de filas con Volume=0/NaN.",
                {"ratio": ratio, "n": int(zero.sum())},
            )

    def _check_volume_outliers(self, df: pd.DataFrame) -> None:
        if not self.context.volume_expected:
            return
        if "Volume" not in df.columns or len(df) < self.volume_window:
            return

        volume = pd.to_numeric(df["Volume"], errors="coerce").astype("float64")
        z_rob = self._robust_z(np.log1p(volume), self.volume_window)
        high = z_rob.abs() > self.volume_zscore_thr
        if high.any():
            count = int(high.sum())
            severity = Severity.ERROR if count > max(1, int(0.01 * len(volume))) else Severity.WARNING
            self._push(
                "volume_zscore_robust",
                severity,
                f"{count} días con |z_rob(log1p(volume))| > {self.volume_zscore_thr}",
                {"count": count, "max_z": float(z_rob.abs().max(skipna=True))},
            )

        q01 = volume.rolling(self.volume_window, min_periods=max(20, self.volume_window // 2)).quantile(0.01)
        low = volume < q01
        if low.any():
            self._push(
                "volume_below_p1",
                Severity.WARNING,
                f"{int(low.sum())} días con volume < p1 (ventana {self.volume_window}).",
            )

    def _check_price_move_with_abnormal_low_vol(self, df: pd.DataFrame) -> None:
        if not self.context.volume_expected:
            return
        if not {"Close", "Volume"}.issubset(df.columns) or len(df) < 2:
            return

        close = pd.to_numeric(df["Close"], errors="coerce")
        volume = pd.to_numeric(df["Volume"], errors="coerce").astype("float64")
        bps = close.pct_change().abs() * 10000.0

        minp = max(20, self.volume_window // 2)
        ref_p01 = volume.rolling(self.volume_window, min_periods=minp).quantile(0.01)
        ref_median = volume.rolling(self.volume_window, min_periods=minp).median()

        zero_or_nan = volume.isna() | (volume <= 0)
        materially_low = (volume <= ref_p01) & (volume <= (ref_median * self.relative_low_volume_ratio))
        move_mask = bps > self.price_move_bps_thr

        severe_mask = move_mask & zero_or_nan
        warning_mask = move_mask & materially_low & ~zero_or_nan

        if severe_mask.any():
            self._push(
                "price_move_with_zero_vol",
                Severity.ERROR,
                f"Movimientos > {int(self.price_move_bps_thr)} bps con volumen cero o ausente.",
                {"sample": df.index[severe_mask][:10].astype(str).tolist()},
            )

        if warning_mask.any():
            self._push(
                "price_move_with_abnormally_low_vol",
                Severity.WARNING,
                f"Movimientos > {int(self.price_move_bps_thr)} bps con volumen anormalmente bajo relativo a su ventana.",
                {
                    "sample": df.index[warning_mask][:10].astype(str).tolist(),
                    "relative_low_volume_ratio": self.relative_low_volume_ratio,
                },
            )

    def _check_price_move_with_near_zero_vol(self, df: pd.DataFrame) -> None:
        self._check_price_move_with_abnormal_low_vol(df)

    def _check_static_prices(self, df: pd.DataFrame) -> None:
        required = {"Open", "High", "Low", "Close"}
        if not required.issubset(df.columns):
            return

        o = pd.to_numeric(df["Open"], errors="coerce")
        h = pd.to_numeric(df["High"], errors="coerce")
        l = pd.to_numeric(df["Low"], errors="coerce")
        c = pd.to_numeric(df["Close"], errors="coerce")

        static = (h == l) & (l == o) & (o == c)
        if not static.any():
            return

        groups = (static != static.shift()).cumsum()
        run_lengths = static.groupby(groups).transform("size")
        long_runs = run_lengths[static & (run_lengths >= self.static_run_days_err)]

        if long_runs.any():
            self._push(
                "static_runs",
                Severity.ERROR,
                f"Rachas OHLC estáticas >= {self.static_run_days_err} días.",
                {"max_run": int(long_runs.max())},
            )

    def _check_low_variance(self, df: pd.DataFrame) -> None:
        if len(df) < self.long_window:
            return

        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        price_series = pd.to_numeric(df[price_col], errors="coerce")
        var_price = price_series.rolling(self.long_window).var()
        low_price = var_price < self.variance_eps
        if low_price.any():
            self._push(
                "low_variance_price",
                Severity.WARNING,
                f"Varianza muy baja en {price_col} (ventana {self.long_window}).",
            )

        if self.context.volume_expected and "Volume" in df.columns:
            volume = pd.to_numeric(df["Volume"], errors="coerce").replace(0, np.nan)
            var_vol = volume.rolling(self.long_window).var()
            low_vol = var_vol < self.variance_eps
            if low_vol.any():
                self._push(
                    "low_variance_volume",
                    Severity.WARNING,
                    f"Varianza muy baja en Volume (ventana {self.long_window}).",
                )

    def _check_few_uniques(self, df: pd.DataFrame) -> None:
        if len(df) < self.long_window:
            return

        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        price_series = pd.to_numeric(df[price_col], errors="coerce")
        ratio = price_series.rolling(self.long_window).apply(lambda s: pd.Series(s).nunique() / max(len(s), 1), raw=False)
        few = ratio < self.unique_ratio_thr
        if few.any():
            self._push(
                "few_unique_price",
                Severity.WARNING,
                f"Pocos valores únicos en {price_col} (ratio < {self.unique_ratio_thr}).",
            )

        if self.context.volume_expected and "Volume" in df.columns:
            volume = pd.to_numeric(df["Volume"], errors="coerce")
            ratio_v = volume.rolling(self.long_window).apply(lambda s: pd.Series(s).nunique() / max(len(s), 1), raw=False)
            few_v = ratio_v < self.unique_ratio_thr
            if few_v.any():
                self._push(
                    "few_unique_volume",
                    Severity.WARNING,
                    f"Pocos valores únicos en Volume (ratio < {self.unique_ratio_thr}).",
                )

    def _check_calendar(self, df: pd.DataFrame, mkt: Optional[str] = "XNYS") -> None:
        if len(df) < 10:
            return

        if self.context.is_24_7:
            self._push(
                "calendar_skipped_24_7",
                Severity.INFO,
                "Chequeo de calendario omitido: activo 24/7.",
            )
            return

        if not self.context.calendar_validation_supported:
            self._push(
                "calendar_skipped_context",
                Severity.INFO,
                "Chequeo de calendario omitido por el contexto del activo.",
            )
            return

        if not mkt:
            self._push(
                "calendar_skipped_no_market",
                Severity.WARNING,
                "No se pudo validar el calendario porque no hay MIC/calendario resuelto.",
            )
            return

        try:
            idx = df.index
            if getattr(idx, "tz", None) is None:
                start_utc = pd.Timestamp(idx.min()).tz_localize("UTC")
                end_utc = pd.Timestamp(idx.max()).tz_localize("UTC")
                idx_naive_norm = idx.normalize()
            else:
                start_utc = idx.min().tz_convert("UTC")
                end_utc = idx.max().tz_convert("UTC")
                idx_naive_norm = idx.tz_convert("UTC").tz_localize(None).normalize()

            sessions_norm = None
            if xc is not None:
                cal = xc.get_calendar(mkt)
                sess = cal.sessions_in_range(start_utc, end_utc)
                sessions_norm = sess.tz_localize(None).normalize() if getattr(sess, "tz", None) else sess.normalize()
            elif pmc is not None:
                cal = pmc.get_calendar(mkt)
                sched = cal.schedule(start_date=start_utc.date(), end_date=end_utc.date())
                sess = pd.DatetimeIndex(sched.index)
                sessions_norm = sess.normalize()

            if sessions_norm is None:
                expected = pd.bdate_range(idx.min(), idx.max(), freq="B")
                missing = expected.difference(idx_naive_norm)
                if len(missing) > self.business_days_tol:
                    self._push(
                        "missing_business_days",
                        Severity.WARNING,
                        f"{len(missing)} business days faltantes (> {self.business_days_tol}).",
                        {"sample": [str(d) for d in missing[:10]]},
                    )

                weekend = idx_naive_norm[idx_naive_norm.dayofweek >= 5]
                if len(weekend) > self.business_days_tol:
                    self._push(
                        "weekend_with_data",
                        Severity.WARNING,
                        f"{len(weekend)} días de fin de semana con datos.",
                    )
                return

            missing = sessions_norm.difference(idx_naive_norm)
            extras = idx_naive_norm.difference(sessions_norm)

            if len(missing) > self.business_days_tol:
                self._push(
                    "missing_business_days",
                    Severity.WARNING,
                    f"{len(missing)} sesiones de mercado faltantes (> {self.business_days_tol}).",
                    {"sample": [str(d) for d in missing[:10]]},
                )

            if len(extras) > self.business_days_tol:
                self._push(
                    "outside_official_sessions",
                    Severity.WARNING,
                    f"{len(extras)} días con datos fuera de sesiones oficiales.",
                    {"sample": [str(d) for d in extras[:10]]},
                )
        except Exception as exc:
            self._push(
                "calendar_validation_error",
                Severity.WARNING,
                f"No se pudo validar el calendario oficial: {exc}",
                {"market": mkt},
            )

    def _resolve_calendar_provider_name(self) -> str:
        if xc is not None:
            return "exchange_calendars"
        if pmc is not None:
            return "pandas_market_calendars"
        return "business_day_fallback"

    def _robust_z(self, s: pd.Series, win: int) -> pd.Series:
        median = s.rolling(win, min_periods=max(20, win // 2)).median()
        mad = (s - median).abs().rolling(win, min_periods=max(20, win // 2)).median()
        denom = 1.4826 * mad.replace(0, np.nan)
        return (s - median) / denom

    def _push(self, check: str, severity: Severity, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.findings.append(DQFinding(check=check, severity=severity, message=message, details=details or {}))

    def _report(self, symbol: str, df: pd.DataFrame, sidecar_path: Optional[str]) -> Dict[str, Any]:
        errors = [f for f in self.findings if f.severity == Severity.ERROR]
        warnings = [f for f in self.findings if f.severity == Severity.WARNING]
        infos = [f for f in self.findings if f.severity == Severity.INFO]

        status = "FAIL" if errors else ("WARNING" if warnings else "PASS")
        date_start = str(df.index.min()) if isinstance(df.index, pd.DatetimeIndex) and len(df) else None
        date_end = str(df.index.max()) if isinstance(df.index, pd.DatetimeIndex) and len(df) else None

        return {
            "suite_name": SUITE_NAME,
            "suite_version": self.SUITE_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "symbol": symbol,
            "status": status,
            "row_count": int(len(df)) if df is not None else 0,
            "date_start": date_start,
            "date_end": date_end,
            "n_errors": len(errors),
            "n_warnings": len(warnings),
            "n_infos": len(infos),
            "context": self.context.to_dict(),
            "capabilities": self.get_capabilities(),
            "params": {
                "jump_threshold": self.jump_threshold,
                "split_match_tol": self.split_match_tol,
                "volume_zero_ratio_err": self.volume_zero_ratio_err,
                "volume_window": self.volume_window,
                "volume_zscore_thr": self.volume_zscore_thr,
                "price_move_bps_thr": self.price_move_bps_thr,
                "relative_low_volume_ratio": self.relative_low_volume_ratio,
                "static_run_days_err": self.static_run_days_err,
                "variance_eps": self.variance_eps,
                "unique_ratio_thr": self.unique_ratio_thr,
                "long_window": self.long_window,
                "business_days_tol": self.business_days_tol,
                "adj_ratio_std_tol": self.adj_ratio_std_tol,
                "div_explain_tol": self.div_explain_tol,
                "market": self.context.market,
                "missing_split_col_severity": self.missing_split_col_severity.value,
                "discard_on_error": self.discard_on_error,
                "save_sidecar": self.save_sidecar,
            },
            "sidecar_path": sidecar_path,
            "findings": [f.to_dict() for f in self.findings],
            "errors": [f.to_dict() for f in errors],
            "warnings": [f.to_dict() for f in warnings],
            "infos": [f.to_dict() for f in infos],
        }

    def _save_sidecar(self, report: Dict[str, Any], symbol: str, df: pd.DataFrame) -> str:
        safe_symbol = "".join(char for char in (symbol or "SYMBOL") if char.isalnum() or char in "-_")
        first = str(df.index.min().date()) if len(df) else "na"
        last = str(df.index.max().date()) if len(df) else "na"
        output_dir = Path(self.sidecar_outdir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{safe_symbol}_{first}_{last}.dq.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)



