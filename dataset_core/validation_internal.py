from __future__ import annotations

from typing import Any

import pandas as pd

from dataset_core.data_quality import AB3DataQualitySuite


class InternalValidationUnsupportedError(RuntimeError):
    """Raised when strict internal validation cannot run safely."""


class InternalValidationExecutionError(RuntimeError):
    """Raised when strict internal validation crashes."""


class InternalValidationGateError(RuntimeError):
    """Raised when strict internal validation finds blocking errors."""


def _build_dq_input(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if "date" not in working.columns:
        raise ValueError("Internal validation requires a 'date' column.")

    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date"]).set_index("date").sort_index()
    working.index.name = "Date"

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj_close": "Adj Close",
        "volume": "Volume",
        "dividends": "Dividends",
        "stock_splits": "Stock Splits",
    }
    working = working.rename(columns=rename_map)

    defaults: dict[str, Any] = {
        "Adj Close": working["Close"] if "Close" in working.columns else pd.NA,
        "Dividends": 0.0,
        "Stock Splits": 0.0,
        "Volume": 0,
    }
    ordered_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]
    for column in ordered_columns:
        if column not in working.columns:
            working[column] = defaults[column]

    return working[ordered_columns].copy()


class InternalDQService:
    def assess_support(self, interval: str, actions: bool, dq_context: dict[str, Any]) -> dict[str, Any]:
        reasons: list[str] = []
        normalized_interval = str(interval).strip().lower()

        if normalized_interval != "1d":
            reasons.append("Internal DQ safe mode is guaranteed only for interval=1d.")
        if dq_context.get("corporate_actions_expected", True) and not actions:
            reasons.append(
                "This instrument profile expects corporate actions, but actions were disabled."
            )

        return {
            "supported": len(reasons) == 0,
            "interval": normalized_interval,
            "reasons": reasons,
        }

    def run(
        self,
        frame: pd.DataFrame,
        symbol: str,
        interval: str,
        actions: bool,
        dq_mode: str,
        dq_context: dict[str, Any],
    ) -> dict[str, Any]:
        support = self.assess_support(interval=interval, actions=actions, dq_context=dq_context)

        if dq_mode == "off":
            return {
                "status": "skipped",
                "support": support,
                "report": None,
                "reason": "dq_mode=off",
            }

        if not support["supported"]:
            detail = " | ".join(support["reasons"])
            if dq_mode == "strict":
                raise InternalValidationUnsupportedError(detail)
            return {
                "status": "unsupported",
                "support": support,
                "report": None,
                "reason": detail,
            }

        try:
            suite = AB3DataQualitySuite(
                save_sidecar=False,
                discard_on_error=False,
                market=dq_context.get("market"),
                context=dq_context,
            )
            _, report = suite.run(_build_dq_input(frame), symbol=symbol)
        except Exception as exc:
            if dq_mode == "strict":
                raise InternalValidationExecutionError(str(exc)) from exc
            return {
                "status": "execution_error",
                "support": support,
                "report": None,
                "reason": str(exc),
            }

        n_errors = int(report.get("n_errors", 0))
        n_warnings = int(report.get("n_warnings", 0))
        status = "passed" if n_errors == 0 and n_warnings == 0 else (
            "passed_with_warnings" if n_errors == 0 else "failed"
        )

        if dq_mode == "strict" and n_errors > 0:
            raise InternalValidationGateError(
                f"Internal DQ failed for {symbol} (errors={n_errors}, warnings={n_warnings})."
            )

        return {
            "status": status,
            "support": support,
            "report": report,
            "reason": None,
        }
