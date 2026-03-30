from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

from dataset_core import (
    BatchOrchestrator,
    DatasetRequest,
    ExternalValidationConfig,
    ProviderConfig,
    RequestContractError,
    TemporalRange,
    parse_extras,
    resolve_ticker_inputs,
)
from dataset_core.settings import DEFAULT_OUTPUT_ROOT, DQ_MODES, LISTING_PREFERENCES, PRESET_NAMES, SUPPORTED_INTERVALS

LOGGER_NAME = "DatasetFactory.CLI"
logger = logging.getLogger(LOGGER_NAME)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reproducible OHLCV datasets for single tickers or real batches, "
            "including Qlib-ready output, internal DQ and pluggable external validation."
        )
    )
    ticker_group = parser.add_mutually_exclusive_group(required=True)
    ticker_group.add_argument("--ticker", type=str, help="Single ticker input.")
    ticker_group.add_argument("--tickers", type=str, help="Ticker list separated by comma, space or newline.")
    ticker_group.add_argument("--tickers-file", type=str, help="TXT/CSV file containing the ticker universe.")

    parser.add_argument("--years", default=None, type=int, help="Rolling lookback expressed in years.")
    parser.add_argument("--start", default=None, type=str, help="Exact range start in YYYY-MM-DD or ISO-8601.")
    parser.add_argument("--end", default=None, type=str, help="Exact range end in YYYY-MM-DD or ISO-8601.")
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTPUT_ROOT),
        type=str,
        help="Workspace root for runs, exports, manifests, reports and temp artifacts.",
    )
    parser.add_argument("--interval", default="1d", choices=SUPPORTED_INTERVALS, help="Requested download interval.")
    parser.add_argument("--mode", default="base", choices=PRESET_NAMES, help="Output preset.")
    parser.add_argument(
        "--extras",
        default="",
        type=str,
        help="Optional extra columns: adj_close, dividends, stock_splits, factor.",
    )
    parser.add_argument(
        "--filename",
        default=None,
        type=str,
        help="Custom CSV filename for single non-qlib runs. Not allowed for batch or qlib mode.",
    )
    parser.add_argument("--auto-adjust", action="store_true", help="Request adjusted prices from yfinance.")
    parser.add_argument("--no-actions", action="store_true", help="Disable dividends and split retrieval.")
    parser.add_argument(
        "--qlib-sanitization",
        action="store_true",
        help="Generate a Qlib-ready artifact in parallel with the general sanitized dataset.",
    )
    parser.add_argument("--dq-mode", default="report", choices=DQ_MODES, help="Internal DQ mode.")
    parser.add_argument("--dq-market", default="AUTO", type=str, help="Manual DQ market override.")
    parser.add_argument(
        "--listing-preference",
        default="exact_symbol",
        choices=LISTING_PREFERENCES,
        help="Listing resolution preference for unsuffixed tickers.",
    )
    parser.add_argument("--reference-dir", default=None, type=str, help="Directory with reference CSVs for external validation.")
    parser.add_argument("--manual-events-file", default=None, type=str, help="CSV/JSON file with manual split events.")
    parser.add_argument("--provider-max-workers", default=None, type=int, help="Override provider max_workers.")
    parser.add_argument("--provider-retries", default=None, type=int, help="Override provider retries.")
    parser.add_argument("--provider-timeout", default=None, type=float, help="Override provider timeout in seconds.")
    parser.add_argument("--provider-min-delay", default=None, type=float, help="Override provider minimum delay.")
    parser.add_argument(
        "--provider-max-intraday-lookback-days",
        default=None,
        type=int,
        help="Override provider maximum intraday lookback window in days.",
    )
    parser.add_argument(
        "--provider-allow-partial-intraday",
        action="store_true",
        help="Allow intraday truncation beyond the lookback window.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="CLI logging verbosity.",
    )
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_request_from_args(args: argparse.Namespace) -> DatasetRequest:
    tickers = resolve_ticker_inputs(
        ticker=args.ticker,
        tickers=args.tickers,
        tickers_file=args.tickers_file,
    )
    if args.mode == "qlib" and args.filename:
        raise RequestContractError("Custom filenames are not supported in qlib mode.")

    time_range = TemporalRange.from_inputs(
        years=args.years,
        start=args.start,
        end=args.end,
        interval=args.interval,
    )
    provider = ProviderConfig(
        max_workers=args.provider_max_workers,
        retries=args.provider_retries,
        timeout=args.provider_timeout,
        min_delay=args.provider_min_delay,
        max_intraday_lookback_days=args.provider_max_intraday_lookback_days,
        allow_partial_intraday=args.provider_allow_partial_intraday,
    )
    external_validation = ExternalValidationConfig(
        reference_dir=None if not args.reference_dir else Path(args.reference_dir).expanduser().resolve(),
        manual_events_file=None
        if not args.manual_events_file
        else Path(args.manual_events_file).expanduser().resolve(),
    )

    return DatasetRequest(
        tickers=tickers,
        time_range=time_range,
        output_dir=Path(args.outdir).expanduser().resolve(),
        interval=args.interval,
        mode=args.mode,
        extras=parse_extras(args.extras),
        listing_preference=args.listing_preference,
        dq_mode=args.dq_mode,
        dq_market=args.dq_market,
        auto_adjust=bool(args.auto_adjust),
        actions=not bool(args.no_actions),
        qlib_sanitization=bool(args.qlib_sanitization) or args.mode == "qlib",
        filename_override=args.filename,
        provider=provider,
        external_validation=external_validation,
    )


def summarize_batch(batch_result) -> None:
    counts = batch_result.status_counts
    logger.info("Output root: %s", batch_result.output_root)
    logger.info("CSV dir: %s", batch_result.csv_dir)
    logger.info("Meta dir: %s", batch_result.meta_dir)
    logger.info("Report dir: %s", batch_result.report_dir)
    logger.info(
        "Batch summary | success=%s warning=%s error=%s",
        counts.get("success", 0),
        counts.get("warning", 0),
        counts.get("error", 0),
    )
    for result in batch_result.results:
        logger.info(
            "Ticker %s | status=%s | qlib_compatible=%s | csv=%s",
            result.ticker,
            result.status,
            result.qlib_compatible,
            result.artifacts.csv,
        )


def run_cli(
    argv: Optional[Sequence[str]] = None,
    orchestrator: Optional[BatchOrchestrator] = None,
):
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    request = build_request_from_args(args)
    runner = orchestrator or BatchOrchestrator()
    batch_result = runner.run(request)
    summarize_batch(batch_result)
    return batch_result


def export_one_ticker(
    ticker: str,
    years: Optional[int],
    start: Optional[str],
    end: Optional[str],
    outdir: str,
    interval: str = "1d",
    filename: Optional[str] = None,
    auto_adjust: bool = False,
    actions: bool = True,
    dq_mode: str = "report",
    dq_market: str = "AUTO",
    listing_preference: str = "exact_symbol",
    mode: str = "base",
    extras: Optional[Sequence[str] | str] = None,
    qlib_sanitization: bool = False,
    reference_dir: Optional[str] = None,
    manual_events_file: Optional[str] = None,
    provider_max_workers: Optional[int] = None,
    provider_retries: Optional[int] = None,
    provider_timeout: Optional[float] = None,
    provider_min_delay: Optional[float] = None,
    provider_max_intraday_lookback_days: Optional[int] = None,
    provider_allow_partial_intraday: bool = False,
) -> Path:
    request = DatasetRequest(
        tickers=[ticker],
        time_range=TemporalRange.from_inputs(
            years=years,
            start=start,
            end=end,
            interval=interval,
        ),
        output_dir=Path(outdir).expanduser().resolve(),
        interval=interval,
        mode=mode,
        extras=parse_extras(extras),
        listing_preference=listing_preference,
        dq_mode=dq_mode,
        dq_market=dq_market,
        auto_adjust=auto_adjust,
        actions=actions,
        qlib_sanitization=qlib_sanitization or mode == "qlib",
        filename_override=filename,
        provider=ProviderConfig(
            max_workers=provider_max_workers,
            retries=provider_retries,
            timeout=provider_timeout,
            min_delay=provider_min_delay,
            max_intraday_lookback_days=provider_max_intraday_lookback_days,
            allow_partial_intraday=provider_allow_partial_intraday,
        ),
        external_validation=ExternalValidationConfig(
            reference_dir=None if not reference_dir else Path(reference_dir).expanduser().resolve(),
            manual_events_file=None
            if not manual_events_file
            else Path(manual_events_file).expanduser().resolve(),
        ),
    )
    result = BatchOrchestrator().run(request).results[0]
    if result.artifacts.csv is None:
        raise RuntimeError(f"Ticker export failed: {' | '.join(result.errors)}")
    return result.artifacts.csv


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        batch_result = run_cli(argv=argv)
    except RequestContractError as exc:
        logger.error("%s", exc)
        return 1

    counts = batch_result.status_counts
    successes = counts.get("success", 0) + counts.get("warning", 0)
    errors = counts.get("error", 0)
    if errors and successes:
        return 2
    if errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
