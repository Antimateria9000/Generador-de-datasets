from __future__ import annotations

import logging
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock

LOG_FILENAME = "dataset_factory.log"
RUNTIME_LOGGER_NAME = "DatasetFactory.Runtime"
_LOGGER_LOCK = Lock()


class _DefaultContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = "-"
        if not hasattr(record, "ticker"):
            record.ticker = "-"
        if not hasattr(record, "stage"):
            record.stage = "-"
        return True


class RuntimeLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        merged = dict(self.extra)
        merged.update(kwargs.get("extra", {}))
        kwargs["extra"] = merged
        return msg, kwargs

    def bind(self, **context: object) -> "RuntimeLoggerAdapter":
        merged = dict(self.extra)
        merged.update({key: value for key, value in context.items() if value is not None})
        return RuntimeLoggerAdapter(self.logger, merged)


@dataclass(frozen=True)
class RunLoggerHandle:
    logger: RuntimeLoggerAdapter
    log_path: Path


def _build_handler(log_path: Path) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        log_path,
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | run_id=%(run_id)s | ticker=%(ticker)s | stage=%(stage)s | %(message)s"
        )
    )
    handler.addFilter(_DefaultContextFilter())
    return handler


def configure_run_logger(run_id: str, workspace_root: Path) -> RunLoggerHandle:
    log_dir = Path(workspace_root).expanduser().resolve() / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / LOG_FILENAME
    logger_name = f"{RUNTIME_LOGGER_NAME}.{run_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    expected_log_path = str(log_path.resolve())
    with _LOGGER_LOCK:
        current_handlers = list(logger.handlers)
        for handler in current_handlers:
            if getattr(handler, "baseFilename", None) != expected_log_path:
                logger.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass

        if not any(getattr(handler, "baseFilename", None) == expected_log_path for handler in logger.handlers):
            logger.addHandler(_build_handler(log_path))

    adapter = RuntimeLoggerAdapter(
        logger,
        {
            "run_id": run_id,
            "ticker": "-",
            "stage": "-",
        },
    )
    return RunLoggerHandle(logger=adapter, log_path=log_path)


def close_run_logger(run_id: str) -> None:
    logger_name = f"{RUNTIME_LOGGER_NAME}.{run_id}"
    logger = logging.getLogger(logger_name)
    with _LOGGER_LOCK:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.flush()
            except Exception:
                pass
            try:
                handler.close()
            except Exception:
                pass


def bind_runtime_logger(logger: RuntimeLoggerAdapter, **context: object) -> RuntimeLoggerAdapter:
    return logger.bind(**context)
