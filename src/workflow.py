"""Structured workflow tracking for long benchmark runs.

The tracker writes both a human-readable logger stream and a JSON summary that
captures the current stage, timing, and per-step status. This keeps the entire
pipeline inspectable after interruption without depending on console output.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def setup_logging(output: Path | None = None, verbose: bool = True) -> logging.Logger:
    """Configure a single project logger for console and optional file output."""
    logger = logging.getLogger("deap_benchmark")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO if verbose else logging.WARNING)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output / "workflow.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


@dataclass
class WorkflowTracker:
    """Accumulate run status and persist a structured execution trace."""

    output: Path
    run_name: str
    logger: logging.Logger | None = None
    started_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    stages: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)

    def log(self, level: int, message: str, **data: Any) -> None:
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "level": logging.getLevelName(level),
            "message": message,
        }
        if data:
            event["data"] = data
        self.events.append(event)
        if self.logger is not None:
            self.logger.log(level, message + (f" | {data}" if data else ""))
        self._write()

    @contextmanager
    def stage(self, name: str, **data: Any) -> Iterator[None]:
        entry: dict[str, Any] = {
            "name": name,
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        if data:
            entry["data"] = data
        self.stages.append(entry)
        self._write()
        self.log(logging.INFO, f"stage_started:{name}", **data)
        try:
            yield
        except Exception as exc:
            entry["status"] = "failed"
            entry["finished_utc"] = datetime.now(timezone.utc).isoformat()
            entry["error"] = f"{type(exc).__name__}: {exc}"
            self._write()
            self.log(logging.ERROR, f"stage_failed:{name}", error=entry["error"])
            raise
        else:
            entry["status"] = "ok"
            entry["finished_utc"] = datetime.now(timezone.utc).isoformat()
            self._write()
            self.log(logging.INFO, f"stage_completed:{name}")

    def finish(self, **data: Any) -> None:
        self.log(logging.INFO, "workflow_completed", **data)

    def _write(self) -> None:
        payload = {
            "run_name": self.run_name,
            "started_utc": self.started_utc,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "stages": self.stages,
            "events": self.events,
        }
        (self.output / "workflow.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        with (self.output / "workflow.jsonl").open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event) + "\n")
