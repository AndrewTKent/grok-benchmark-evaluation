# src/progress.py
from __future__ import annotations
import json, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Any, Dict

PROGRESS_PREFIX = "TB_PROGRESS"  # Visible in pane/logs, easy to grep

@dataclass
class ProgressEvent:
    ts: float
    phase: str            # e.g., "init", "plan", "act", "verify", "done"
    msg: str              # short human hint
    step: int | None = None
    total_steps: int | None = None
    extra: Dict[str, Any] | None = None

class ProgressReporter:
    """Writes JSONL progress + returns timestamped markers for AgentResult."""
    def __init__(self, logging_dir: Optional[Path] = None):
        self.logging_dir = logging_dir
        self.jsonl: Optional[Path] = None
        if logging_dir:
            logging_dir.mkdir(parents=True, exist_ok=True)
            self.jsonl = logging_dir / "progress.jsonl"
        self._markers: List[Tuple[float, str]] = []

    def record(self, event: ProgressEvent) -> Tuple[float, str]:
        data = asdict(event)
        # 1) persist to file
        if self.jsonl:
            with self.jsonl.open("a") as f:
                f.write(json.dumps(data) + "\n")
        # 2) return a compact marker label for AgentResult
        label = f"{event.phase}:{event.msg}"
        self._markers.append((event.ts, label))
        return event.ts, label

    @property
    def markers(self) -> List[Tuple[float, str]]:
        return list(self._markers)
