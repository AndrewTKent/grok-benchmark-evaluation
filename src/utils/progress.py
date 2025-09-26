from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json, os

@dataclass
class ProgressEvent:
    ts: float
    phase: str
    msg: str
    step: Optional[int] = None
    total_steps: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None

class ProgressReporter:
    def __init__(self, logging_dir: Optional[Path] = None):
        self.logging_dir = logging_dir
        self.jsonl: Optional[Path] = None
        if logging_dir:
            logging_dir.mkdir(parents=True, exist_ok=True)
            self.jsonl = logging_dir / "progress.jsonl"
        self._markers: List[Tuple[float, str]] = []

    def record(self, event: ProgressEvent) -> Tuple[float, str]:
        data = asdict(event)
        if self.jsonl:
            with self.jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                f.flush()
                try: os.fsync(f.fileno())
                except OSError: pass
        label = f"{event.phase}:{event.msg}"
        self._markers.append((event.ts, label))
        return event.ts, label

    @property
    def markers(self) -> List[Tuple[float, str]]:
        return list(self._markers)
