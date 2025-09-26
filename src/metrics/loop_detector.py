from collections import Counter
from typing import List, Tuple

IGNORED = {"pwd","clear","ls","echo"}

class CommandLoopDetector:
    def __init__(self, window_size: int = 5, threshold: int = 2):
        self.window_size = window_size
        self.threshold = threshold
        self.command_history: List[str] = []
        self.loop_penalties = 0
        self.detected_patterns: List[Tuple[str,int]] = []

    def check_for_loop(self, command: str) -> bool:
        if (command or "").strip().split(" ")[0] in IGNORED:
            self.command_history.append(command); self._trim(); return False
        self.command_history.append(command); self._trim()
        recent = self.command_history[-self.window_size:]
        if len(recent) >= self.threshold:
            cnt = Counter(recent)
            if cnt[command] >= self.threshold:
                self.loop_penalties += 1; self.detected_patterns.append((command, cnt[command])); return True
        if len(self.command_history) >= 4:
            a,b,c,d = self.command_history[-4:]
            if a==c and b==d and a!=b:
                self.loop_penalties += 1; self.detected_patterns.append((f"alternating: {a}, {b}",2)); return True
        return False

    def get_loop_score(self) -> float:
        return max(0.0, 1.0 - 0.2 * self.loop_penalties)

    def _trim(self):
        if len(self.command_history) > self.window_size * 2:
            self.command_history = self.command_history[-self.window_size * 2:]
