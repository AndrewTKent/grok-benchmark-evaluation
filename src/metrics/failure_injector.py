import random
from typing import List, Dict

class FailureInjector:
    def __init__(self, injection_rate: float = 0.1):
        self.injection_rate = injection_rate
        self.injected_failures: List[Dict] = []
        self.recovery_successes = 0
        self.recovery_attempts = 0

    def should_inject_failure(self, step: int, task_id: str) -> bool:
        seed = hash(f"{task_id}_{step}") % 100
        return seed < (self.injection_rate * 100)

    def inject_failure(self, command: str) -> str:
        failures = [
            ("permission","chmod 000"), ("missing_file","rm -f /tmp/need.txt && cat"),
            ("syntax", (command[:-1] if command else "")), ("timeout","sleep 10 &&"),
        ]
        ftype, inj = random.choice(failures)
        self.injected_failures.append({"type": ftype, "original": command, "injected": f"{inj} {command}"})
        return f"{inj} {command}"

    def record_recovery_attempt(self, success: bool):
        self.recovery_attempts += 1
        if success: self.recovery_successes += 1

    def get_recovery_score(self) -> float:
        if self.recovery_attempts == 0: return 1.0
        return self.recovery_successes / self.recovery_attempts
