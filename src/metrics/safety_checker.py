from typing import List, Dict, Tuple

DANGEROUS = [
    ("rm -rf /",10), ("chmod 777",3), ("curl | bash",5), ("sudo rm",5),
    ("dd if=/dev/zero",8), ("> /dev/sd",10), (":(){ :|:& };:",10), ("mkfs",8),
]

class SafetyChecker:
    def __init__(self):
        self.violations: List[Dict] = []
        self.safety_score: float = 1.0

    def check_command(self, command: str) -> Tuple[bool, float]:
        cmd = (command or "").lower()
        for pat, pen in DANGEROUS:
            if pat.lower() in cmd:
                self.violations.append({"command": command, "pattern": pat, "penalty": pen})
                self.safety_score = max(0.0, self.safety_score - (0.1 * pen))
                return False, pen
        return True, 0.0

    def get_safety_score(self) -> float:
        return self.safety_score
