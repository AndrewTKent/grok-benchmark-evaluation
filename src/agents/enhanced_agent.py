# src/agents/enhanced_grok_agent.py
"""Enhanced Terminal-Bench Agent with composite scoring, loop detection, and failure injection (refactored)."""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Ensure package root in sys.path
sys.path.insert(0, str(Path(__file__).parents[1]))

from terminal_bench.agents.base_agent import BaseAgent, AgentResult
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession

from src.clients.grok_client import GrokClient
from src.metrics.loop_detector import CommandLoopDetector
from src.metrics.safety_checker import SafetyChecker
from src.metrics.failure_injector import FailureInjector
from src.metrics.scoring import compute_composite_score
from src.utils.tmux_compat import safe_read_pane, send_with_enter


@dataclass
class EnhancedMetrics:
    """Enhanced metrics for comprehensive evaluation"""
    success: bool = False
    composite_score: float = 0.0
    efficiency_score: float = 0.0
    recovery_score: float = 0.0
    safety_score: float = 0.0
    loop_count: int = 0
    recovery_attempts: int = 0
    safety_violations: int = 0
    steps_taken: int = 0
    time_taken: float = 0.0
    commands_executed: List[str] = None
    error_patterns: Dict[str, int] = None

    def __post_init__(self):
        if self.commands_executed is None:
            self.commands_executed = []
        if self.error_patterns is None:
            self.error_patterns = {}


class EnhancedGrokTerminalAgent(BaseAgent):
    """
    Enhanced Terminal-Bench agent with composite scoring and advanced metrics.
    """

    def __init__(
        self,
        model: str = None,
        enable_loop_detection: bool = True,
        enable_failure_injection: bool = False,
        injection_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model or os.getenv("GROK_MODEL", "grok-2-1212")
        self.client = GrokClient(model=self.model)

        # Enhanced components
        self.loop_detector = CommandLoopDetector() if enable_loop_detection else None
        self.failure_injector = FailureInjector(injection_rate) if enable_failure_injection else None
        self.safety_checker = SafetyChecker()

        # Metrics tracking
        self.current_metrics = EnhancedMetrics()
        self.all_task_metrics: List[EnhancedMetrics] = []

        # Conversation management
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 10
        self.step_count = 0
        self.current_task_instruction: Optional[str] = None
        self.task_start_time: Optional[float] = None

        # Debug mode
        self.debug = os.getenv("GROK_DEBUG", "false").lower() == "true"

        if self.debug:
            print(f"[EnhancedGrokAgent] Initialized with model: {self.model}")
            print(f"  Loop Detection       : {enable_loop_detection}")
            print(f"  Failure Injection    : {enable_failure_injection}")
            if enable_failure_injection:
                print(f"  Injection Rate       : {injection_rate}")

    # ------------------------- BaseAgent required API -------------------------

    @staticmethod
    def name() -> str:
        return "EnhancedGrokTerminalAgent"

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Optional[Path] = None,
    ) -> AgentResult:
        """Execute task with enhanced metrics tracking."""
        self.task_start_time = time.time()
        self.current_metrics = EnhancedMetrics()

        rendered_instruction = self._render_instruction(instruction)
        self.set_task_instruction(rendered_instruction)

        # Create enhanced logging directory
        if logging_dir:
            enhanced_dir = logging_dir / "enhanced_metrics"
            enhanced_dir.mkdir(exist_ok=True)
        else:
            enhanced_dir = None

        markers: List[Tuple[float, str]] = []
        max_steps = 20

        for step in range(1, max_steps + 1):
            self.step_count = step
            self.current_metrics.steps_taken = step

            # Get current observation
            observation = safe_read_pane(session)

            # Get next action (enhanced)
            command = self.get_enhanced_action(observation)

            # Loop detection
            if self.loop_detector and self.loop_detector.check_for_loop(command):
                if self.debug:
                    print(f"[LOOP DETECTED] Command repetition: {command}")
                self.current_metrics.loop_count += 1
                markers.append((time.time(), f"loop_detected:{command[:30]}"))

            # Safety check
            is_safe, safety_penalty = self.safety_checker.check_command(command)
            if not is_safe:
                self.current_metrics.safety_violations += 1
                markers.append((time.time(), f"safety_violation:{safety_penalty}"))

            # Failure injection (optional)
            if self.failure_injector and self.failure_injector.should_inject_failure(step, instruction):
                original = command
                command = self.failure_injector.inject_failure(command)
                markers.append((time.time(), f"failure_injected:{command[:30]}"))
                if self.debug:
                    print(f"[FAILURE INJECTED] Original: {original}, Injected: {command}")

            # Execute command
            self.current_metrics.commands_executed.append(command)
            send_with_enter(session, command)
            time.sleep(0.2)

            # Check result
            output = safe_read_pane(session)

            if "TASK_COMPLETE" in output:
                self.current_metrics.success = True
                markers.append((time.time(), "task_complete"))
                break

            if "TASK_FAILED" in output:
                self.current_metrics.success = False
                markers.append((time.time(), "task_failed"))
                break

            # Error classification and recovery attempt bookkeeping
            error_type = self._classify_error(output)
            if error_type:
                self.current_metrics.error_patterns[error_type] = (
                    self.current_metrics.error_patterns.get(error_type, 0) + 1
                )

                if self.failure_injector and len(self.current_metrics.commands_executed) > 1:
                    # naive recovery success heuristic: "error" not in output
                    recovery_success = "error" not in output.lower()
                    self.failure_injector.record_recovery_attempt(recovery_success)
                    if recovery_success:
                        markers.append((time.time(), "recovery_success"))

        # Final metrics
        self.current_metrics.time_taken = time.time() - self.task_start_time
        self._calculate_composite_score()

        # Persist metrics
        if enhanced_dir:
            self._save_enhanced_metrics(enhanced_dir, instruction)

        # Aggregate history
        self.all_task_metrics.append(self.current_metrics)

        return AgentResult(
            total_input_tokens=0,
            total_output_tokens=0,
            failure_mode=FailureMode.NONE if self.current_metrics.success else FailureMode.SYSTEM,
            timestamped_markers=markers,
        )

    # ------------------------- Enhanced action & scoring -------------------------

    def get_enhanced_action(self, observation: str) -> str:
        """Get next action using enhanced prompting."""
        system_prompt = self._build_enhanced_system_prompt()

        if self.loop_detector and self.loop_detector.loop_penalties > 0:
            system_prompt += f"\n\nWARNING: You have repeated commands {self.loop_detector.loop_penalties} times. Try a DIFFERENT approach!"

        messages = self._build_messages(system_prompt, observation)

        try:
            response = self.client.chat_completion(
                messages=messages,
                temperature=0.0 if self.step_count < 5 else 0.3,
                max_tokens=200,
            )
            raw = response["choices"][0]["message"]["content"]
            command = self._clean_command(raw)
            command = self._validate_enhanced_command(command, observation)
            self._update_history(observation, command)
            return command
        except Exception as e:
            print(f"[EnhancedAgent] Error: {e}")
            return "echo 'Agent error: Failed to generate command'"

    def _calculate_composite_score(self) -> None:
        """Compute composite score and component scores."""
        # Efficiency subscore
        max_steps = 20
        eff_steps = max(0.0, 1.0 - (self.current_metrics.steps_taken / max_steps))
        eff_time = max(0.0, 1.0 - (self.current_metrics.time_taken / 300.0))
        self.current_metrics.efficiency_score = (eff_steps + eff_time) / 2.0

        # Recovery subscore
        if self.failure_injector:
            self.current_metrics.recovery_score = self.failure_injector.get_recovery_score()
        else:
            total_errors = sum(self.current_metrics.error_patterns.values())
            if total_errors > 0:
                unique_errors = len(self.current_metrics.error_patterns)
                self.current_metrics.recovery_score = min(1.0, unique_errors / max(1, total_errors))
            else:
                self.current_metrics.recovery_score = 1.0

        # Safety subscore
        self.current_metrics.safety_score = self.safety_checker.get_safety_score()

        # Loop score
        loop_score = self.loop_detector.get_loop_score() if self.loop_detector else 1.0

        # Composite
        self.current_metrics.composite_score = compute_composite_score(
            success=self.current_metrics.success,
            steps_taken=self.current_metrics.steps_taken,
            time_taken_sec=self.current_metrics.time_taken,
            recovery_score=self.current_metrics.recovery_score,
            safety_score=self.current_metrics.safety_score,
            loop_score=loop_score,
        )

    # ------------------------- Prompting & parsing helpers -------------------------

    def _build_enhanced_system_prompt(self) -> str:
        base = """You are an advanced bash automation agent with recovery capabilities.

CRITICAL RULES:
1. ONE bash command per response - no explanations
2. LEARN from failures - never repeat failed commands
3. ADAPT your approach when commands fail
4. PRIORITIZE safety - avoid dangerous operations

RECOVERY STRATEGIES:
- If "command not found": apt-get install or use alternative
- If "permission denied": use sudo or chmod appropriately
- If "file not found": create it or check path
- If syntax error: fix the syntax, don't repeat

EFFICIENCY GOALS:
- Minimize steps to completion
- Avoid unnecessary operations
- Use efficient commands (grep vs cat | grep)

TASK COMPLETION:
- Success: echo "TASK_COMPLETE"
- Failure: echo "TASK_FAILED: reason"
"""
        if self.current_task_instruction:
            base += f"\n\nTASK: {self.current_task_instruction}\n"
        return base

    def _build_messages(self, system_prompt: str, observation: str) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}]
        history_window = min(self.max_history, len(self.conversation_history))
        if history_window > 0:
            messages.extend(self.conversation_history[-history_window:])

        max_len = 2000
        if len(observation) > max_len:
            observation = observation[: max_len // 2] + "\n... [TRUNCATED] ...\n" + observation[-max_len // 2:]

        messages.append({"role": "user", "content": f"Terminal output:\n{observation}\n\nNext command:"})
        return messages

    def _validate_enhanced_command(self, command: str, observation: str) -> str:
        if self.loop_detector and len(self.loop_detector.command_history) > 1:
            last_cmd = self.loop_detector.command_history[-1]
            if command == last_cmd and "error" in observation.lower():
                if "command not found" in observation:
                    missing = self._extract_missing_command(observation)
                    if missing:
                        return f"apt-get update && apt-get install -y {missing}"
                return "echo 'Breaking loop - trying alternative'"
        return command

    @staticmethod
    def _clean_command(raw_response: str) -> str:
        command = (raw_response or "").strip()
        if "```" in command:
            parts = command.split("```")
            if len(parts) >= 2:
                command = parts[1]
                if command.startswith(("bash", "sh", "shell")):
                    command = "\n".join(command.split("\n")[1:])
        command = command.strip("`").strip()
        lines = command.split("\n")
        return lines[0].strip() if lines else command

    @staticmethod
    def _extract_missing_command(observation: str) -> Optional[str]:
        for line in observation.split("\n"):
            if "command not found" in line.lower():
                parts = line.split(":")
                if len(parts) >= 2:
                    # try to get the token before colon if present, else the right side
                    left = parts[0].split()
                    return (left[-1] if left else parts[1].strip())
        return None

    def _update_history(self, observation: str, command: str) -> None:
        self.conversation_history.append({"role": "user", "content": f"Terminal: {observation[:500]}"})
        self.conversation_history.append({"role": "assistant", "content": command})
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]

    def set_task_instruction(self, instruction: str) -> None:
        self.current_task_instruction = instruction
        self.conversation_history = []
        self.step_count = 0
        if self.loop_detector:
            self.loop_detector.command_history = []
            self.loop_detector.loop_penalties = 0
        if self.safety_checker:
            self.safety_checker.violations = []
            self.safety_checker.safety_score = 1.0

    # ------------------------- Error classification -------------------------

    @staticmethod
    def _classify_error(output: str) -> Optional[str]:
        out = (output or "").lower()
        patterns = {
            "command_not_found": ["command not found", "not found"],
            "permission_denied": ["permission denied", "operation not permitted", "access denied"],
            "file_not_found": ["no such file", "cannot find", "does not exist"],
            "syntax_error": ["syntax error", "unexpected token", "parse error"],
            "timeout": ["timeout", "timed out"],
            "connection": ["connection refused", "network unreachable"],
        }
        for typ, pats in patterns.items():
            if any(p in out for p in pats):
                return typ
        if "error" in out:
            return "general_error"
        return None

    # ------------------------- Persistence -------------------------

    def _save_enhanced_metrics(self, enhanced_dir: Path, task_id: str) -> None:
        metrics_file = enhanced_dir / f"metrics_{task_id[:30]}.json"
        metrics_dict = asdict(self.current_metrics)
        metrics_dict["task_id"] = task_id
        metrics_dict["model"] = self.model
        metrics_dict["timestamp"] = time.time()

        if self.loop_detector:
            metrics_dict["loop_patterns"] = self.loop_detector.detected_patterns
        if self.failure_injector:
            metrics_dict["injected_failures"] = self.failure_injector.injected_failures
        metrics_dict["safety_violations"] = self.safety_checker.violations

        with metrics_file.open("w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)
