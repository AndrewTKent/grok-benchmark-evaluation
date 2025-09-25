# src/terminal_agent.py
"""Terminal-Bench Agent implementation for Grok — TB-compatible version with progress tracking (fixed)"""
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from terminal_bench.agents.base_agent import BaseAgent, AgentResult
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession

from src.grok_client import GrokClient

PROGRESS_PREFIX = "TB_PROGRESS"  # Visible in pane/logs, easy to grep


# ---- Progress helpers ----

@dataclass
class ProgressEvent:
    ts: float
    phase: str            # e.g., "init", "plan", "act", "verify", "done"
    msg: str              # short human hint
    step: Optional[int] = None
    total_steps: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None


class ProgressReporter:
    """Writes JSONL progress + accumulates timestamped markers for AgentResult."""
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
            with self.jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
        # 2) return a compact marker label for AgentResult
        label = f"{event.phase}:{event.msg}"
        self._markers.append((event.ts, label))
        return event.ts, label

    @property
    def markers(self) -> List[Tuple[float, str]]:
        return list(self._markers)


class GrokTerminalAgent(BaseAgent):
    """
    Terminal-Bench compatible agent that uses Grok.

    Implements the BaseAgent abstract interface exactly as required:
      - name() -> str
      - perform_task(instruction: str, session: TmuxSession, logging_dir: Path | None) -> AgentResult

    Also includes a stateless helper method get_action(observation) returning the next shell command.
    """

    def __init__(self, model: str = None, **kwargs):
        super().__init__(**kwargs)  # BaseAgent handles version/prompt_template kwargs
        self.model = model or os.getenv("GROK_MODEL", "grok-2-1212")
        self.client = GrokClient(model=self.model)

        # Lightweight conversational context for next-command prediction
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 10
        self.step_count = 0
        self.current_task_instruction: Optional[str] = None

        # Debug mode from environment
        self.debug = os.getenv("GROK_DEBUG", "false").lower() == "true"
        if self.debug:
            print(f"[GrokTerminalAgent] Initialized with model: {self.model}")
            if self.version:
                print(f"[GrokTerminalAgent] Version: {self.version}")
            if self.prompt_template:
                print(f"[GrokTerminalAgent] Using prompt template: {self.prompt_template}")

    # ------------------------- BaseAgent required API -------------------------

    @staticmethod
    def name() -> str:
        """Return the name of this agent (required by BaseAgent)."""
        return "GrokTerminalAgent"

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Optional[Path] = None,
    ) -> AgentResult:
        """
        Execute a task described by `instruction` using the provided tmux `session`.

        Progress reporting:
        • JSONL progress at logging_dir/progress.jsonl
        • Echoed TB_PROGRESS lines (visible in TB logs and runner console)
        • timestamped_markers on AgentResult
        """
        rendered_instruction = self._render_instruction(instruction)
        self.set_task_instruction(rendered_instruction)

        # progress reporter
        progress = ProgressReporter(logging_dir)
        markers: List[Tuple[float, str]] = []

        def now_ts() -> float:
            try:
                return session.get_asciinema_timestamp()
            except Exception:
                return time.time()

        # init progress
        ts, label = progress.record(ProgressEvent(
            ts=now_ts(), phase="init", msg="received_instruction",
            extra={"chars": len(rendered_instruction)}
        ))
        markers.append((ts, label))
        try:
            self._echo_progress(session, {"phase": "init", "msg": "received_instruction"})
        except Exception:
            pass

        # Minimal perception→action loop with progress
        max_steps = 20
        for step in range(1, max_steps + 1):
            observation = self._safe_read_pane(session)
            cmd = self.get_action(observation)

            ts, label = progress.record(ProgressEvent(
                ts=now_ts(), phase="act", msg="next_command",
                step=step, total_steps=max_steps, extra={"cmd_preview": (cmd or "")[:160]}
            ))
            markers.append((ts, label))
            try:
                self._echo_progress(session, {"phase": "act", "step": step, "total": max_steps})
            except Exception:
                pass

            # Send the command (always press Enter in a TB-compatible way)
            self._send_with_enter(session, cmd)
            time.sleep(0.2)

            out = self._safe_read_pane(session)
            if "TASK_COMPLETE" in out:
                ts, label = progress.record(ProgressEvent(
                    ts=now_ts(), phase="done", msg="task_complete", step=step, total_steps=step
                ))
                markers.append((ts, label))
                try:
                    self._echo_progress(session, {"phase": "done", "msg": "task_complete"})
                except Exception:
                    pass
                break

            if "TASK_FAILED" in out:
                ts, label = progress.record(ProgressEvent(
                    ts=now_ts(), phase="done", msg="task_failed", step=step, total_steps=step
                ))
                markers.append((ts, label))
                try:
                    self._echo_progress(session, {"phase": "done", "msg": "task_failed"})
                except Exception:
                    pass
                break

        return AgentResult(
            total_input_tokens=0,
            total_output_tokens=0,
            failure_mode=FailureMode.NONE,
            timestamped_markers=markers,
        )

    # ------------------------- Convenience helpers -------------------------

    def reset(self) -> None:
        """Reset agent state between tasks."""
        self.conversation_history = []
        self.step_count = 0
        self.current_task_instruction = None
        if self.debug:
            print("[GrokTerminalAgent] Agent reset for new task")

    def set_task_instruction(self, instruction: str) -> None:
        """Keep the instruction available for prompting the model."""
        self.current_task_instruction = instruction
        if self.debug:
            preview = instruction.replace("\n", " ")[:120]
            print(f"[GrokTerminalAgent] Task instruction set: {preview}...")

    def get_action(self, observation: str) -> str:
        """
        Convert terminal observation into the next shell command.
        """
        self.step_count += 1

        if self.debug:
            print(f"\n[GrokTerminalAgent] Step {self.step_count}")
            print(f"  Observation length: {len(observation)}")
            print(f"  First 200 chars: {observation[:200]}...")

        system_prompt = self._build_system_prompt()
        messages = self._build_messages(system_prompt, observation)

        try:
            response = self.client.chat_completion(
                messages=messages,
                temperature=0.0,   # Deterministic for benchmarking
                max_tokens=200     # Commands should be concise
            )
            raw_response = response["choices"][0]["message"]["content"]
            command = self._clean_command(raw_response)

            if self.debug:
                print(f"  Raw response: {raw_response}")
                print(f"  Cleaned command: {command}")

            command = self._validate_command(command, observation)
            self._update_history(observation, command)
            return command

        except Exception as e:
            print(f"[GrokTerminalAgent] Error getting Grok response: {e}")
            return "echo 'Agent error: Failed to generate command'"

    # ------------------------- Prompt construction -------------------------

    def _build_system_prompt(self) -> str:
        """Build the system prompt based on context."""
        base_prompt = """You are a bash shell automation agent. Respond with ONLY the next command.

    CRITICAL RULES:
    1. ONE bash command per response - no explanations, no markdown
    2. If previous command failed, try a DIFFERENT approach
    3. After 2 failures, install missing tools: apt-get update && apt-get install -y [package]

    ENVIRONMENT STATE:
    - You are ALWAYS in bash shell (never Python REPL)
    - If you see >>> prompt, immediately respond: exit()
    - If you see > prompt (heredoc), complete it or use Ctrl+C

    PYTHON CODE EXECUTION:
    NEVER type Python statements directly. Instead use:
    - python3 -c "code_here"
    - echo 'code' > script.py && python3 script.py
    - For multi-line: cat > script.py << 'EOF' [then on next lines add the content]

    COMMON MISTAKES TO AVOID:
    ✗ import json → ✓ python3 -c "import json; ..."
    ✗ Repeating failed commands → ✓ Try different approach
    ✗ pip3 install torch (slow) → ✓ echo "Skipping large install" or use --no-deps

    MISSING COMMANDS - Install these packages:
    - sqlite3: apt-get install -y sqlite3
    - file: apt-get install -y file  
    - pip/pip3: apt-get install -y python3-pip
    - john: Already at /app/john/run/john
    - perl: apt-get install -y perl

    HEREDOC SYNTAX:
    cat > filename << 'EOF'
    [content will follow in subsequent responses]
    EOF

    TASK COMPLETION:
    - Success: echo "TASK_COMPLETE"
    - Failure: echo "TASK_FAILED: reason"

    ANTI-PATTERNS (NEVER DO):
    1. Typing Python code in bash
    2. Repeating the same failed command
    3. Using pip for large packages without timeout consideration
    4. Forgetting to exit() from Python REPL"""

        if self.current_task_instruction:
            base_prompt += f"\n\nTASK: {self.current_task_instruction}\n"
            
        base_prompt += "\nRespond with ONLY the command. Nothing else."
        
        return base_prompt

    def _build_messages(self, system_prompt: str, observation: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

        history_window = min(self.max_history, len(self.conversation_history))
        if history_window > 0:
            messages.append({
                "role": "system",
                "content": f"You are at step {self.step_count}. Previous {history_window} actions are in the conversation history."
            })
            messages.extend(self.conversation_history[-history_window:])

        user_prompt = self._format_observation(observation)
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _format_observation(self, observation: str) -> str:
        max_obs_length = 2000
        if len(observation) > max_obs_length:
            observation = (
                observation[:max_obs_length // 2]
                + "\n... [OUTPUT TRUNCATED] ...\n"
                + observation[-max_obs_length // 2:]
            )
        return f"""Terminal output:
{observation}

What is the next command to execute? Remember: respond with ONLY the command."""

    # ------------------------- Command cleaning/validation -------------------------

    def _clean_command(self, raw_response: str) -> str:
        command = (raw_response or "").strip()
        if "```" in command:
            parts = command.split("```")
            if len(parts) >= 2:
                command = parts[1]
                for lang in ["bash", "shell", "sh", "zsh"]:
                    if command.startswith(lang):
                        command = command[len(lang):]
        command = command.strip("`").strip()

        lines = [ln.strip() for ln in command.split("\n") if ln.strip()]
        for line in lines:
            if any(line.startswith(cmd) for cmd in [
                "ls", "cd", "echo", "cat", "grep", "find", "mkdir", "touch",
                "rm", "mv", "cp", "pwd", "export", "source", "./", "python",
                "bash", "sh", "chmod", "chown", "wget", "curl", "git", "docker",
                "apt", "pip", "npm", "make", "gcc", "test", "["
            ]):
                return line
        return lines[0] if lines else command

    def _validate_command(self, command: str, observation: str) -> str:
        # Check if we're repeating a failed pattern
        if hasattr(self, '_last_failed_commands'):
            if command in self._last_failed_commands:
                # We're repeating - try to fix the root cause
                if "command not found" in observation:
                    missing = self._extract_missing_command(observation)
                    if missing == "file":
                        return "apt-get update && apt-get install -y file"
                    elif missing == "sqlite3":
                        return "apt-get update && apt-get install -y sqlite3"
                return "echo 'Avoiding command loop - trying alternative approach'"
        
        # Track failed commands
        if "command not found" in observation or "No such file" in observation:
            if not hasattr(self, '_last_failed_commands'):
                self._last_failed_commands = []
            self._last_failed_commands.append(command)
            if len(self._last_failed_commands) > 5:
                self._last_failed_commands.pop(0)
        
        return command

    def _extract_missing_command(self, observation: str) -> Optional[str]:
        for line in observation.split("\n"):
            if "command not found" in line.lower():
                parts = line.split(":")
                if len(parts) >= 2:
                    return parts[1].strip()
        return None

    # ------------------------- History maintenance -------------------------

    def _update_history(self, observation: str, command: str) -> None:
        obs_for_history = observation[:500] if len(observation) > 500 else observation
        self.conversation_history.append({
            "role": "user",
            "content": f"Terminal output:\n{obs_for_history}"
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": command
        })
        max_total_history = self.max_history * 2
        if len(self.conversation_history) > max_total_history:
            self.conversation_history = self.conversation_history[-max_total_history:]

    # ------------------------- TB compatibility helpers -------------------------

    def _safe_read_pane(self, session: TmuxSession) -> str:
        """Try various read methods for different TB versions."""
        try:
            return session.read_pane()
        except Exception:
            try:
                return session.capture_pane()
            except Exception:
                return ""

    @staticmethod
    def _single_quote(s: str) -> str:
        """Safely single-quote a string for POSIX shells."""
        return "'" + s.replace("'", "'\"'\"'") + "'"

    def _echo_progress(self, session: TmuxSession, obj: dict) -> None:
        """Echo a TB_PROGRESS line with safe quoting and guaranteed newline."""
        payload = json.dumps(obj, ensure_ascii=False)
        line = f"{PROGRESS_PREFIX} {payload}"
        cmd = f"printf %s\\n {self._single_quote(line)}"
        self._send_with_enter(session, cmd)

    # ---- unified “press Enter” sending (handles TB versions without enter= kwarg) ----
    def _send_with_enter(self, session: TmuxSession, text: str) -> None:
        """Send text and then an Enter in a way compatible with multiple TB versions."""
        try:
            # Most tmux bindings will accept literal newline appended
            session.send_keys((text or "") + "\n")
        except TypeError:
            # Fallback: send text, then send a standalone newline
            try:
                session.send_keys(text or "")
                session.send_keys("\n")
            except Exception:
                # Last resort: try carriage return
                session.send_keys(text or "")
                session.send_keys("\r")
