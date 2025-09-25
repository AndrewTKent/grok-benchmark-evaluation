"""Terminal-Bench Agent implementation for Grok — TB-compatible version"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from terminal_bench.agents.base_agent import BaseAgent, AgentResult
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession

from src.grok_client import GrokClient


class GrokTerminalAgent(BaseAgent):
    """
    Terminal-Bench compatible agent that uses Grok.

    Implements the BaseAgent abstract interface exactly as required:
      - name() -> str
      - perform_task(instruction: str, session: TmuxSession, logging_dir: Path | None) -> AgentResult

    Also includes a stateless helper method get_action(observation) returning the next shell command
    when you wire it into a tmux loop (optional; not mandated by BaseAgent).
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
        logging_dir: Path | None = None,
    ) -> AgentResult:
        """
        Execute a task described by `instruction` using the provided tmux `session`.

        This minimal implementation:
          - Applies an optional Jinja2 prompt template via BaseAgent._render_instruction
          - Records a timestamped marker using session.get_asciinema_timestamp()
          - Stores the instruction for inclusion in the system prompt used by get_action()
          - Returns an AgentResult with required fields

        You can extend this to run a full perception→action loop by:
          * reading pane output from `session`,
          * calling `self.get_action(observation)`,
          * sending commands with `session.send_keys(command)`,
          * and terminating when you detect completion (e.g., TASK_COMPLETE).
        """
        # Honor optional prompt template from BaseAgent
        rendered_instruction = self._render_instruction(instruction)

        # Save task instruction for system prompts
        self.set_task_instruction(rendered_instruction)

        # Example of recording a timestamped marker for the harness
        markers: List[Tuple[float, str]] = []
        try:
            ts = session.get_asciinema_timestamp()
            markers.append((ts, "received_instruction"))
        except Exception:
            # If session isn't fully initialized for timestamps yet, ignore
            pass

        # If desired, you could do a no-op announcement in the shell (disabled by default):
        # session.send_keys("echo 'GrokTerminalAgent ready'")

        # NOTE: Token accounting is model/integration specific; set to 0 if not tracked
        result = AgentResult(
            total_input_tokens=0,
            total_output_tokens=0,
            failure_mode=FailureMode.NONE,
            timestamped_markers=markers,
        )

        if self.debug:
            print(f"[GrokTerminalAgent] perform_task: instruction length={len(rendered_instruction)}")
            if logging_dir:
                print(f"[GrokTerminalAgent] logging_dir: {logging_dir}")

        return result

    # ------------------------- Convenience helpers (optional) -------------------------

    def reset(self):
        """Reset agent state between tasks (not required by BaseAgent but useful)."""
        self.conversation_history = []
        self.step_count = 0
        self.current_task_instruction = None
        if self.debug:
            print("[GrokTerminalAgent] Agent reset for new task")

    def set_task_instruction(self, instruction: str):
        """Keep the instruction available for prompting the model."""
        self.current_task_instruction = instruction
        if self.debug:
            preview = instruction.replace("\n", " ")[:120]
            print(f"[GrokTerminalAgent] Task instruction set: {preview}...")

    def get_action(self, observation: str) -> str:
        """
        Stateless helper to convert terminal observation into the next shell command.
        Wire this into a loop that reads from `session` and calls `session.send_keys(...)`.
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
        base_prompt = """You are an AI agent operating in a Linux terminal environment through Terminal-Bench.
You receive terminal output and must respond with shell commands to complete tasks.

CRITICAL RULES:
1. Respond with ONLY the command to execute - no explanations, no markdown, no commentary
2. Use proper bash syntax
3. Be careful with file operations - use absolute paths when needed
4. Check for command success before proceeding
5. If you need to signal task completion, use: echo "TASK_COMPLETE"
6. If you encounter an error you cannot resolve, use: echo "TASK_FAILED: <reason>"

EXECUTION CONTEXT:
- You are in a Docker container with standard Linux tools
- The working directory may change based on your commands
- Files you create persist within the task session
- Network access may be limited"""
        if self.current_task_instruction:
            base_prompt += f"\n\nCURRENT TASK:\n{self.current_task_instruction}"
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
        command = raw_response.strip()
        if "```" in command:
            parts = command.split("```")
            if len(parts) >= 2:
                command = parts[1]
                for lang in ["bash", "shell", "sh", "zsh"]:
                    if command.startswith(lang):
                        command = command[len(lang):]
        command = command.strip("`").strip()

        lines = command.split("\n")
        for line in lines:
            line = line.strip()
            if any(line.startswith(cmd) for cmd in [
                "ls", "cd", "echo", "cat", "grep", "find", "mkdir", "touch",
                "rm", "mv", "cp", "pwd", "export", "source", "./", "python",
                "bash", "sh", "chmod", "chown", "wget", "curl", "git", "docker",
                "apt", "pip", "npm", "make", "gcc", "test", "["
            ]):
                return line
        return command

    def _validate_command(self, command: str, observation: str) -> str:
        if not command or command.isspace():
            return "echo 'No command generated'"

        dangerous = ["rm -rf /", "dd if=/dev/zero", ":(){ :|:& };:", "> /dev/sda"]
        for pattern in dangerous:
            if pattern in command:
                return f"echo 'Safety: Blocked dangerous command pattern: {pattern}'"

        if len(command) > 500:
            first_line = command.split("\n")[0].strip()
            if first_line:
                command = first_line
            else:
                return "echo 'Command too long or complex'"

        if "command not found" in observation.lower() and self.step_count > 1:
            missing_cmd = self._extract_missing_command(observation)
            if missing_cmd:
                return f"echo 'Missing command: {missing_cmd}'"

        return command

    def _extract_missing_command(self, observation: str) -> Optional[str]:
        for line in observation.split("\n"):
            if "command not found" in line.lower():
                parts = line.split(":")
                if len(parts) >= 2:
                    return parts[1].strip()
        return None

    # ------------------------- History maintenance -------------------------

    def _update_history(self, observation: str, command: str):
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
