"""Terminal-Bench Agent implementation for Grok - Fixed Version"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grok_client import GrokClient

class GrokTerminalAgent:
    """
    Terminal-Bench compatible agent that uses Grok.
    
    This agent implements the minimal interface required by Terminal-Bench:
    - get_action(observation) -> action
    - reset() 
    
    Terminal-Bench handles the actual execution loop and Docker containers.
    """
    
    def __init__(self, model: str = None, **kwargs):
        """Initialize the Grok agent with API client and conversation tracking."""
        self.model = model or os.getenv('GROK_MODEL', 'grok-2-1212')
        self.client = GrokClient(model=self.model)
        self.conversation_history = []
        self.max_history = 10  # Keep last N exchanges
        self.step_count = 0
        self.current_task_instruction = None
        
        # Debug mode from environment
        self.debug = os.getenv('GROK_DEBUG', 'false').lower() == 'true'
        
        if self.debug:
            print(f"[GrokTerminalAgent] Initialized with model: {self.model}")
    
    def reset(self):
        """
        Reset agent state between tasks.
        Called by Terminal-Bench at the start of each new task.
        """
        self.conversation_history = []
        self.step_count = 0
        self.current_task_instruction = None
        
        if self.debug:
            print("[GrokTerminalAgent] Agent reset for new task")
    
    def set_task_instruction(self, instruction: str):
        """
        Store the task instruction for context.
        Terminal-Bench may call this to set the initial task.
        """
        self.current_task_instruction = instruction
        if self.debug:
            print(f"[GrokTerminalAgent] Task instruction set: {instruction[:100]}...")
    
    def get_action(self, observation: str) -> str:
        """
        Main method that Terminal-Bench calls to get the next action.
        
        Args:
            observation: Current terminal output/state from Terminal-Bench
        
        Returns:
            Shell command to execute next
        """
        self.step_count += 1
        
        if self.debug:
            print(f"\n[GrokTerminalAgent] Step {self.step_count}")
            print(f"  Observation length: {len(observation)} chars")
            print(f"  First 200 chars: {observation[:200]}...")
        
        # Build the system prompt for terminal operation
        system_prompt = self._build_system_prompt()
        
        # Build conversation messages
        messages = self._build_messages(system_prompt, observation)
        
        try:
            # Get Grok's response
            response = self.client.chat_completion(
                messages=messages,
                temperature=0.0,  # Deterministic for benchmarking
                max_tokens=200    # Commands should be concise
            )
            
            # Extract and clean the command
            raw_response = response['choices'][0]['message']['content']
            command = self._clean_command(raw_response)
            
            if self.debug:
                print(f"  Raw response: {raw_response}")
                print(f"  Cleaned command: {command}")
            
            # Validate command
            command = self._validate_command(command, observation)
            
            # Update conversation history
            self._update_history(observation, command)
            
            return command
            
        except Exception as e:
            error_msg = f"Error getting Grok response: {e}"
            print(f"[GrokTerminalAgent] {error_msg}")
            
            # Return a safe fallback command
            return "echo 'Agent error: Failed to generate command'"
    
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

        # Add task-specific context if available
        if self.current_task_instruction:
            base_prompt += f"\n\nCURRENT TASK:\n{self.current_task_instruction}"
        
        return base_prompt
    
    def _build_messages(self, system_prompt: str, observation: str) -> List[Dict[str, str]]:
        """Build the message list for the API call."""
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add relevant conversation history (not all, to avoid context bloat)
        history_window = min(self.max_history, len(self.conversation_history))
        if history_window > 0:
            # Add a summary of what we've done so far
            messages.append({
                "role": "system",
                "content": f"You are at step {self.step_count}. Previous {history_window} actions are in the conversation history."
            })
            
            # Add the actual history
            for entry in self.conversation_history[-history_window:]:
                messages.append(entry)
        
        # Add current observation
        user_prompt = self._format_observation(observation)
        messages.append({"role": "user", "content": user_prompt})
        
        return messages
    
    def _format_observation(self, observation: str) -> str:
        """Format the observation for the prompt."""
        # Truncate very long observations to avoid context issues
        max_obs_length = 2000
        if len(observation) > max_obs_length:
            observation = (
                observation[:max_obs_length//2] + 
                "\n... [OUTPUT TRUNCATED] ...\n" + 
                observation[-max_obs_length//2:]
            )
        
        return f"""Terminal output:
{observation}

What is the next command to execute? Remember: respond with ONLY the command."""
    
    def _clean_command(self, raw_response: str) -> str:
        """Clean and extract the command from Grok's response."""
        command = raw_response.strip()
        
        # Remove markdown code blocks if present
        if '```' in command:
            # Extract content between backticks
            parts = command.split('```')
            if len(parts) >= 2:
                command = parts[1]
                # Remove language identifier
                for lang in ['bash', 'shell', 'sh', 'zsh']:
                    if command.startswith(lang):
                        command = command[len(lang):]
        
        # Remove inline backticks
        command = command.strip('`').strip()
        
        # Remove any explanatory text (look for common patterns)
        # If there's a line that looks like a command, extract it
        lines = command.split('\n')
        for line in lines:
            line = line.strip()
            # Heuristic: commands often start with these
            if any(line.startswith(cmd) for cmd in [
                'ls', 'cd', 'echo', 'cat', 'grep', 'find', 'mkdir', 'touch',
                'rm', 'mv', 'cp', 'pwd', 'export', 'source', './', 'python',
                'bash', 'sh', 'chmod', 'chown', 'wget', 'curl', 'git', 'docker',
                'apt', 'pip', 'npm', 'make', 'gcc', 'test', '['
            ]):
                return line
        
        # If no command pattern found, return the cleaned version
        return command
    
    def _validate_command(self, command: str, observation: str) -> str:
        """Validate and potentially fix the command."""
        # Check for empty or invalid commands
        if not command or command.isspace():
            return "echo 'No command generated'"
        
        # Check for dangerous patterns (be conservative in benchmarking)
        dangerous_patterns = [
            'rm -rf /',
            'dd if=/dev/zero',
            ':(){ :|:& };:',  # Fork bomb
            '> /dev/sda'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in command:
                return f"echo 'Safety: Blocked dangerous command pattern: {pattern}'"
        
        # Check if the command is too long (might indicate explanation text)
        if len(command) > 500:
            # Try to extract just the first line
            first_line = command.split('\n')[0].strip()
            if first_line:
                command = first_line
            else:
                return "echo 'Command too long or complex'"
        
        # Handle special cases based on observation
        if "command not found" in observation.lower() and self.step_count > 1:
            # Previous command failed, might need to install something
            missing_cmd = self._extract_missing_command(observation)
            if missing_cmd:
                # Don't auto-install in benchmarking context, just note it
                return f"echo 'Missing command: {missing_cmd}'"
        
        return command
    
    def _extract_missing_command(self, observation: str) -> Optional[str]:
        """Extract the missing command from error message."""
        lines = observation.split('\n')
        for line in lines:
            if "command not found" in line.lower():
                # Pattern: "bash: xyz: command not found"
                parts = line.split(':')
                if len(parts) >= 2:
                    return parts[1].strip()
        return None
    
    def _update_history(self, observation: str, command: str):
        """Update conversation history with the latest exchange."""
        # Add user observation (truncated for history)
        obs_for_history = observation[:500] if len(observation) > 500 else observation
        self.conversation_history.append({
            "role": "user",
            "content": f"Terminal output:\n{obs_for_history}"
        })
        
        # Add assistant response
        self.conversation_history.append({
            "role": "assistant",
            "content": command
        })
        
        # Trim history if it gets too long
        max_total_history = self.max_history * 2  # user + assistant pairs
        if len(self.conversation_history) > max_total_history:
            # Keep only the most recent exchanges
            self.conversation_history = self.conversation_history[-max_total_history:]