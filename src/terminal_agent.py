"""Terminal-Bench Agent implementation for Grok"""
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the CORRECT base class from terminal-bench
from terminal_bench.agents.base_agent import BaseAgent
from src.grok_client import GrokClient

class GrokTerminalAgent(BaseAgent):
    """Terminal-Bench compatible agent that uses Grok"""
    
    @staticmethod
    def name() -> str:
        """Return the name of the agent"""
        return "GrokTerminalAgent"
    
    def __init__(self, model: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model = model or os.getenv('GROK_MODEL', 'grok-2-1212')
        self.client = GrokClient(model=self.model)
        self.conversation_history = []
    
    def perform_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implements the abstract perform_task method required by BaseAgent.
        Executes the task by iteratively calling get_action based on task observations.
        
        Args:
            task: Dictionary containing task details (e.g., instruction, initial state)
        
        Returns:
            Dictionary with task results (e.g., actions taken, final state)
        """
        print(f"Starting task: {task.get('id', 'unknown')}")
        self.reset()  # Clear history for new task
        
        # Extract task instruction or initial observation
        observation = task.get('instruction', '') or task.get('initial_state', '')
        if not observation:
            print("Error: No instruction or initial state provided in task")
            return {"status": "error", "message": "No instruction provided"}
        
        max_steps = 10  # Prevent infinite loops
        actions = []
        
        for step in range(max_steps):
            print(f"Step {step + 1}: Processing observation...")
            command = self.get_action(observation)
            actions.append(command)
            print(f"  Executed command: {command}")
            
            # Simulate task execution (since we don't have actual terminal feedback here)
            # In a real harness, terminal-bench would provide the next observation
            # For now, assume task completes if command indicates completion
            if "Task completed" in command:
                print("Task marked as completed")
                return {
                    "status": "completed",
                    "actions": actions,
                    "final_command": command
                }
            
            # Mock next observation (replace with actual harness feedback if available)
            observation = f"Output of '{command}': [mock response]"
            
            # Stop if command is empty or invalid
            if not command or "Error occurred" in command:
                print("Task failed due to invalid command")
                return {
                    "status": "error",
                    "actions": actions,
                    "message": command
                }
        
        print("Task failed: Maximum steps reached")
        return {
            "status": "failed",
            "actions": actions,
            "message": "Maximum steps reached"
        }
    
    def get_action(self, observation: str) -> str:
        """
        Main method that terminal-bench calls to get next action.
        Takes terminal observation, returns shell command.
        """
        # System prompt for terminal tasks
        system_prompt = """You are an AI assistant operating in a Linux terminal.
You receive terminal output and must respond with shell commands to complete tasks.
Rules:
- Respond with ONLY the command to execute, no explanations
- Use proper bash syntax
- Think step-by-step but only output the command
- When task is complete, you can run 'echo "Task completed"'"""

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation history (last 5 exchanges)
        for entry in self.conversation_history[-10:]:
            messages.append(entry)
        
        # Add current observation
        messages.append({
            "role": "user",
            "content": f"Terminal output:\n{observation}\n\nWhat command should I run next? Reply with ONLY the command."
        })
        
        try:
            # Get Grok's response
            response = self.client.chat_completion(
                messages=messages,
                temperature=0.0,  # Deterministic for benchmarking
                max_tokens=200
            )
            
            command = response['choices'][0]['message']['content'].strip()
            
            # Clean the command
            command = self._clean_command(command)
            
            # Update history
            self.conversation_history.append({
                "role": "user",
                "content": f"Terminal output:\n{observation}"
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": command
            })
            
            return command
            
        except Exception as e:
            print(f"Error getting Grok response: {e}")
            return "echo 'Error occurred in agent'"
    
    def _clean_command(self, command: str) -> str:
        """Clean command output from Grok"""
        # Remove markdown code blocks if present
        if '```' in command:
            # Extract content between first pair of backticks
            parts = command.split('```')
            if len(parts) >= 2:
                # Take the content between first pair of backticks
                command = parts[1]
                # Remove language identifier if present (e.g., 'bash')
                if command.startswith('bash'):
                    command = command[4:]
                elif command.startswith('sh'):
                    command = command[2:]
        
        # Remove any leading/trailing backticks
        command = command.strip('`').strip()
        
        return command
    
    def reset(self):
        """Reset agent state between tasks"""
        self.conversation_history = []