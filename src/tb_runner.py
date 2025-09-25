"""Terminal-Bench runner with Grok integration"""
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.grok_client import GrokClient
from src.terminal_agent import GrokTerminalAgent

class TerminalBenchRunner:
    """Handles running Terminal-Bench with Grok"""
    
    def __init__(self, model: str = None):
        self.model = model or os.getenv('GROK_MODEL', 'grok-2-1212')
        self.client = GrokClient(model=self.model)
        
    def test_connection(self) -> bool:
        """Test Grok API connection"""
        print("   Sending test request to Grok API...")
        try:
            response = self.client.generate(
                "Respond with 'OK' and nothing else.",
                temperature=0.0,
                max_tokens=10
            )
            success = 'OK' in response
            if success:
                print("   Response received successfully.")
            else:
                print("   Unexpected response content.")
            return success
        except Exception as e:
            print(f"   Connection test failed: {e}")
            return False
    
    def run_with_tb_cli(
        self,
        dataset: str = "terminal-bench-core==0.1.1",
        task_ids: Optional[List[str]] = None,
        n_concurrent: int = 1,
        n_attempts: int = 1
    ) -> Dict[str, Any]:
        """Run using terminal-bench CLI with custom agent"""
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/tb_{self.model}_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory created: {output_dir}")
        
        # Build command
        cmd = [
            "tb", "run",
            "--dataset", dataset,
            "--agent-import-path", "src.terminal_agent:GrokTerminalAgent",
            "--n-concurrent", str(n_concurrent),
            "--n-attempts", str(n_attempts),
            "--output-path", str(output_dir),
            "--agent-kwarg", f"model={self.model}",
        ]
        
        # Add specific tasks if provided
        if task_ids:
            for task_id in task_ids:
                cmd.extend(["--task-id", task_id])
        
        print(f"Executing Terminal-Bench command: {' '.join(cmd)}")
        print("Live progress from Terminal-Bench will be shown below...\n")
        
        # Set environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path.cwd())
        
        # Run the command with live output (no capture)
        try:
            result = subprocess.run(
                cmd,
                env=env,
                check=True,  # Raise on error
                timeout=3600  # 1 hour timeout for long benchmarks
            )
            print("\nBenchmark completed successfully.")
            
            # Parse results if available
            results_file = output_dir / "results.json"
            if results_file.exists():
                print(f"Loading results from {results_file}...")
                with open(results_file) as f:
                    return json.load(f)
            
            # Check for alternative output formats
            for file in output_dir.glob("*.json"):
                print(f"Loading alternative results from {file}...")
                with open(file) as f:
                    return json.load(f)
                    
            return {"status": "completed", "message": "No JSON results found, check output_dir for logs"}
            
        except subprocess.TimeoutExpired:
            print("Benchmark timed out after 1 hour.")
            return {"status": "timeout"}
        except subprocess.CalledProcessError as e:
            print(f"Benchmark failed with exit code {e.returncode}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            print(f"Unexpected error running benchmark: {e}")
            return {"status": "error", "message": str(e)}
    
    def run_simple_docker_test(self) -> Dict[str, Any]:
        """Run a simple Docker-based test without terminal-bench"""
        
        print("Running simple Docker test...")
        
        # Test task
        task = {
            "instruction": "Create a file named 'test.txt' containing 'Hello from Grok!'",
            "check": "cat test.txt",
            "expected": "Hello from Grok!"
        }
        
        print("   Starting Docker container...")
        container_id = subprocess.run(
            "docker run -d -it ubuntu:22.04 /bin/bash",
            shell=True,
            capture_output=True,
            text=True
        ).stdout.strip()
        
        try:
            print(f"   Container ID: {container_id}")
            # Run task
            messages = [
                {
                    "role": "system",
                    "content": "You are in a Linux terminal. Respond with shell commands only."
                },
                {
                    "role": "user",
                    "content": f"Task: {task['instruction']}\nWhat command should I run?"
                }
            ]
            
            print("   Generating command with Grok...")
            response = self.client.chat_completion(messages, temperature=0.0)
            command = response['choices'][0]['message']['content'].strip()
            print(f"   Generated command: {command}")
            
            # Execute command
            print("   Executing command in container...")
            subprocess.run(
                f"docker exec {container_id} bash -c '{command}'",
                shell=True
            )
            
            # Check result
            print("   Checking result...")
            check_result = subprocess.run(
                f"docker exec {container_id} bash -c '{task['check']}'",
                shell=True,
                capture_output=True,
                text=True
            )
            
            success = task['expected'] in check_result.stdout
            if success:
                print("   Test passed!")
            else:
                print("   Test failed - unexpected output.")
            
            return {
                "test": "simple_docker",
                "success": success,
                "command": command,
                "output": check_result.stdout
            }
            
        finally:
            # Cleanup
            print("   Cleaning up container...")
            subprocess.run(f"docker rm -f {container_id}", shell=True, capture_output=True)