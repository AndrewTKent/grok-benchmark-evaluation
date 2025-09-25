"""Improved Terminal-Bench runner with live progress tailing (always on)"""
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import time
import logging
import shutil
import threading
import glob
import re  # <-- added

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.grok_client import GrokClient


def _shorten(s: str, n: int = 120) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"


def _sanitize_for_fs(s: str) -> str:
    """
    Replace characters that are awkward for file systems with '-'.
    Keeps alphanumerics, dot, underscore, dash. Everything else -> '-'.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


class _Tailer:
    """
    Polling-based tailer for many files that appear dynamically under output_dir.
    Always-on, no external deps. Meant for progress.jsonl and commands.txt.
    """
    def __init__(self, output_dir: Path, poll_interval: float = 1.0):
        self.output_dir = output_dir
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        # file -> (fp, last_pos)
        self._jsonl_positions: Dict[Path, int] = {}
        self._cmd_positions: Dict[Path, int] = {}

    def start(self):
        t = threading.Thread(target=self._watch_loop, name="TBProgressWatcher", daemon=True)
        t.start()
        self._threads.append(t)

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2)

    def _watch_loop(self):
        # Print a single banner
        print("\n== Live Task Progress (progress.jsonl / commands.txt) ==")
        printed_hint = False

        while not self._stop.is_set():
            try:
                # Typical TB layout:
                # output_dir/<run_id>/<task_id>/<trial_id>/agent-logs/progress.jsonl
                jsonl_paths = glob.glob(str(self.output_dir / "*" / "*" / "*" / "agent-logs" / "progress.jsonl"))
                cmd_paths   = glob.glob(str(self.output_dir / "*" / "*" / "*" / "commands.txt"))

                # Tail JSONL progress
                for p in jsonl_paths:
                    path = Path(p)
                    if path not in self._jsonl_positions:
                        self._jsonl_positions[path] = 0  # start at beginning

                    try:
                        with path.open("r", encoding="utf-8") as f:
                            f.seek(self._jsonl_positions[path])
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    evt = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                task = self._extract_task_from_path(path)
                                phase = evt.get("phase", "?")
                                msg = evt.get("msg", "")
                                step = evt.get("step")
                                total = evt.get("total_steps")
                                step_part = f"[{step}/{total}]" if step and total else ""
                                preview = ""
                                extra = evt.get("extra") or {}
                                if "cmd_preview" in extra:
                                    preview = f"  cmd: {_shorten(extra['cmd_preview'], 100)}"
                                print(f"[TB_PROGRESS] {task} {step_part} {phase}: {msg}{(' | ' + preview) if preview else ''}")
                            self._jsonl_positions[path] = f.tell()
                    except FileNotFoundError:
                        continue
                    except PermissionError:
                        continue

                # Tail commands.txt
                for p in cmd_paths:
                    path = Path(p)
                    if path not in self._cmd_positions:
                        self._cmd_positions[path] = 0
                    try:
                        with path.open("r", encoding="utf-8") as f:
                            f.seek(self._cmd_positions[path])
                            new_lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                            if new_lines:
                                task = self._extract_task_from_path(path)
                                # Print only the last actionable command-looking line
                                last = new_lines[-1]
                                print(f"[TB_CMD] {task} -> {_shorten(last, 140)}")
                            self._cmd_positions[path] = f.tell()
                    except FileNotFoundError:
                        continue
                    except PermissionError:
                        continue

                if not printed_hint and (jsonl_paths or cmd_paths):
                    printed_hint = True
                    print("(progress streaming... press Ctrl+C to stop the run)")

                time.sleep(self.poll_interval)
            except Exception:
                # Never crash the watcher
                time.sleep(self.poll_interval)

    @staticmethod
    def _extract_task_from_path(path: Path) -> str:
        """
        Given .../<run_id>/<task_id>/<trial_id>/agent-logs/progress.jsonl
        return "<task_id>/<trial_id>"
        """
        parts = path.parts
        # Find the index of output_dir in parts; assume path ends with /<run>/<task>/<trial>/...
        # Safer: just pick the last segments
        try:
            trial = parts[-3]  # e.g., 'super-benchmark-upet.1-of-1.2025-...'
            task = parts[-4]   # e.g., 'super-benchmark-upet'
            return f"{task}/{trial}"
        except Exception:
            return str(path.parent)


class TerminalBenchRunner:
    """Handles running Terminal-Bench with Grok - Progress Always On"""

    def __init__(self, model: str = None):
        self.model = (model or os.getenv('GROK_MODEL') or os.getenv('XAI_MODEL') or 'grok-4-fast-reasoning')
        self.client = GrokClient(model=self.model)
        self.debug = os.getenv('GROK_DEBUG', 'false').lower() == 'true'

    def test_connection(self) -> bool:
        """Test Grok API connection"""
        print("Testing Grok API connection...")
        try:
            response = self.client.generate(
                "Respond with 'OK' and nothing else.",
                temperature=0.0,
                max_tokens=10
            )
            success = isinstance(response, str) and 'OK' in response
            if success:
                print(f"✓ Connection successful. Model: {self.model}")
            else:
                print(f"✗ Unexpected response: {response}")
            return success
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

    def verify_terminal_bench_setup(self) -> bool:
        """Verify Terminal-Bench is properly installed and configured"""
        print("\nVerifying Terminal-Bench setup...")
        checks_passed = True
        logger = logging.getLogger(__name__)

        # Check if tb CLI is in PATH
        if shutil.which('tb') is None:
            print("✗ Terminal-Bench CLI (tb) not found in PATH")
            print("  Try: pip install terminal-bench")
            logger.error("Terminal-Bench CLI (tb) not found in PATH")
            checks_passed = False
        else:
            # Check if tb CLI is operational
            try:
                result = subprocess.run(
                    ["tb", "--help"],
                    capture_output=True,
                    text=True,
                    timeout=20
                )
                if result.returncode == 0:
                    print("✓ Terminal-Bench CLI is operational")
                    logger.info("Terminal-Bench CLI check passed")
                else:
                    print(f"✗ Terminal-Bench CLI failed: {result.stderr.strip()}")
                    logger.error(f"Terminal-Bench CLI failed: {result.stderr}")
                    checks_passed = False
            except subprocess.TimeoutExpired:
                print("✗ Terminal-Bench CLI timed out")
                logger.error("Terminal-Bench CLI timed out")
                checks_passed = False
            except Exception as e:
                print(f"✗ Error checking Terminal-Bench: {str(e)}")
                logger.error(f"Error checking Terminal-Bench: {str(e)}")
                checks_passed = False

        # Check if Docker is running
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=8
            )
            if result.returncode == 0:
                print("✓ Docker is running")
                logger.info("Docker check passed")
            else:
                print("✗ Docker is not running")
                print("  Start Docker and try again")
                logger.error(f"Docker check failed: {result.stderr}")
                checks_passed = False
        except FileNotFoundError:
            print("✗ Docker not found")
            print("  Terminal-Bench requires Docker")
            logger.error("Docker not found")
            checks_passed = False
        except Exception as e:
            print(f"✗ Error checking Docker: {str(e)}")
            logger.error(f"Error checking Docker: {str(e)}")
            checks_passed = False

        # Check if our agent module is importable
        try:
            from src.terminal_agent import GrokTerminalAgent  # noqa: F401
            print("✓ GrokTerminalAgent module is importable")
            logger.info("GrokTerminalAgent import check passed")
        except ImportError as e:
            print(f"✗ Cannot import GrokTerminalAgent: {str(e)}")
            logger.error(f"Cannot import GrokTerminalAgent: {str(e)}")
            checks_passed = False

        return checks_passed

    def run_with_tb_cli(
        self,
        dataset: str = "terminal-bench-core==0.1.1",
        task_ids: Optional[List[str]] = None,
        n_concurrent: int = 1,
        n_attempts: int = 1,
        timeout_per_task: int = 300  # 5 minutes per task
    ) -> Dict[str, Any]:
        """
        Run Terminal-Bench using the official CLI with our custom agent.
        Live progress tailing is always enabled.
        """
        # Verify setup first
        if not self.verify_terminal_bench_setup():
            return {
                "status": "error",
                "message": "Terminal-Bench setup verification failed"
            }

        # Create output directory (timestamp removed from model folder)
        sanitized_model = _sanitize_for_fs(self.model)
        output_dir = Path(f"results/tb_{sanitized_model}")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")

        # Build the command
        cmd = self._build_tb_command(
            dataset, task_ids, n_concurrent, n_attempts, output_dir
        )

        print(f"\nExecuting command: {' '.join(cmd)}")
        print("=" * 60)
        print("Terminal-Bench execution starting...\n")

        # Set up environment
        env = self._prepare_environment()

        # Start tailer (it will pick up files as soon as they appear)
        tailer = _Tailer(output_dir)
        tailer.start()

        # Execute with timeout and live output
        start_time = time.time()
        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream tb's own output as well
            output_lines: List[str] = []
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line, end='')
                    output_lines.append(line)

            # Wait for completion
            return_code = process.wait()

            elapsed_time = time.time() - start_time
            print(f"\n{'=' * 60}")
            print(f"Execution completed in {elapsed_time:.1f} seconds")

            # Stop the tailer
            tailer.stop()

            # Parse and return results
            return self._parse_results(
                output_dir, return_code, output_lines, elapsed_time
            )

        except subprocess.TimeoutExpired:
            tailer.stop()
            print(f"\n✗ Benchmark timed out after {timeout_per_task * len(task_ids or [1])} seconds")
            return {"status": "timeout"}
        except KeyboardInterrupt:
            tailer.stop()
            print("\n✗ Benchmark interrupted by user")
            self._cleanup_docker_containers()
            return {"status": "interrupted"}
        except Exception as e:
            tailer.stop()
            print(f"\n✗ Unexpected error: {e}")
            return {"status": "error", "message": str(e)}

    def _build_tb_command(
        self,
        dataset: str,
        task_ids: Optional[List[str]],
        n_concurrent: int,
        n_attempts: int,
        output_dir: Path
    ) -> List[str]:
        """Build the Terminal-Bench CLI command"""
        cmd = [
            "tb", "run",
            "--dataset", dataset,
            "--agent-import-path", "src.terminal_agent:GrokTerminalAgent",
            "--n-concurrent", str(n_concurrent),
            "--n-attempts", str(n_attempts),
            "--output-path", str(output_dir),
        ]

        # Add model as agent kwarg
        cmd.extend(["--agent-kwarg", f"model={self.model}"])

        # Add specific task IDs if provided
        if task_ids:
            for task_id in task_ids:
                cmd.extend(["--task-id", task_id])

        # Add verbosity if debug mode
        if self.debug:
            cmd.append("--verbose")

        return cmd

    def _prepare_environment(self) -> Dict[str, str]:
        """Prepare environment variables for Terminal-Bench execution"""
        env = os.environ.copy()

        # Ensure current directory is in PYTHONPATH
        current_path = str(Path.cwd())
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{current_path}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = current_path

        # Pass through Grok-related environment variables
        grok_vars = ['XAI_API_KEY', 'GROK_API_KEY', 'GROK_MODEL', 'GROK_DEBUG', 'XAI_MODEL']
        for var in grok_vars:
            if var in os.environ:
                env[var] = os.environ[var]

        return env

    def _parse_results(
        self,
        output_dir: Path,
        return_code: int,
        output_lines: List[str],
        elapsed_time: float
    ) -> Dict[str, Any]:
        """Parse Terminal-Bench results from output directory"""
        results: Dict[str, Any] = {
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "elapsed_time": elapsed_time,
            "output_dir": str(output_dir)
        }

        # Look for Terminal-Bench result files
        result_patterns = ["results.json", "summary.json", "*.json"]
        for pattern in result_patterns:
            for result_file in output_dir.glob(pattern):
                try:
                    with open(result_file, encoding="utf-8") as f:
                        data = json.load(f)
                        results["benchmark_results"] = data
                        print(f"\n✓ Loaded results from {result_file.name}")

                        # Extract key metrics if available
                        if isinstance(data, dict):
                            if "score" in data:
                                results["score"] = data["score"]
                            if "tasks" in data:
                                results["num_tasks"] = len(data["tasks"])
                                successes = sum(
                                    1 for task in data["tasks"]
                                    if task.get("status") == "success"
                                )
                                if len(data["tasks"]) > 0:
                                    results["success_rate"] = successes / len(data["tasks"])

                        break  # Use first valid result file
                except Exception as e:
                    print(f"  Warning: Could not parse {result_file.name}: {e}")

        # If no result file found, keep raw output
        if "benchmark_results" not in results:
            results["raw_output"] = output_lines
            print("\n⚠ No result files found. Check output_dir for logs.")

        return results

    def _cleanup_docker_containers(self):
        """Clean up any running Docker containers from Terminal-Bench"""
        print("\nCleaning up Docker containers...")
        try:
            # Stop all containers with tb- prefix (Terminal-Bench convention)
            subprocess.run(
                "docker ps -q --filter 'name=tb-' | xargs -r docker stop",
                shell=True,
                capture_output=True,
                timeout=10
            )

            # Remove stopped containers
            subprocess.run(
                "docker ps -aq --filter 'name=tb-' | xargs -r docker rm",
                shell=True,
                capture_output=True,
                timeout=10
            )

            print("✓ Docker containers cleaned up")
        except Exception as e:
            print(f"  Warning: Cleanup failed: {e}")

    def run_diagnostic_test(self) -> Dict[str, Any]:
        """Run a comprehensive diagnostic test"""
        print("\n" + "=" * 60)
        print("RUNNING DIAGNOSTIC TEST")
        print("=" * 60)

        results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "checks": {}
        }

        # 1. API Connection
        print("\n1. API Connection Test:")
        results["checks"]["api"] = self.test_connection()

        # 2. Terminal-Bench Setup
        print("\n2. Terminal-Bench Setup:")
        results["checks"]["setup"] = self.verify_terminal_bench_setup()

        # 3. Simple Agent Test (without full Terminal-Bench)
        print("\n3. Agent Instantiation Test:")
        try:
            from src.terminal_agent import GrokTerminalAgent
            agent = GrokTerminalAgent(model=self.model)

            # Test get_action with a simple observation
            test_obs = "$ pwd\n/home/user\n$ "
            action = agent.get_action(test_obs)

            print(f"✓ Agent created successfully")
            print(f"  Test observation: {test_obs.strip()}")
            print(f"  Generated action: {action}")
            results["checks"]["agent"] = True
            results["test_action"] = action
        except Exception as e:
            print(f"✗ Agent test failed: {e}")
            results["checks"]["agent"] = False

        # 4. Docker Test
        print("\n4. Docker Test:")
        try:
            # Run a simple command in Docker
            test_cmd = "docker run --rm alpine echo 'Docker test successful'"
            result = subprocess.run(
                test_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )

            if "Docker test successful" in result.stdout:
                print("✓ Docker execution working")
                results["checks"]["docker"] = True
            else:
                print(f"✗ Docker test failed: {result.stderr}")
                results["checks"]["docker"] = False
        except Exception as e:
            print(f"✗ Docker test error: {e}")
            results["checks"]["docker"] = False

        # Summary
        all_passed = all(results["checks"].values())
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)
        for check, passed in results["checks"].items():
            status = "✓" if passed else "✗"
            print(f"{status} {check.upper()}: {'PASSED' if passed else 'FAILED'}")

        print("\n" + ("=" * 60))
        if all_passed:
            print("✓ All diagnostics passed! Ready to run benchmarks.")
        else:
            print("✗ Some diagnostics failed. Please fix issues before running benchmarks.")

        # Save diagnostic results
        diag_file = Path(f"results/diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        diag_file.parent.mkdir(exist_ok=True)
        with open(diag_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nDiagnostic results saved to: {diag_file}")

        return results
