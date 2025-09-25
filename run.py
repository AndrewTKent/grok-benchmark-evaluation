#!/usr/bin/env python3
"""
Main runner for Terminal-Bench with Grok evaluation — Progress Always On
"""
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add imports
from src.tb_runner import TerminalBenchRunner

os.system('clear')


def run_quick_test(runner: TerminalBenchRunner) -> int:
    """Run a quick test with a simple task"""
    print("=" * 60)
    print("QUICK TEST MODE")
    print("=" * 60)

    print("\nThis will test:")
    print("1. Grok API connection")
    print("2. Agent initialization")
    print("3. Simple command generation")
    print("4. Docker execution (if available)")

    # Run diagnostic
    results = runner.run_diagnostic_test()

    if not all(results["checks"].values()):
        print("\n⚠️  Some checks failed. You may still be able to run benchmarks.")
        response = input("\nContinue with a simple Terminal-Bench test? (y/n): ")
        if response.lower() != 'y':
            return 1

    # Try a simple Terminal-Bench task if available
    print("\n" + "=" * 60)
    print("RUNNING SIMPLE TERMINAL-BENCH TEST")
    print("=" * 60)

    print("\nAttempting to run a simple task from terminal-bench-core...")
    print("(This may take a minute to download the dataset on first run)\n")

    try:
        # Try to run just one simple task with minimal settings
        test_results = runner.run_with_tb_cli(
            dataset="terminal-bench-core==0.1.1",
            task_ids=None,  # Let TB choose
            n_concurrent=5,
            n_attempts=1,
            timeout_per_task=60  # 1 minute timeout for quick test
        )

        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)

        if test_results.get("status") == "completed":
            print("✅ Test completed successfully!")
            if "success_rate" in test_results:
                print(f"Success rate: {test_results['success_rate']*100:.1f}%")
        else:
            print(f"⚠️  Test status: {test_results.get('status')}")

        # Save test results
        test_file = Path(f"results/quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        test_file.parent.mkdir(exist_ok=True)
        with open(test_file, "w") as f:
            json.dump(test_results, f, indent=2)
        print(f"\nTest results saved to: {test_file}")

    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        print("\nThis might be because:")
        print("- Terminal-Bench is not properly installed")
        print("- Docker is not running")
        print("- The dataset needs to be downloaded")
        print("\nTry running: ./setup.sh --force")
        return 1

    return 0


def list_available_tasks() -> None:
    """List available tasks in Terminal-Bench (from the downloaded dataset)."""
    import re
    from pathlib import Path

    print("=" * 60)
    print("AVAILABLE TERMINAL-BENCH TASKS")
    print("=" * 60)

    dataset_spec = "terminal-bench-core==0.1.1"
    dataset_name, dataset_ver = dataset_spec.split("==", 1)

    # 1) Ensure dataset is downloaded and capture its location from CLI output
    try:
        dl = subprocess.run(
            ["tb", "datasets", "download", "--dataset", dataset_spec],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        print("❌ Terminal-Bench CLI (tb) not found. Install with: pip install terminal-bench")
        return
    except Exception as e:
        print(f"❌ Error invoking tb datasets download: {e}")
        return

    # Try to parse the dataset path from stdout/stderr
    m = re.search(r"Dataset location:\s*(.+)", dl.stdout) or re.search(r"Dataset location:\s*(.+)", dl.stderr)
    if m:
        dataset_path = Path(m.group(1).strip())
    else:
        # Fallback to default cache path
        dataset_path = Path.home() / ".cache" / "terminal-bench" / dataset_name / dataset_ver

    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return

    # 2) List tasks: in current TB builds, each task is a top-level directory under the dataset path
    try:
        entries = sorted(p.name for p in dataset_path.iterdir() if p.is_dir())
        # Prefer entries that look like real tasks (contain task.yaml), otherwise show all directories
        task_dirs = [name for name in entries if (dataset_path / name / "task.yaml").exists()]
        tasks = task_dirs if task_dirs else entries

        if not tasks:
            print(f"⚠️  No tasks found under: {dataset_path}")
            return

        print(f"📁 Dataset: {dataset_name}@{dataset_ver}")
        print(f"📂 Location: {dataset_path}")
        print(f"📝 {len(tasks)} tasks:\n")
        for t in tasks:
            print(f"- {t}")

        print("\nTip: run a single task with:")
        print(f"  tb run --dataset {dataset_spec} --task {tasks[0]}  # example")

    except Exception as e:
        print(f"❌ Error listing tasks: {e}")



def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Terminal-Bench with Grok (progress always on)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run diagnostic tests
  %(prog)s --diagnostic

  # Quick test with simple task
  %(prog)s --test

  # List available tasks
  %(prog)s --list-tasks

  # Run specific tasks
  %(prog)s --task-id task1 --task-id task2

  # Run full benchmark with parallel execution
  %(prog)s --n-concurrent 4 --n-attempts 2

  # Use a specific Grok model
  %(prog)s --model grok-4-fast-reasoning --test
        """
    )

    parser.add_argument("--model", help="Grok model to use (overrides .env)")
    parser.add_argument("--dataset", default="terminal-bench-core==0.1.1",
                        help="Dataset version (default: terminal-bench-core==0.1.1)")
    parser.add_argument("--task-id", action="append", dest="task_ids",
                        help="Specific task IDs to run (can be used multiple times)")
    parser.add_argument("--n-concurrent", type=int, default=5,
                        help="Number of concurrent runs (default: 5)")
    parser.add_argument("--n-attempts", type=int, default=1,
                        help="Number of attempts per task (default: 1)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout per task in seconds (default: 300)")
    parser.add_argument("--test", action="store_true",
                        help="Run quick test with diagnostic")
    parser.add_argument("--diagnostic", action="store_true",
                        help="Run diagnostic tests only")
    parser.add_argument("--list-tasks", action="store_true",
                        help="List available tasks in the dataset")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")

    args = parser.parse_args()

    # Set debug mode
    if args.debug:
        os.environ['GROK_DEBUG'] = 'true'

    # Initialize runner (progress always-on by default)
    runner = TerminalBenchRunner(model=args.model)

    # Handle different modes
    if args.diagnostic:
        results = runner.run_diagnostic_test()
        return 0 if all(results["checks"].values()) else 1

    if args.list_tasks:
        list_available_tasks()
        return 0

    if args.test:
        return run_quick_test(runner)

    # Run full benchmark
    print("=" * 60)
    print(f"RUNNING TERMINAL-BENCH WITH GROK")
    print("=" * 60)
    print(f"Model: {runner.model}")
    print(f"Dataset: {args.dataset}")
    if args.task_ids:
        print(f"Tasks: {', '.join(args.task_ids)}")
    else:
        print("Tasks: ALL (from dataset)")
    print(f"Concurrent runs: {args.n_concurrent}")
    print(f"Attempts per task: {args.n_attempts}")
    print(f"Timeout per task: {args.timeout}s")
    print("=" * 60)
    print("Live progress: ENABLED (tails progress.jsonl & commands.txt)\n")

    # Confirm before running full benchmark
    if not args.task_ids and args.n_concurrent > 1:
        print("\n⚠️  Running full benchmark with concurrency.")
        print("This may use significant API credits.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Benchmark cancelled.")
            return 0

    print("\nStarting benchmark...")

    try:
        results = runner.run_with_tb_cli(
            dataset=args.dataset,
            task_ids=args.task_ids,
            n_concurrent=args.n_concurrent,
            n_attempts=args.n_attempts,
            timeout_per_task=args.timeout
        )
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        print("Cleaning up Docker containers...")
        runner._cleanup_docker_containers()
        return 1

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    if results.get("status") == "completed":
        print(f"✅ Status: COMPLETED")
    else:
        print(f"⚠️  Status: {results.get('status', 'UNKNOWN')}")

    if "elapsed_time" in results:
        print(f"⏱️  Time: {results['elapsed_time']:.1f} seconds")

    if "success_rate" in results:
        print(f"📊 Success Rate: {results['success_rate']*100:.1f}%")

    if "num_tasks" in results:
        print(f"📝 Tasks Run: {results['num_tasks']}")

    if "output_dir" in results:
        print(f"📁 Results: {results['output_dir']}")

    # Analyze failures if available
    if results.get("benchmark_results") and isinstance(results["benchmark_results"], dict):
        tasks = results["benchmark_results"].get("tasks", [])
        failed_tasks = [t for t in tasks if t.get("status") != "success"]

        if failed_tasks:
            print(f"\n⚠️  {len(failed_tasks)} tasks failed:")
            for task in failed_tasks[:5]:  # Show first 5 failures
                print(f"  - {task.get('id', 'unknown')}: {task.get('error', 'no error message')}")
            if len(failed_tasks) > 5:
                print(f"  ... and {len(failed_tasks) - 5} more")

    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path(f"results/benchmark_summary_{timestamp}.json")
    summary_file.parent.mkdir(exist_ok=True)

    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Full results saved to: {summary_file}")

    # Provide next steps
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Review detailed results in the output directory")
    print("2. Analyze failure patterns to identify Grok's weaknesses")
    print("3. Compare with other models' performance on Terminal-Bench")
    print("4. Consider improvements to the benchmark based on findings")

    return 0


if __name__ == "__main__":
    sys.exit(main())
