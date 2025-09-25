#!/usr/bin/env python3
"""Main runner for Terminal-Bench with Grok evaluation"""
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add imports
from src.tb_runner import TerminalBenchRunner

def main():
    parser = argparse.ArgumentParser(description="Run Terminal-Bench with Grok")
    parser.add_argument("--model", help="Grok model to use (overrides .env)")
    parser.add_argument("--dataset", default="terminal-bench-core==0.1.1", 
                        help="Dataset to run")
    parser.add_argument("--task-id", action="append", dest="task_ids",
                        help="Specific task IDs to run (can be used multiple times)")
    parser.add_argument("--n-concurrent", type=int, default=1,
                        help="Number of concurrent runs")
    parser.add_argument("--n-attempts", type=int, default=1,
                        help="Number of attempts per task")
    parser.add_argument("--test", action="store_true",
                        help="Run connection test and simple Docker test")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = TerminalBenchRunner(model=args.model)
    
    # Test mode
    if args.test:
        print("=" * 60)
        print("TESTING GROK CONNECTION AND DOCKER")
        print("=" * 60)
        
        # Test API connection
        print("\n1. Testing Grok API connection...")
        if runner.test_connection():
            print("   ✓ API connection successful")
        else:
            print("   ✗ API connection failed - check your XAI_API_KEY in .env")
            return 1
        
        # Test Docker
        print("\n2. Testing Docker setup...")
        docker_result = runner.run_simple_docker_test()
        if docker_result['success']:
            print(f"   ✓ Docker test passed")
            print(f"   Command used: {docker_result['command']}")
        else:
            print(f"   ✗ Docker test failed")
            print(f"   Output: {docker_result['output']}")
        
        print("\n" + "=" * 60)
        return 0
    
    # Run Terminal-Bench
    print("=" * 60)
    print(f"RUNNING TERMINAL-BENCH WITH GROK")
    print(f"Model: {runner.model}")
    print(f"Dataset: {args.dataset}")
    if args.task_ids:
        print(f"Tasks: {', '.join(args.task_ids)}")
    print(f"Concurrent runs: {args.n_concurrent}")
    print(f"Attempts per task: {args.n_attempts}")
    print("=" * 60 + "\n")
    
    print("Starting benchmark... (Live output from Terminal-Bench CLI below)")
    try:
        results = runner.run_with_tb_cli(
            dataset=args.dataset,
            task_ids=args.task_ids,
            n_concurrent=args.n_concurrent,
            n_attempts=args.n_attempts
        )
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user. Cleaning up...")
        # Optional: Add cleanup for running Docker containers
        subprocess.run("docker ps -q | xargs -r docker stop", shell=True, capture_output=True)
        subprocess.run("docker ps -a -q | xargs -r docker rm", shell=True, capture_output=True)
        print("Docker containers stopped and removed.")
        return 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    if isinstance(results, dict):
        print(json.dumps(results, indent=2)[:500])  # First 500 chars
        
        # Save full results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(f"results/summary_{timestamp}.json")
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nFull results saved to: {output_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())            