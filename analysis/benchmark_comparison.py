# analysis/benchmark_comparison.py
import json
import os
from pathlib import Path

def analyze_benchmark(bench_path: str, name: str):
    """Analyze benchmark structure and characteristics"""
    
    analysis = {
        "name": name,
        "total_examples": 0,
        "categories": set(),
        "avg_prompt_length": 0,
        "file_structure": [],
        "evaluation_metrics": [],
        "complexity_indicators": []
    }
    
    # Find test files
    test_files = list(Path(bench_path).rglob("*.json")) + \
                 list(Path(bench_path).rglob("*.jsonl"))
    
    print(f"\n=== {name} ===")
    print(f"Found {len(test_files)} test files")
    
    # Sample analysis of test structure
    for file in test_files[:3]:  # Look at first 3 files
        with open(file, 'r') as f:
            if file.suffix == '.jsonl':
                lines = f.readlines()
                print(f"\nFile: {file.name}")
                print(f"  - {len(lines)} test cases")
                if lines:
                    sample = json.loads(lines[0])
                    print(f"  - Keys: {list(sample.keys())}")
                    if 'prompt' in sample or 'question' in sample:
                        prompt_key = 'prompt' if 'prompt' in sample else 'question'
                        print(f"  - Sample prompt length: {len(sample[prompt_key])} chars")
            
    return analysis

# Analyze each benchmark
benchmarks = [
    ("analysis/benchmark_exploration/tau-bench", "tau-bench"),
    ("analysis/benchmark_exploration/tau2-bench", "tau2-bench"),
    ("analysis/benchmark_exploration/merlin-bench", "merlin-bench")
]

for path, name in benchmarks:
    if os.path.exists(path):
        analyze_benchmark(path, name)