# original_benchmark/run_original.py
import json
import time
from pathlib import Path
from typing import List, Dict
import sys
sys.path.append('..')
from scripts.xai_client import XAIClient
from tqdm import tqdm

class BenchmarkRunner:
    def __init__(self, benchmark_name: str = "tau-bench"):
        self.benchmark_name = benchmark_name
        self.client = XAIClient()
        self.results = []
        
    def load_test_cases(self, subset_size: int = 10) -> List[Dict]:
        """Load a subset of test cases for initial testing"""
        test_file = f"analysis/benchmark_exploration/{self.benchmark_name}/data/test.jsonl"
        
        test_cases = []
        with open(test_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= subset_size:
                    break
                test_cases.append(json.loads(line))
        
        return test_cases
    
    def run_single_test(self, test_case: Dict) -> Dict:
        """Run a single test case"""
        start_time = time.time()
        
        # Extract prompt (adjust key based on benchmark format)
        prompt = test_case.get('prompt', test_case.get('question', ''))
        
        # Get Grok response
        response = self.client.complete(
            prompt=prompt,
            model="grok-3",
            max_tokens=500,
            temperature=0.1  # Lower temperature for consistency
        )
        
        elapsed_time = time.time() - start_time
        
        return {
            "test_id": test_case.get('id', 'unknown'),
            "prompt": prompt,
            "expected": test_case.get('answer', test_case.get('target', '')),
            "grok_response": response['choices'][0]['text'],
            "latency": elapsed_time,
            "tokens_used": response.get('usage', {})
        }
    
    def run_subset(self, n: int = 10):
        """Run a small subset for initial testing"""
        print(f"Loading {n} test cases from {self.benchmark_name}...")
        test_cases = self.load_test_cases(n)
        
        print(f"Running {len(test_cases)} tests...")
        for test in tqdm(test_cases):
            try:
                result = self.run_single_test(test)
                self.results.append(result)
                
                # Save intermediate results
                with open(f'results/initial_run_{self.benchmark_name}.jsonl', 'a') as f:
                    f.write(json.dumps(result) + '\n')
                    
            except Exception as e:
                print(f"Error on test {test.get('id')}: {e}")
                
        return self.results

# Run initial test
if __name__ == "__main__":
    runner = BenchmarkRunner("tau-bench")
    results = runner.run_subset(5)  # Start with just 5 for testing
    
    print("\n=== Initial Results ===")
    for r in results:
        print(f"Test ID: {r['test_id']}")
        print(f"Latency: {r['latency']:.2f}s")
        print(f"Response preview: {r['grok_response'][:100]}...")
        print("-" * 40)