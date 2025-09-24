# scripts/test_api.py
from xai_client import XAIClient
import json

def test_api_functionality():
    client = XAIClient()
    
    # Test 1: Basic connection
    print("Testing basic connection...")
    if not client.test_connection():
        return False
    
    # Test 2: Response format
    print("\nTesting response format...")
    response = client.complete(
        "What is 2+2?",
        model="grok-3",
        max_tokens=10
    )
    
    print(f"Full response structure:")
    print(json.dumps(response, indent=2))
    
    # Test 3: Measure latency
    print("\nMeasuring API latency...")
    import time
    times = []
    for i in range(3):
        start = time.time()
        client.complete("Hi", max_tokens=5)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Request {i+1}: {elapsed:.2f}s")
    
    print(f"Average latency: {sum(times)/len(times):.2f}s")
    
    return True

if __name__ == "__main__":
    test_api_functionality()