# scripts/xai_client.py
import os
import time
from typing import Dict, List, Optional
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

class XAIClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def complete(self, 
                 prompt: str, 
                 model: str = "grok-3",
                 max_tokens: int = 1000,
                 temperature: float = 0.7) -> Dict:
        """Make a completion request to Grok"""
        
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(
            f"{self.base_url}/completions",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        
        return response.json()
    
    def test_connection(self):
        """Test API connection with a simple request"""
        try:
            result = self.complete(
                "Hello, can you confirm you're working?",
                max_tokens=50
            )
            print("✅ API connection successful!")
            print(f"Response: {result.get('choices', [{}])[0].get('text', 'No response')}")
            return True
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            return False