# src/grok_client.py
import os
import json
import requests
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import time

load_dotenv()

class GrokClient:
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None
    ):
        """
        Initialize Grok client with configuration from .env file.
        
        Environment variables:
        - XAI_API_KEY or GROK_API_KEY: API key for authentication
        - GROK_MODEL: Model name (e.g., grok-2-1212, grok-beta)
        - GROK_BASE_URL: Base API URL
        - GROK_TIMEOUT: Request timeout in seconds
        - GROK_MAX_RETRIES: Maximum number of retry attempts
        - GROK_DEFAULT_TEMPERATURE: Default temperature for generation
        - GROK_DEFAULT_MAX_TOKENS: Default max tokens for generation
        """
        # Load configuration from environment with defaults
        self.api_key = (
            api_key or 
            os.getenv('XAI_API_KEY') or 
            os.getenv('GROK_API_KEY')
        )
        if not self.api_key:
            raise ValueError(
                "API key not found. Set XAI_API_KEY or GROK_API_KEY in .env file"
            )
            
        self.model = model or os.getenv('GROK_MODEL', 'grok-2-1212')
        self.base_url = base_url or os.getenv('GROK_BASE_URL', 'https://api.x.ai/v1')
        self.timeout = timeout or int(os.getenv('GROK_TIMEOUT', '60'))
        self.max_retries = max_retries or int(os.getenv('GROK_MAX_RETRIES', '3'))
        
        # Default generation parameters
        self.default_temperature = float(os.getenv('GROK_DEFAULT_TEMPERATURE', '0.7'))
        self.default_max_tokens = int(os.getenv('GROK_DEFAULT_MAX_TOKENS', '1000'))
        
        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Log configuration (useful for debugging)
        if os.getenv('GROK_DEBUG', 'false').lower() == 'true':
            self._log_config()
    
    def _log_config(self):
        """Log configuration for debugging"""
        print(f"Grok Client Configuration:")
        print(f"  Model: {self.model}")
        print(f"  Base URL: {self.base_url}")
        print(f"  Timeout: {self.timeout}s")
        print(f"  Max Retries: {self.max_retries}")
        print(f"  API Key: {'Set' if self.api_key else 'Not Set'}")
        print(f"  Default Temperature: {self.default_temperature}")
        print(f"  Default Max Tokens: {self.default_max_tokens}")
    
    def chat_completion(
        self, 
        messages: List[Dict], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        OpenAI-compatible chat completion interface.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            API response dictionary
        """
        # Build request payload
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.default_temperature),
            "max_tokens": kwargs.get("max_tokens", self.default_max_tokens),
        }
        
        # Add optional parameters if provided
        optional_params = ['top_p', 'n', 'stream', 'stop', 'frequency_penalty', 'presence_penalty']
        for param in optional_params:
            if param in kwargs:
                payload[param] = kwargs[param]
        
        # Retry logic
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                # Log raw response for debugging if needed
                if os.getenv('GROK_DEBUG', 'false').lower() == 'true':
                    print(f"Response status: {response.status_code}")
                    if response.status_code != 200:
                        print(f"Response body: {response.text}")
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Request timeout. Retrying in {wait_time}s... (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                continue
                
            except requests.exceptions.HTTPError as e:
                last_exception = e
                if response.status_code == 429:  # Rate limit
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** (attempt + 1)
                        print(f"Rate limited. Waiting {wait_time}s... (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(wait_time)
                        continue
                elif response.status_code == 400:
                    # Bad request - log details for debugging
                    print(f"Bad request error. Response: {response.text}")
                    print(f"Request payload: {json.dumps(payload, indent=2)}")
                raise
                
            except Exception as e:
                last_exception = e
                print(f"Unexpected error in chat_completion: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        
        # If we get here, all retries failed
        raise last_exception or Exception("All retry attempts failed")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Simple text generation interface.
        
        Args:
            prompt: The prompt text
            **kwargs: Additional parameters for generation
        
        Returns:
            Generated text string
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.chat_completion(messages, **kwargs)
        
        # Extract text from response
        if 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['message']['content']
        else:
            raise ValueError(f"Unexpected response format: {response}")
    
    def generate_with_system(
        self, 
        prompt: str, 
        system_message: str = None,
        **kwargs
    ) -> str:
        """
        Generate with optional system message.
        
        Args:
            prompt: User prompt
            system_message: System message for context
            **kwargs: Additional parameters
        
        Returns:
            Generated text
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        response = self.chat_completion(messages, **kwargs)
        return response['choices'][0]['message']['content']
    
    def test_connection(self) -> bool:
        """
        Test if the API connection is working.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.generate(
                "Say 'Hello' and nothing else.", 
                temperature=0.0,
                max_tokens=10
            )
            print(f"Connection test successful. Response: {response}")
            return True
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False