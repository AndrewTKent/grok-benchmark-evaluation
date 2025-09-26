#!/usr/bin/env python3
"""Quick test of Grok connection"""
import os
from dotenv import load_dotenv
load_dotenv()

from src.grok_client import GrokClient

client = GrokClient()
if client.test_connection():
    print("✅ Grok API connection successful!")
else:
    print("❌ Grok API connection failed. Check your .env file.")
