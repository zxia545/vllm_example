#!/usr/bin/env python3

import sys
import subprocess
import time
import requests
from openai import OpenAI

def test_vllm_setup():
    print("Testing vLLM setup...")
    
    # Start vLLM server with Qwen2.5-0.5B model
    print("Starting vLLM server...")
    process = subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "Qwen/Qwen2.5-0.5B-Instruct",
        "--port", "8000",
        "--host", "0.0.0.0"
    ])
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(30)
    
    try:
        # Test API connection
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
        
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            messages=[
                {"role": "user", "content": "Hello! Please respond with 'vLLM is working!'"}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        print(f"✅ vLLM is working! Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Clean up
        process.terminate()
        process.wait()
        print("Server stopped.")

if __name__ == "__main__":
    test_vllm_setup() 