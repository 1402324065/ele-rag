import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

print("=" * 60)
print("LLM Connection Diagnostics")
print("=" * 60)

# 1. Check environment variables
print("\n1. Checking environment variables:")
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("MODEL_NAME")

print(f"   OPENAI_API_KEY: {'Configured' if api_key else 'Not configured'}")
print(f"   OPENAI_BASE_URL: {base_url}")
print(f"   MODEL_NAME: {model_name}")

if not api_key or api_key == "your_api_key_here":
    print("   ERROR: API Key not configured or using default value")
    exit(1)

print("   Environment variables OK")

# 2. Test connection
print("\n2. Testing DeepSeek API connection...")
try:
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a test assistant. Just reply with 'Connection OK'."},
            {"role": "user", "content": "Test"}
        ],
        temperature=0.7,
        max_tokens=100
    )
    print(f"   OK! Model replied: {response.choices[0].message.content}")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")
    print("\nPossible reasons:")
    print("   1. Network issue - Check your internet connection")
    print("   2. Invalid API Key - Verify your DeepSeek API key")
    print("   3. Proxy issue - Disable proxy or configure it properly")
    print("   4. API Key expired or no balance - Check DeepSeek console")
    exit(1)

print("\n" + "=" * 60)
print("Diagnostics complete! LLM connection is working")
print("=" * 60)
