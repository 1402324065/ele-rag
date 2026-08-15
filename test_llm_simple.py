import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("MODEL_NAME")

print(f"? API Key: {api_key[:10]}...{api_key[-6:] if api_key else 'None'}")
print(f"? Base URL: {base_url}")
print(f"? Model: {model_name}")

try:
    print("\n? Testing connection with 10s timeout...")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=10.0)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Say 'Connection OK!' in Chinese"}],
        max_tokens=20,
        temperature=0.1,
    )
    print("\n? Success!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"\n? Error type: {type(e).__name__}")
    print(f"? Error message: {e}")
    
    import traceback
    print("\n? Full traceback:")
    traceback.print_exc()
