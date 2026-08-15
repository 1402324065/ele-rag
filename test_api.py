import requests


def test_api():
    base_url = "http://127.0.0.1:8000"

    print("=" * 60)
    print("Testing Power Knowledge Assistant API...")
    print("=" * 60)

    print("\n1. Testing /health endpoint:")
    resp = requests.get(f"{base_url}/health", timeout=10)
    print(f"   Status: {resp.status_code}")
    print(f"   Response: {resp.json()}")

    print("\n2. Testing /api/v1/chat/test endpoint:")
    resp = requests.get(f"{base_url}/api/v1/chat/test", timeout=10)
    print(f"   Status: {resp.status_code}")
    print(f"   Response: {resp.json()}")

    print("\n3. Testing /api/v1/chat POST endpoint (query transformer inspection):")
    chat_resp = requests.post(
        f"{base_url}/api/v1/chat",
        json={"query": "Transformer inspection points"},
        timeout=10,
    )
    print(f"   Status: {chat_resp.status_code}")
    result = chat_resp.json()
    print(f"\nAnswer:\n{result.get('answer', '(no answer)')}")
    print("\nSources:")
    for i, s in enumerate(result.get("sources", []), 1):
        print(f"\n{i}. Score: {s.get('score', 'N/A')}")
        print(f"   Source: {s.get('source', 'N/A')}")
        print(f"   Text:\n{s.get('text', '(no text)')}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_api()
