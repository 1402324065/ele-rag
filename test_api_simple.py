import json
import urllib.request
import urllib.parse


def http_get(url: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post(url: str, data: dict, timeout: int = 10) -> dict:
    json_data = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=json_data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_api():
    base_url = "http://127.0.0.1:8000"

    print("=" * 60)
    print("Testing Power Knowledge Assistant API...")
    print("=" * 60)

    print("\n1. Testing /health endpoint:")
    resp = http_get(f"{base_url}/health")
    print(f"   Response: {resp}")

    print("\n2. Testing /api/v1/chat/test endpoint:")
    resp = http_get(f"{base_url}/api/v1/chat/test")
    print(f"   Response: {resp}")

    print("\n3. Testing /api/v1/chat POST endpoint:")
    chat_resp = http_post(
        f"{base_url}/api/v1/chat",
        {"query": "Transformer inspection points"},
    )

    print(f"\nAnswer:\n{chat_resp.get('answer', '(no answer)')}")
    print("\nSources:")
    for i, s in enumerate(chat_resp.get("sources", []), 1):
        print(f"\n{i}. Score: {s.get('score', 'N/A')}")
        print(f"   Source: {s.get('source', 'N/A')}")
        print(f"   Text:\n{s.get('text', '(no text)')}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("You can also open http://127.0.0.1:8000/docs in your browser for the interactive Swagger UI!")
    print("=" * 60)


if __name__ == "__main__":
    test_api()
