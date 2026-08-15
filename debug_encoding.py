import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.document_loader import load_single_document

print("=== Checking data/raw/ files ===")
data_raw_dir = PROJECT_ROOT / "data" / "raw"
for file_path in data_raw_dir.rglob("*"):
    if not file_path.is_file():
        continue
    print(f"\n? Checking: {file_path.name}")
    try:
        text = load_single_document(file_path)
        print(f"? Loaded successfully! Length: {len(text)}")
        print(f"First 100 chars: {text[:100]}...")
    except Exception as e:
        print(f"? Failed! {type(e).__name__}: {e}")

print("\n=== Checking data/processed/ files ===")
data_processed_dir = PROJECT_ROOT / "data" / "processed"
for file_path in data_processed_dir.rglob("*"):
    if not file_path.is_file():
        continue
    print(f"\n? Checking: {file_path.name}")
    try:
        text = file_path.read_text(encoding="utf-8")
        print(f"? Loaded successfully! Length: {len(text)}")
    except Exception as e:
        print(f"? Failed! {type(e).__name__}: {e}")
        try:
            text_gbk = file_path.read_text(encoding="gbk")
            print(f"   GBK fallback worked! Length: {len(text_gbk)}")
        except:
            pass
