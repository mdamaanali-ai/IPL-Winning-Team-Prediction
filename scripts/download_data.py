from pathlib import Path
import sys
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DATA_URL, DATA_DIR, MATCHES_FILE

DATA_DIR.mkdir(parents=True, exist_ok=True)
print("Downloading IPL match dataset...")
r = requests.get(DATA_URL, timeout=60)
r.raise_for_status()
MATCHES_FILE.write_bytes(r.content)
print(f"Saved: {MATCHES_FILE}")
