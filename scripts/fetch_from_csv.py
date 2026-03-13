from __future__ import annotations
import csv, os, re, time, pathlib, sys
from urllib.parse import urlparse
import requests

RAW_DIR = pathlib.Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def safe_name(name: str) -> str:
    name = re.sub(r"[^\w\-\. ]+", "_", name.strip())
    return re.sub(r"\s+", " ", name).strip()

def filename_for(url: str, title: str | None = None) -> pathlib.Path:
    p = urlparse(url)
    ext = ".html" if not os.path.splitext(p.path)[1] else os.path.splitext(p.path)[1]
    base = safe_name(title or pathlib.Path(p.path).stem or "doc")[:120]
    if not base:
        base = "doc"
    return RAW_DIR / f"{base}{ext}"

def fetch(url: str, out: pathlib.Path, timeout=30):
    headers = {"User-Agent": "CBV2-fetcher/1.0"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    out.write_bytes(r.content)

def main(csv_path="sources.csv"):
    path = pathlib.Path(csv_path)
    if not path.exists():
        print(f"ERROR: {csv_path} not found.")
        sys.exit(1)

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    ok = 0
    for i, row in enumerate(rows, 1):
        url = (row.get("url") or row.get("URL") or "").strip()
        title = (row.get("title") or row.get("Title") or "").strip()
        if not url:
            print(f"[{i}/{len(rows)}] SKIP (no URL)"); continue
        out = filename_for(url, title=title)
        if out.exists():
            print(f"[{i}/{len(rows)}] EXISTS  {out.name}"); continue
        try:
            fetch(url, out)
            print(f"[{i}/{len(rows)}] OK      {out.name}")
            ok += 1
            time.sleep(0.2)  # be gentle
        except Exception as e:
            print(f"[{i}/{len(rows)}] FAIL    {url}  -> {e}")
    print(f"[OK] Done. Downloaded {ok} files into {RAW_DIR}")

if __name__ == "__main__":
    # allow: PYTHONPATH=. python scripts/fetch_from_csv.py sources.csv
    main(sys.argv[1] if len(sys.argv) > 1 else "sources.csv")
