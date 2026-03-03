
from __future__ import annotations
import json, pathlib
from typing import Dict, Any, Iterator

def jsonl_iter(path: str) -> Iterator[Dict[str, Any]]:
    p = pathlib.Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(path: str, rows):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
