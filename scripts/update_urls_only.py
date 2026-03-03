# update_urls_only.py
from pinecone import Pinecone
import json, os

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX"])
namespace = os.environ.get("PINECONE_NAMESPACE", "default")

count = 0
with open("data/processed/chunks.jsonl") as f:
    for line in f:
        obj = json.loads(line)
        meta = obj.get("metadata", {})
        index.update(
            id=obj["id"],
            set_metadata=meta,
            namespace=namespace
        )
        count += 1
        if count % 100 == 0:
            print(f"Updated {count} chunks...")

print(f"✅ Completed: updated metadata (URLs) for {count} chunks in namespace '{namespace}'.")
