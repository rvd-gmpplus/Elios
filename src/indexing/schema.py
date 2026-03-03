
from __future__ import annotations
from typing import Dict, Any

def embed_payload(chunk_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder: in Step 3 we'll add dense + sparse payload building.
    """
    return {
        "text": chunk_text,
        "metadata": metadata,
    }
