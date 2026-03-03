
from __future__ import annotations
import tiktoken

_ENCODER = None

def get_encoder(name: str = "cl100k_base"):
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding(name)
    return _ENCODER

def count_tokens(text: str) -> int:
    return len(get_encoder().encode(text or ""))
