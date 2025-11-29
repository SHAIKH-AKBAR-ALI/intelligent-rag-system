# WHAT THIS MODULE DOES:
# - Small text helpers.

import re
import hashlib

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
