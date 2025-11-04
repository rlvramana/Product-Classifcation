"""
Purpose: one place to clean product names consistently across the project.

We keep informative words (brand, color, garment) and remove noise
(emojis, SKUs, pack sizes, punctuation). We will classify product names only.
"""

import re
from typing import Optional

# Matches most emoji code points so we can drop them.

_EMOJI = re.compile(r"[\U0001F100-\U0001FFFF]")


def normalize_text(s: Optional[str]) -> str:
    """Normalize a product name string into a clean, comparable form."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)

    # 1) tidy and lowercase
    s = s.strip().lower()

    # 2) remove emojis
    s = _EMOJI.sub(" ", s)

    # 3) strip SKU/ID-like tokens (letters+digits 6+ chars, asin patterns)
    s = re.sub(r"\b([a-z]*\d[a-z\d]{5,}|asin: ?[a-z\d]{6,})\b", " ", s)

    # 4) drop quantities like 10oz, 2 lb, 3 pack, pk 3, 2pcs
    s = re.sub(r"\b(\d+\s?(oz|fl oz|lb|g|ml|ct|inch|in|cm|mm))\b", " ", s)
    s = re.sub(r"\b(pack of|pk|pcs?|count|ct)\s*\d+\b", " ", s)

    # 5) collapse punctuation to spaces, then collapse extra spaces
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
