import re
import unicodedata
from typing import Optional


def normalize_text(text: Optional[str]) -> str:
    """
    Normalize product text for joining and modeling.

    Steps:
      - Handle missing values.
      - Strip leading/trailing whitespace.
      - Lowercase.
      - Unicode NFKD normalization.
      - Remove accents.
      - Replace non [a-z0-9] characters with space.
      - Collapse multiple spaces into one.
    """
    if text is None:
        return ""

    # Convert to string and strip outer whitespace
    text = str(text).strip()

    if not text:
        return ""

    # Unicode normalize and remove accents
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    # Lowercase
    text = text.lower()

    # Replace non [a-z0-9] with space
    text = re.sub(r"[^a-z0-9]+", " ", text)

    # Collapse multiple spaces and strip again
    text = re.sub(r"\s+", " ", text).strip()

    return text
