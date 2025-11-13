from pathlib import Path

# Base project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"

# Example default sample JSON path pattern
# We will use this in the notebook to find one file to inspect.
EXPORT_GLOB_PATTERN = "export_shopper=*/0000_part_00.json"
