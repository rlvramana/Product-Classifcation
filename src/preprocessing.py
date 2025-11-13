from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


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

    text = str(text).strip()
    if not text:
        return ""

    # Unicode normalize and remove accents
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_json_file_products(path: Path, chunk_size: int = 100_000) -> pd.DataFrame:
    """
    Load one JSON file containing shopper events and return a DataFrame
    with one row per event that has usable product text.

    Returned columns (when present in the source):
      - event_id
      - event_type
      - start_time_local
      - remove_amazon
      - month              (from the parent folder name)
      - product_text_raw   (copy of remove_amazon)
      - product_text_norm  (normalized text)
    """
    cols_keep = ["event_id", "event_type", "start_time_local", "remove_amazon"]
    dfs = []
    month_name = path.parent.name

    json_iter = pd.read_json(
        path,
        lines=True,
        chunksize=chunk_size,
    )

    for chunk in json_iter:
        existing_cols = [c for c in cols_keep if c in chunk.columns]
        df = chunk[existing_cols].copy()

        df["month"] = month_name
        df["product_text_raw"] = df.get("remove_amazon")
        df["product_text_norm"] = df["product_text_raw"].apply(normalize_text)

        # Keep only rows with non-empty normalized text
        df = df[df["product_text_norm"] != ""]
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(
            columns=[
                "event_id",
                "event_type",
                "start_time_local",
                "remove_amazon",
                "month",
                "product_text_raw",
                "product_text_norm",
            ]
        )

    return pd.concat(dfs, ignore_index=True)


def build_products_from_files(
    file_paths: Iterable[Path],
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """
    Load multiple JSON files and combine them into a single products table
    with usable product text, using load_json_file_products.
    """
    dfs = []
    for path in file_paths:
        df = load_json_file_products(Path(path), chunk_size=chunk_size)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def join_products_with_labels(
    products_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join products with labels on product_text_norm.

    products_df must contain:
      - product_text_norm
      - product_text_raw
    labels_df must contain:
      - product_text_norm
      - label
      - label_raw
    """
    required_prod_cols = {"product_text_norm", "product_text_raw"}
    required_lab_cols = {"product_text_norm", "label", "label_raw"}

    missing_prod = required_prod_cols - set(products_df.columns)
    missing_lab = required_lab_cols - set(labels_df.columns)

    if missing_prod:
        raise KeyError(f"products_df is missing columns: {missing_prod}")
    if missing_lab:
        raise KeyError(f"labels_df is missing columns: {missing_lab}")

    joined = products_df.merge(
        labels_df[list(required_lab_cols)],
        on="product_text_norm",
        how="inner",
    )

    return joined


def to_product_level(joined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert an event-level joined table into a product-level table.

    Steps:
      - Drop any product_text_norm values that have both labels 0 and 1.
      - For remaining texts, keep a single representative row per product_text_norm.
    """
    if "product_text_norm" not in joined_df.columns or "label" not in joined_df.columns:
        raise KeyError(
            "joined_df must contain 'product_text_norm' and 'label' columns."
        )

    # Find conflicts: same normalized text with both labels 0 and 1
    label_counts = joined_df.groupby("product_text_norm")["label"].nunique()
    conflict_texts = label_counts[label_counts > 1].index

    # Drop conflicting products entirely
    no_conflicts = joined_df[
        ~joined_df["product_text_norm"].isin(conflict_texts)
    ].copy()

    # Deduplicate: keep one row per product_text_norm.
    # Sort so the choice is deterministic.
    no_conflicts = no_conflicts.sort_values(
        ["product_text_norm", "start_time_local"],
        na_position="last",
    )

    product_level = no_conflicts.drop_duplicates(
        subset=["product_text_norm"],
        keep="first",
    ).copy()

    return product_level
