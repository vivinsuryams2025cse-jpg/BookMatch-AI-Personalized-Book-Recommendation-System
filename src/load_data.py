"""
BookMatch AI: Personalized Book Recommendation System
Day 2: Dataset Collection & Loading

This module handles:
1. Verifying and loading the book dataset (Goodbooks-10k Extended).
2. Automatic download capability if the dataset is not found locally.
3. Schema validation and data integrity checks for recommendation features.
4. Summary diagnostics and sample preview.
"""

import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional, List
import pandas as pd

# Safe terminal encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Default paths and dataset source
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "books.csv"
DATASET_SOURCE_URL = (
    "https://raw.githubusercontent.com/malcolmosh/goodbooks-10k-extended/master/books_enriched.csv"
)

# Essential columns required for content-based book recommendations
RECOMMENDATION_COLUMNS: List[str] = [
    "book_id",
    "title",
    "authors",
    "genres",
    "description",
    "average_rating",
]


def download_dataset(
    url: str = DATASET_SOURCE_URL,
    destination: Path = DEFAULT_DATASET_PATH,
) -> Path:
    """
    Downloads the dataset from the remote source if it is not present locally.

    Args:
        url: Remote URL of the raw CSV file.
        destination: Local Path where the file should be saved.

    Returns:
        Path to the downloaded file.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Downloading dataset from remote source...")
    print(f"    Source URL: {url}")
    print(f"    Destination: {destination}")

    try:
        # Stream download with user agent header to prevent blocking
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "BookMatch-AI-DataLoader/1.0"},
        )
        with urllib.request.urlopen(req) as response, open(destination, "wb") as out_file:
            total_size = response.headers.get("Content-Length")
            total_bytes = int(total_size) if total_size else None
            downloaded = 0
            chunk_size = 1024 * 64  # 64 KB chunks

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = (downloaded / total_bytes) * 100
                    print(
                        f"\r    Progress: {downloaded / (1024 * 1024):.2f} MB / "
                        f"{total_bytes / (1024 * 1024):.2f} MB ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )

        print("\n[+] Download completed successfully!")
        return destination
    except Exception as e:
        if destination.exists():
            destination.unlink()  # Clean up partial download
        raise RuntimeError(f"Failed to download dataset: {e}") from e


def load_dataset(
    filepath: Optional[str | Path] = None,
    auto_download: bool = True,
) -> pd.DataFrame:
    """
    Loads the book dataset from CSV into a Pandas DataFrame.
    If the file is not found and auto_download is True, downloads it first.

    Args:
        filepath: Optional path to the CSV file. Defaults to data/books.csv.
        auto_download: If True, automatically downloads the file if missing.

    Returns:
        pd.DataFrame containing the raw book dataset.
    """
    path = Path(filepath) if filepath else DEFAULT_DATASET_PATH

    if not path.exists():
        if auto_download:
            print(f"[!] Dataset not found at: {path}")
            download_dataset(DATASET_SOURCE_URL, path)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {path}. Set auto_download=True or place books.csv in data/"
            )

    print(f"[*] Loading dataset from: {path.name} ({path.stat().st_size / (1024 * 1024):.2f} MB)...")
    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"[+] Dataset successfully loaded into DataFrame! Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        return df
    except Exception as e:
        raise RuntimeError(f"Error reading CSV file at {path}: {e}") from e


def validate_schema(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> bool:
    """
    Validates that the DataFrame contains the necessary columns for the recommendation pipeline.

    Args:
        df: Loaded Pandas DataFrame.
        required_columns: List of column names that must exist.

    Returns:
        True if all required columns are present, False otherwise.
    """
    cols = required_columns or RECOMMENDATION_COLUMNS
    missing_cols = [col for col in cols if col not in df.columns]

    if missing_cols:
        print(f"[!] Schema Validation Warning: Missing columns: {missing_cols}")
        return False

    print(f"[+] Schema Validation Passed: All {len(cols)} core recommendation columns are present.")
    return True


def inspect_dataset(df: pd.DataFrame) -> None:
    """
    Prints a formatted statistical and structural overview of the loaded dataset.

    Args:
        df: The loaded pandas DataFrame.
    """
    print("\n" + "=" * 70)
    print("  BOOKMATCH AI - DATASET INSPECTION & DIAGNOSTICS")
    print("=" * 70)

    # 1. Basic Shape and Memory
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f" Total Books (Rows)    : {df.shape[0]:,}")
    print(f" Total Features (Cols) : {df.shape[1]}")
    print(f" Memory Usage in RAM   : {memory_mb:.2f} MB")
    print("-" * 70)

    # 2. Key Recommendation Columns Status
    print(" Recommendation Features Summary:")
    for col in RECOMMENDATION_COLUMNS:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            dtype = df[col].dtype
            print(
                f"   * {col:<18} | Type: {str(dtype):<8} | "
                f"Missing: {null_count:>5,} ({null_pct:>5.2f}%)"
            )
        else:
            print(f"   * {col:<18} | [MISSING COLUMN]")
    print("-" * 70)

    # 3. Sample Preview of Top 3 Books
    print(" Sample Preview (First 3 Books):")
    cols_to_preview = [c for c in ["book_id", "title", "authors", "average_rating", "genres"] if c in df.columns]
    for idx, row in df.head(3).iterrows():
        print(f"\n   [{idx + 1}] ID: {row.get('book_id', 'N/A')} | Rating: {row.get('average_rating', 'N/A')}/5.0")
        print(f"       Title  : {row.get('title', 'N/A')}")
        print(f"       Authors: {row.get('authors', 'N/A')}")
        print(f"       Genres : {row.get('genres', 'N/A')}")
        desc = str(row.get('description', '')).replace('\n', ' ')
        preview_desc = (desc[:100] + "...") if len(desc) > 100 else desc
        print(f"       Summary: {preview_desc}")

    print("\n" + "=" * 70)
    print(" [SUCCESS] Day 2 Task Complete: Dataset loaded and ready for Day 3 (EDA)!")
    print("=" * 70 + "\n")


def main():
    """
    Main entry point for Day 2 task execution.
    """
    print("=" * 70)
    print("  BookMatch AI: Day 2 - Dataset Collection & Loading")
    print("=" * 70)

    # Load dataset (downloads automatically if missing)
    df = load_dataset()

    # Validate essential schema
    validate_schema(df)

    # Display inspection and diagnostics
    inspect_dataset(df)


if __name__ == "__main__":
    main()
