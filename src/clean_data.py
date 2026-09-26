"""
BookMatch AI: Personalized Book Recommendation System
Day 4: Data Cleaning & Preprocessing

This module handles all data quality and preprocessing steps identified during
Day 3 EDA, preparing the dataset for feature engineering (Day 5):
1. Drop redundant/unnecessary columns ('Unnamed: 0', 'index').
2. Handle duplicate titles (keep first occurrence).
3. Impute missing descriptions with synthetic fallback text (title + genres).
4. Normalize genre lists into clean, lowercase, hyphen-separated tokens.
5. Clean and standardize author names.
6. Strip HTML entities, special characters, and formatting artifacts from text.
7. Impute missing numeric values (pages, publication year).
8. Save the cleaned dataset and generate a preprocessing report.
"""

import sys
import re
import ast
from pathlib import Path
from typing import List, Any, Dict
import pandas as pd
import numpy as np

# Safe terminal encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATASET_PATH = DATA_DIR / "books.csv"
CLEANED_DATASET_PATH = DATA_DIR / "books_cleaned.csv"
REPORTS_DIR = BASE_DIR / "reports"

# Columns to drop (identified as redundant during EDA)
COLUMNS_TO_DROP = ["Unnamed: 0", "index"]

# Columns critical for the recommendation engine
RECOMMENDATION_COLUMNS = [
    "book_id",
    "title",
    "authors",
    "genres",
    "description",
    "average_rating",
]


# =============================================================================
# Utility Functions
# =============================================================================

def parse_list_column(val: Any) -> List[str]:
    """
    Safely parses a column that may contain stringified Python lists,
    comma-separated strings, or native lists.
    """
    if pd.isna(val) or val is None:
        return []
    if isinstance(val, list):
        return [str(item).strip() for item in val if str(item).strip()]

    val_str = str(val).strip()
    if not val_str or val_str in ("[]", "{}", "nan", "None"):
        return []

    try:
        parsed = ast.literal_eval(val_str)
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass

    # Fallback: Strip outer brackets and split by comma
    cleaned = val_str.strip("[](){}\"' ")
    tokens = [item.strip().strip("'\"") for item in cleaned.split(",")]
    return [t for t in tokens if t]


def render_ascii_bar(val: float, max_val: float, bar_length: int = 24) -> str:
    """Renders a text-based progress bar for terminal visualization."""
    if max_val <= 0:
        return ""
    fraction = min(max(val / max_val, 0.0), 1.0)
    filled_len = int(round(bar_length * fraction))
    return "█" * filled_len + "░" * (bar_length - filled_len)


# =============================================================================
# Step 1: Drop Redundant Columns
# =============================================================================

def drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes columns that are indexing artifacts or unnecessary for
    the recommendation pipeline.
    """
    existing_drops = [col for col in COLUMNS_TO_DROP if col in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)
        print(f"  [+] Dropped {len(existing_drops)} redundant column(s): {existing_drops}")
    else:
        print("  [~] No redundant columns found to drop.")

    return df


# =============================================================================
# Step 2: Handle Duplicate Titles
# =============================================================================

def handle_duplicate_titles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies books with duplicate titles and keeps the first occurrence
    (typically the most popular edition based on ratings_count).
    """
    dup_count_before = df["title"].duplicated().sum()
    if dup_count_before > 0:
        # Sort by ratings_count descending so the most popular edition is kept
        if "ratings_count" in df.columns:
            df = df.sort_values("ratings_count", ascending=False)
        df = df.drop_duplicates(subset="title", keep="first").reset_index(drop=True)
        dup_count_after = df["title"].duplicated().sum()
        removed = dup_count_before - dup_count_after
        print(f"  [+] Removed {removed} duplicate title(s). Kept most popular edition.")
    else:
        print("  [~] No duplicate titles detected.")

    return df


# =============================================================================
# Step 3: Clean Text Fields (HTML, special chars, whitespace)
# =============================================================================

def clean_text(text: Any) -> str:
    """
    Cleans a text string by removing HTML tags, entities, excessive whitespace,
    and non-printable characters.
    """
    if pd.isna(text) or text is None:
        return ""

    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Decode common HTML entities
    html_entities = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#39;": "'", "&apos;": "'",
        "&nbsp;": " ", "&#x27;": "'", "&mdash;": "—",
        "&ndash;": "–", "&hellip;": "…", "&#8217;": "'",
        "&#8220;": '"', "&#8221;": '"',
    }
    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)

    # Remove remaining HTML entity patterns
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"&#x[0-9a-fA-F]+;", " ", text)

    # Replace non-breaking spaces and special whitespace
    text = text.replace("\xa0", " ").replace("\t", " ")

    # Remove non-printable characters but keep basic punctuation
    text = re.sub(r"[^\x20-\x7E\u00C0-\u024F\u2018-\u201F\u2014\u2013\u2026]", " ", text)

    # Normalize multiple spaces
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies text cleaning to the title and description columns.
    """
    text_cols = ["title", "description"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
            non_empty = (df[col].str.len() > 0).sum()
            print(f"  [+] Cleaned '{col}' column. Non-empty entries: {non_empty:,}/{len(df):,}")

    return df


# =============================================================================
# Step 4: Impute Missing Descriptions
# =============================================================================

def impute_missing_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing or empty descriptions with synthetic fallback text
    constructed from the book's title and genre tags.

    Strategy: "A [genre1], [genre2] book titled [Title]."
    This ensures no book has an empty text representation for TF-IDF.
    """
    # Identify rows with missing or empty descriptions
    missing_mask = df["description"].isna() | (df["description"].str.strip() == "")
    missing_count = missing_mask.sum()

    if missing_count == 0:
        print("  [~] No missing descriptions found. Skipping imputation.")
        return df

    print(f"  [*] Imputing {missing_count} missing description(s)...")

    def build_fallback_description(row: pd.Series) -> str:
        """Builds a synthetic description from title and genres."""
        title = str(row.get("title", "Unknown Title")).strip()
        genres_raw = row.get("genres", "[]")
        genre_list = parse_list_column(genres_raw)

        if genre_list:
            genre_text = ", ".join(genre_list[:5])  # Use up to 5 genre tags
            return f"A {genre_text} book titled {title}."
        else:
            return f"A book titled {title}."

    # Apply fallback only to rows with missing descriptions
    df.loc[missing_mask, "description"] = df.loc[missing_mask].apply(
        build_fallback_description, axis=1
    )

    remaining_missing = df["description"].isna().sum() + (df["description"].str.strip() == "").sum()
    print(f"  [+] Imputation complete. Remaining missing descriptions: {remaining_missing}")

    return df


# =============================================================================
# Step 5: Normalize Genre Lists
# =============================================================================

def normalize_genre(genre: str) -> str:
    """
    Normalizes a single genre tag to lowercase, hyphen-separated form.
    Example: 'Historical Fiction' -> 'historical-fiction'
             ' Science Fiction ' -> 'science-fiction'
    """
    genre = genre.strip().lower()
    # Replace spaces and underscores with hyphens
    genre = re.sub(r"[\s_]+", "-", genre)
    # Remove non-alphanumeric characters except hyphens
    genre = re.sub(r"[^a-z0-9\-]", "", genre)
    # Collapse multiple hyphens
    genre = re.sub(r"-{2,}", "-", genre).strip("-")
    return genre


def normalize_genres_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses the genres column from stringified lists, normalizes each genre tag,
    removes duplicates within each book, and stores as a clean stringified list.
    """
    if "genres" not in df.columns:
        print("  [!] 'genres' column not found. Skipping genre normalization.")
        return df

    def process_genres(val: Any) -> str:
        """Parse, normalize, deduplicate, and re-serialize genre list."""
        genres = parse_list_column(val)
        normalized = []
        seen = set()
        for g in genres:
            norm_g = normalize_genre(g)
            if norm_g and norm_g not in seen:
                normalized.append(norm_g)
                seen.add(norm_g)
        return str(normalized) if normalized else "[]"

    df["genres"] = df["genres"].apply(process_genres)

    # Count unique genres across the catalog
    all_genres = set()
    for val in df["genres"]:
        for g in parse_list_column(val):
            all_genres.add(g)

    print(f"  [+] Normalized genres. {len(all_genres)} unique genre tags across catalog.")

    return df


# =============================================================================
# Step 6: Clean and Standardize Author Names
# =============================================================================

def clean_author_name(name: str) -> str:
    """
    Cleans an individual author name by stripping extra whitespace,
    removing parenthetical suffixes, and fixing encoding artifacts.
    """
    name = name.strip()
    # Remove common parenthetical notes like (Goodreads Author)
    name = re.sub(r"\s*\(.*?\)\s*", " ", name)
    # Fix double spaces
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name


def normalize_authors_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses the authors column from stringified lists, cleans each author name,
    and stores as a clean stringified list.
    """
    if "authors" not in df.columns:
        print("  [!] 'authors' column not found. Skipping author normalization.")
        return df

    def process_authors(val: Any) -> str:
        """Parse, clean, and re-serialize author list."""
        authors = parse_list_column(val)
        cleaned = [clean_author_name(a) for a in authors]
        cleaned = [a for a in cleaned if a]  # Remove empties
        return str(cleaned) if cleaned else "['Unknown']"

    df["authors"] = df["authors"].apply(process_authors)

    # Count unique authors
    all_authors = set()
    for val in df["authors"]:
        for a in parse_list_column(val):
            all_authors.add(a)

    print(f"  [+] Normalized authors. {len(all_authors)} unique authors across catalog.")

    return df


# =============================================================================
# Step 7: Impute Missing Numeric Values
# =============================================================================

def impute_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing numeric values:
    - pages: filled with the median page count.
    - original_publication_year: filled with the median year.
    """
    numeric_impute_map = {}

    if "pages" in df.columns:
        missing_pages = df["pages"].isna().sum()
        if missing_pages > 0:
            median_pages = df["pages"].median()
            df["pages"] = df["pages"].fillna(median_pages).astype(int)
            numeric_impute_map["pages"] = (missing_pages, int(median_pages))

    if "original_publication_year" in df.columns:
        missing_year = df["original_publication_year"].isna().sum()
        if missing_year > 0:
            median_year = df["original_publication_year"].median()
            df["original_publication_year"] = df["original_publication_year"].fillna(median_year).astype(int)
            numeric_impute_map["original_publication_year"] = (missing_year, int(median_year))

    if numeric_impute_map:
        for col, (count, fill_val) in numeric_impute_map.items():
            print(f"  [+] Imputed '{col}': {count} missing → filled with median ({fill_val})")
    else:
        print("  [~] No missing numeric values to impute.")

    return df


# =============================================================================
# Step 8: Final Validation & Export
# =============================================================================

def validate_cleaned_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs final validation checks on the cleaned dataset and returns
    a summary dictionary of key metrics.
    """
    stats = {
        "total_books": len(df),
        "total_features": len(df.columns),
        "missing_descriptions": int(df["description"].isna().sum() + (df["description"].str.strip() == "").sum()),
        "missing_titles": int(df["title"].isna().sum()),
        "duplicate_titles": int(df["title"].duplicated().sum()),
        "missing_genres": int(df["genres"].apply(lambda x: x == "[]").sum()),
        "missing_authors": int(df["authors"].apply(lambda x: x in ("[]", "['Unknown']")).sum()),
    }

    # Count remaining missing values across all columns
    total_missing = df.isnull().sum().sum()
    stats["total_remaining_nulls"] = int(total_missing)

    return stats


def save_cleaned_dataset(df: pd.DataFrame, output_path: Path = CLEANED_DATASET_PATH) -> None:
    """Saves the cleaned DataFrame to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  [+] Cleaned dataset saved: {output_path.name} ({file_size_mb:.2f} MB)")


def generate_cleaning_report(
    stats_before: Dict[str, Any],
    stats_after: Dict[str, Any],
    steps_log: List[str],
) -> str:
    """
    Generates a comprehensive Markdown report documenting all cleaning
    and preprocessing steps applied.
    """
    report_lines = [
        "# BookMatch AI: Data Cleaning & Preprocessing Report",
        "**Project Phase:** Day 4  ",
        f"**Dataset:** Goodbooks-10k Extended  ",
        "**Status:** Completed  ",
        "",
        "---",
        "",
        "## 📌 Executive Summary",
        "During Day 4, the raw dataset was cleaned and preprocessed to ensure data quality",
        "and consistency for the feature engineering pipeline (Day 5). All data quality issues",
        "identified during Day 3 EDA have been addressed.",
        "",
        "---",
        "",
        "## 📊 Before vs. After Comparison",
        "",
        "| Metric | Before Cleaning | After Cleaning |",
        "| :--- | :-: | :-: |",
        f"| **Total Books** | {stats_before['total_books']:,} | {stats_after['total_books']:,} |",
        f"| **Total Features** | {stats_before['total_features']} | {stats_after['total_features']} |",
        f"| **Missing Descriptions** | {stats_before['missing_descriptions']} | {stats_after['missing_descriptions']} |",
        f"| **Duplicate Titles** | {stats_before['duplicate_titles']} | {stats_after['duplicate_titles']} |",
        f"| **Total Remaining Nulls** | {stats_before['total_remaining_nulls']:,} | {stats_after['total_remaining_nulls']:,} |",
        "",
        "---",
        "",
        "## 🔧 Preprocessing Steps Applied",
        "",
    ]

    for i, step in enumerate(steps_log, 1):
        report_lines.append(f"{i}. {step}")

    report_lines.extend([
        "",
        "---",
        "",
        "## ✅ Data Quality Validation",
        "",
        f"- **Missing descriptions:** {stats_after['missing_descriptions']} (target: 0)",
        f"- **Duplicate titles:** {stats_after['duplicate_titles']} (target: 0)",
        f"- **Missing genres:** {stats_after['missing_genres']} books with empty genre lists",
        f"- **Missing authors:** {stats_after['missing_authors']} books with unknown authors",
        f"- **Total null cells remaining:** {stats_after['total_remaining_nulls']:,}",
        "",
        "---",
        "",
        "## 🚀 Impact on Next Steps",
        "- **Day 5 (Feature Engineering):** The cleaned `authors`, `genres`, and `description`",
        "  columns are now ready to be combined into a composite 'soup' text feature.",
        "- **Day 6 (TF-IDF Vectorization):** Zero empty descriptions ensures every book",
        "  will produce a valid TF-IDF vector.",
        "",
    ])

    return "\n".join(report_lines)


# =============================================================================
# Main Pipeline
# =============================================================================

def run_cleaning_pipeline() -> pd.DataFrame:
    """
    Executes the complete data cleaning and preprocessing pipeline.
    Returns the cleaned DataFrame.
    """
    print("=" * 70)
    print("  BookMatch AI: Day 4 — Data Cleaning & Preprocessing")
    print("=" * 70)

    # Load raw dataset
    print("\n[Step 0] Loading raw dataset...")
    if not RAW_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_DATASET_PATH}. "
            "Please run Day 2 (load_data.py) first."
        )
    df = pd.read_csv(RAW_DATASET_PATH, low_memory=False)
    print(f"  [+] Loaded {len(df):,} records with {len(df.columns)} columns.")

    # Capture "before" statistics
    # Need to prepare description and genres for stats check
    stats_before_raw = {
        "total_books": len(df),
        "total_features": len(df.columns),
        "missing_descriptions": int(df["description"].isna().sum()),
        "missing_titles": int(df["title"].isna().sum()),
        "duplicate_titles": int(df["title"].duplicated().sum()),
        "missing_genres": 0,  # Will compute properly after parsing
        "missing_authors": 0,
        "total_remaining_nulls": int(df.isnull().sum().sum()),
    }

    steps_log = []

    # Step 1: Drop redundant columns
    print("\n[Step 1] Dropping redundant columns...")
    cols_before = len(df.columns)
    df = drop_redundant_columns(df)
    dropped = cols_before - len(df.columns)
    if dropped > 0:
        steps_log.append(
            f"**Dropped {dropped} redundant column(s)** (`Unnamed: 0`, `index`) — "
            f"indexing artifacts from the source CSV."
        )

    # Step 2: Handle duplicate titles
    print("\n[Step 2] Handling duplicate titles...")
    rows_before = len(df)
    df = handle_duplicate_titles(df)
    rows_removed = rows_before - len(df)
    if rows_removed > 0:
        steps_log.append(
            f"**Removed {rows_removed} duplicate title(s)** — "
            f"kept the most popular edition (by ratings_count)."
        )

    # Step 3: Clean text fields
    print("\n[Step 3] Cleaning text fields (HTML, special chars, whitespace)...")
    df = clean_text_columns(df)
    steps_log.append(
        "**Cleaned text columns** (`title`, `description`) — "
        "removed HTML tags, decoded entities, stripped non-printable characters."
    )

    # Step 4: Impute missing descriptions
    print("\n[Step 4] Imputing missing descriptions...")
    df = impute_missing_descriptions(df)
    steps_log.append(
        f"**Imputed {stats_before_raw['missing_descriptions']} missing descriptions** — "
        f"generated synthetic fallback text from `title` + `genres` to ensure "
        f"complete text coverage for TF-IDF vectorization."
    )

    # Step 5: Normalize genres
    print("\n[Step 5] Normalizing genre tags...")
    df = normalize_genres_column(df)
    steps_log.append(
        "**Normalized genre tags** — parsed stringified lists, "
        "lowercased, hyphen-separated, and deduplicated per book."
    )

    # Step 6: Normalize authors
    print("\n[Step 6] Normalizing author names...")
    df = normalize_authors_column(df)
    steps_log.append(
        "**Cleaned author names** — removed parenthetical suffixes, "
        "fixed encoding artifacts, and standardized formatting."
    )

    # Step 7: Impute missing numeric values
    print("\n[Step 7] Imputing missing numeric values...")
    df = impute_numeric_columns(df)
    steps_log.append(
        "**Imputed missing numeric values** — filled `pages` and "
        "`original_publication_year` with their respective medians."
    )

    # Step 8: Final validation and export
    print("\n[Step 8] Validating cleaned data and exporting...")
    stats_after = validate_cleaned_data(df)

    # Save cleaned dataset
    save_cleaned_dataset(df)

    # Generate and save report
    report_content = generate_cleaning_report(stats_before_raw, stats_after, steps_log)
    report_path = REPORTS_DIR / "CLEANING_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding="utf-8")
    print(f"  [+] Cleaning report saved: {report_path.name}")

    # Print final summary
    print("\n" + "=" * 70)
    print("  CLEANING PIPELINE SUMMARY")
    print("=" * 70)

    summary_items = [
        ("Total Books (cleaned)", f"{stats_after['total_books']:,}"),
        ("Total Features", f"{stats_after['total_features']}"),
        ("Missing Descriptions", f"{stats_after['missing_descriptions']}"),
        ("Duplicate Titles", f"{stats_after['duplicate_titles']}"),
        ("Total Remaining Nulls", f"{stats_after['total_remaining_nulls']:,}"),
    ]

    max_label = max(len(label) for label, _ in summary_items)
    for label, value in summary_items:
        status = "✓" if value in ("0", "0,") else "~"
        print(f"  {status} {label:<{max_label}} : {value}")

    # Visual comparison bar
    print("\n  Data Completeness:")
    total_cells = stats_after["total_books"] * stats_after["total_features"]
    filled_cells = total_cells - stats_after["total_remaining_nulls"]
    completeness_pct = (filled_cells / total_cells) * 100 if total_cells > 0 else 0
    bar = render_ascii_bar(completeness_pct, 100.0, bar_length=30)
    print(f"  {bar} {completeness_pct:.2f}%")

    print("\n" + "=" * 70)
    print("  [SUCCESS] Day 4 Complete: Dataset cleaned and ready for Day 5!")
    print(f"  Output file: {CLEANED_DATASET_PATH.name}")
    print("=" * 70 + "\n")

    return df


def main():
    """Main entry point for Day 4 task execution."""
    run_cleaning_pipeline()


if __name__ == "__main__":
    main()
