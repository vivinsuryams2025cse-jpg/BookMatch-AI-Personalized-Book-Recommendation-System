# BookMatch AI: Data Cleaning & Preprocessing Report
**Project Phase:** Day 4  
**Dataset:** Goodbooks-10k Extended  
**Status:** Completed  

---

## 📌 Executive Summary
During Day 4, the raw dataset was cleaned and preprocessed to ensure data quality
and consistency for the feature engineering pipeline (Day 5). All data quality issues
identified during Day 3 EDA have been addressed.

---

## 📊 Before vs. After Comparison

| Metric | Before Cleaning | After Cleaning |
| :--- | :-: | :-: |
| **Total Books** | 10,000 | 9,964 |
| **Total Features** | 30 | 28 |
| **Missing Descriptions** | 57 | 0 |
| **Duplicate Titles** | 36 | 61 |
| **Total Remaining Nulls** | 2,029 | 1,873 |

---

## 🔧 Preprocessing Steps Applied

1. **Dropped 2 redundant column(s)** (`Unnamed: 0`, `index`) — indexing artifacts from the source CSV.
2. **Removed 36 duplicate title(s)** — kept the most popular edition (by ratings_count).
3. **Cleaned text columns** (`title`, `description`) — removed HTML tags, decoded entities, stripped non-printable characters.
4. **Imputed 57 missing descriptions** — generated synthetic fallback text from `title` + `genres` to ensure complete text coverage for TF-IDF vectorization.
5. **Normalized genre tags** — parsed stringified lists, lowercased, hyphen-separated, and deduplicated per book.
6. **Cleaned author names** — removed parenthetical suffixes, fixed encoding artifacts, and standardized formatting.
7. **Imputed missing numeric values** — filled `pages` and `original_publication_year` with their respective medians.

---

## ✅ Data Quality Validation

- **Missing descriptions:** 0 (target: 0)
- **Duplicate titles:** 61 remaining — these are legitimately different books by different authors sharing the same title (e.g., "Home", "Beloved")
- **Missing genres:** 0 books with empty genre lists
- **Missing authors:** 0 books with unknown authors
- **Total null cells remaining:** 1,873

---

## 🚀 Impact on Next Steps
- **Day 5 (Feature Engineering):** The cleaned `authors`, `genres`, and `description`
  columns are now ready to be combined into a composite 'soup' text feature.
- **Day 6 (TF-IDF Vectorization):** Zero empty descriptions ensures every book
  will produce a valid TF-IDF vector.
