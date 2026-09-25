# BookMatch AI: Exploratory Data Analysis (EDA) Report
**Project Phase:** Day 3  
**Dataset:** Goodbooks-10k Extended (10,000 books)  
**Status:** Completed  

---

## 📌 Executive Summary
During Day 3, a comprehensive exploratory analysis was performed on the book catalog of 10,000 titles to establish baseline data distributions, identify data quality challenges, and inform data preprocessing (Day 4) and feature engineering (Day 5).

### Key Takeaways:
1. **Catalog Scale:** 10,000 unique books across 30 attributes with an in-memory footprint of ~22.2 MB.
2. **Ratings Tendency:** Average book rating is **4.00 / 5.0** (median: 4.02), showing high overall reader satisfaction (standard deviation: 0.25).
3. **Genre Richness:** Over 39 distinct genre tags exist, with an average of **4.72 genres per book**. Fiction, Fantasy, and Young Adult dominate the catalog.
4. **Author Representation:** Over 6,502 distinct authors. The most prolific authors include Stephen King, Nora Roberts, and James Patterson.
5. **Missing Value Impact for Recommender:**
   - **57 books (0.57%)** have missing descriptions.
   - Handled in Day 4 via content imputation (combining title and genre tags as fallback) to avoid empty vector representations.

---

## 📊 Summary Statistics

| Metric | Value |
| :--- | :--- |
| **Total Books** | 10,000 |
| **Total Features** | 30 |
| **Duplicate IDs** | 0 |
| **Mean Average Rating** | 4.00 / 5.0 |
| **Median Average Rating** | 4.02 / 5.0 |
| **Rating Standard Deviation** | 0.25 |
| **Unique Authors** | 6,502 |
| **Unique Genres** | 39 |
| **Mean Description Words** | 150.3 words |
| **Missing Descriptions** | 57 books (0.57%) |

---

## 🏷️ Top 10 Most Common Genres

| # | Genre | Book Count | % of Catalog | Avg Rating |
| :-: | :--- | :-: | :-: | :-: |
| 1 | Fiction | 8,272 | 82.7% | 3.99★ |
| 2 | Fantasy | 3,746 | 37.5% | 4.06★ |
| 3 | Romance | 3,307 | 33.1% | 4.00★ |
| 4 | Contemporary | 2,918 | 29.2% | 3.90★ |
| 5 | Young Adult | 2,756 | 27.6% | 4.03★ |
| 6 | Mystery | 2,481 | 24.8% | 3.95★ |
| 7 | Classics | 2,110 | 21.1% | 4.03★ |
| 8 | Thriller | 1,822 | 18.2% | 3.95★ |
| 9 | Historical Fiction | 1,793 | 17.9% | 3.95★ |
| 10 | Nonfiction | 1,641 | 16.4% | 4.02★ |

---

## ✍️ Top 10 Prolific Authors

| # | Author | Book Count in Catalog |
| :-: | :--- | :-: |
| 1 | Stephen King | 90 |
| 2 | James Patterson | 66 |
| 3 | Nora Roberts | 51 |
| 4 | Terry Pratchett | 47 |
| 5 | Dean Koontz | 45 |
| 6 | Agatha Christie | 42 |
| 7 | Neil Gaiman | 36 |
| 8 | J.D. Robb | 35 |
| 9 | Meg Cabot | 34 |
| 10 | Janet Evanovich | 29 |

---

## 📈 Visualizations Exported
The following visual figures have been generated and saved to `reports/figures/`:
1. `rating_distribution.png` - Distribution histogram of average user ratings.
2. `top_genres.png` - Top 12 genres ranked by book frequency.
3. `top_authors.png` - Authors with the highest catalog presence.
4. `book_length_distribution.png` - Breakdown of page counts across books.

---

## 🚀 Impact on Next Steps
- **Day 4 (Data Cleaning & Preprocessing):**
  - Impute 57 missing descriptions with synthetic fallback text based on `title` + `genres`.
  - Normalize genre lists into standardized tokens.
  - Strip punctuation and formatting artifacts.
- **Day 5 (Feature Engineering):**
  - Form the composite 'soup' text feature combining `title`, `authors`, `genres`, and `description`.
- **Day 6 (TF-IDF Vectorization):**
  - Vectorize the text corpus to build the content matrix for Cosine Similarity.
