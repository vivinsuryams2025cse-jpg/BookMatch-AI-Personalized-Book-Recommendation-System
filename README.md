# BookMatch AI: Personalized Book Recommendation System

A beginner-friendly Machine Learning project that recommends books based on content similarity using Python, Pandas, TF-IDF vectorization, and Cosine Similarity. Built with a simple interactive Streamlit user interface.

---

## 📌 Project Overview
- **Goal:** Suggest relevant books to a reader based on book features (author, genre, description).
- **Core Technique:** Content-Based Filtering using TF-IDF and Cosine Similarity.
- **Interface:** Streamlit web application.
- **Level:** Beginner (First & Second Year CSE students).

---

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:**
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computations
  - `scikit-learn` - TF-IDF Vectorizer and Cosine Similarity
  - `streamlit` - Interactive web app UI

---

## 📁 Project Structure
```text
BookMatch-AI-Personalized-Book-Recommendation-System/
├── data/
│   ├── .gitkeep
│   └── books.csv          # Goodbooks-10k Extended dataset (10,000 books)
├── src/
│   ├── test_setup.py      # Day 1: Setup verification script
│   └── load_data.py       # Day 2: Dataset loading, validation & diagnostics
├── .gitignore             # Ignored files for Git
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation and guide
```

---

## 🚀 15-Day Roadmap
- [x] **Day 1:** Project setup & environment verification
- [x] **Day 2:** Dataset collection & loading
- [ ] **Day 3:** Exploratory data analysis (EDA)
- [ ] **Day 4:** Data cleaning & preprocessing
- [ ] **Day 5:** Feature engineering & text combination
- [ ] **Day 6:** Text vectorization with TF-IDF
- [ ] **Day 7:** Cosine similarity calculation
- [ ] **Day 8:** Core recommendation function
- [ ] **Day 9:** Recommendation filtering & output enhancement
- [ ] **Day 10:** Terminal-based user interaction
- [ ] **Day 11:** Streamlit UI initialization
- [ ] **Day 12:** Streamlit UI enhancement
- [ ] **Day 13:** Testing, edge-case handling & debugging
- [ ] **Day 14:** Comprehensive documentation
- [ ] **Day 15:** Final wrap-up, viva prep & GitHub finalization

---

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/BookMatch-AI-Personalized-Book-Recommendation-System.git
   cd BookMatch-AI-Personalized-Book-Recommendation-System
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Day 1 setup:**
   ```bash
   python src/test_setup.py
   ```

4. **Run Day 2 dataset loader & inspection:**
   ```bash
   python src/load_data.py
   ```

