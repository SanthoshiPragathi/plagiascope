# 🔍 PlagiaScope — Semantic Plagiarism Detector

> A novel plagiarism detection system combining **Word2Vec embeddings** with **0/1 Knapsack optimization** — catches paraphrased content that traditional tools miss.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Algorithm](https://img.shields.io/badge/Algorithm-Knapsack_DP-purple)

---

## 🧠 What Makes This Unique

Traditional plagiarism detectors rely on exact string matching (Jaccard, cosine similarity on word vectors). This system introduces a **novel triple-blend hybrid similarity**:

| Component | Weight | Purpose |
|-----------|--------|---------|
| Word2Vec Cosine | 50% | Structural & contextual similarity |
| Synonym Dictionary | 35% | Catches doctor→physician rewrites |
| Jaccard Lexical | 15% | Exact phrase matching |

This score then feeds into a **0/1 Knapsack optimizer** that selects the *optimal non-overlapping* set of matching n-gram pairs — a combination not found in existing plagiarism detection literature.

---

## ✨ Features

- 🧬 **Semantic analysis** — detects meaning-level plagiarism
- 📊 **Multi-gram analysis** — runs unigram, bigram, trigram simultaneously
- 🎯 **Knapsack optimization** — optimal match selection with positional weighting
- 💡 **TF-IDF boosting** — rare/important phrases weighted higher
- 🖊️ **Word highlighting** — shows exactly which words matched
- 📁 **File upload** — supports .txt files
- 📜 **History** — tracks past comparisons
- ⚡ **FastAPI backend** — fast, modern REST API

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/plagiascope.git
cd plagiascope
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the server
```bash
uvicorn main:app --reload
```

### 4. Open in browser
```
http://localhost:8000
```

---

## 🗂️ Project Structure

```
plagiascope/
├── main.py          # FastAPI backend & routes
├── engine.py        # Core detection algorithm
├── templates/
│   └── index.html   # Frontend UI
├── requirements.txt
└── README.md
```

---

## 📊 How It Works

```
Input Documents
      ↓
  Tokenize + Clean (stopwords, punctuation)
      ↓
  Train Word2Vec on combined corpus
      ↓
  Generate N-Grams (n = 1, 2, 3)
      ↓
  Compute Hybrid Similarity for each pair
  (Word2Vec cosine + Synonym overlap + Jaccard)
      ↓
  Build Knapsack Items
  (value = similarity × TF-IDF boost, weight = positional importance)
      ↓
  0/1 Knapsack DP → optimal non-overlapping matches
      ↓
  Weighted average score across n-gram sizes
      ↓
  Final Plagiarism % + Verdict
```

---

## 🧪 Test Cases

| Source | Suspect | Expected |
|--------|---------|----------|
| "Machine learning is a subset of AI..." | "ML is part of AI which allows..." | HIGH (~100%) |
| "Doctor administered medication..." | "Physician gave medicine..." | LOW-MODERATE (~28%) |
| "Photosynthesis converts sunlight..." | "Python is used for web apps..." | ORIGINAL (0%) |

---

## 👩‍💻 Author

**Vanaparthi Santhoshi Pragathi**  
B.Tech ECE — Vellore Institute of Technology  
[GitHub](https://github.com/SanthoshiPragathi)

---

## 📄 License
MIT
