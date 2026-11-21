# Final Project Report
## Design and Implementation of a Vector Space Information Retrieval Engine from First Principles

**Institution:** Madhav Institute of Technology & Science, Gwalior  
**Department:** B.Tech – Mathematics and Computing  
**Team Members:** Tanay Shrivastava (`BTMC26O1138`), Tanay Jaiswal (`BTMC26O1137`)  
**Project Guide:** Monika Vyas  
**Date:** November 2026  

---

## 1. Executive Summary

This project designs, implements, verifies, and optimizes a deterministic Vector Space Information Retrieval (IR) engine strictly from first principles using Python's standard library alone. Documents and ad-hoc user queries are normalized into clean tokens, mapped into an alphabetically indexed vocabulary $V$, projected into normalized Term-Frequency (TF) vectors in $\mathbb{R}^D$, and ranked using cosine similarity. Following the dense prototype (Milestones 2–3), the system was optimized in Milestone 4 using sparse vectors, precomputed document norms, and an inverted index—achieving identical rankings with substantially lower memory footprint and query latency.

---

## 2. Requirements & Compliance Matrix

| Requirement | Description | Verification Artifact | Status |
|-------------|-------------|-----------------------|--------|
| FR1–FR4 | Corpus ingestion, text normalization, sorted vocabulary $V$, normalized TF vectors | `vsm_engine.py`, `test_vsm.py` | Verified |
| FR5–FR7 | Pure-Python dot product, L2 norm, cosine similarity, top-$k$ ranked retrieval | `vsm_engine.py`, `test_vsm.py` | Verified |
| FR8 | Safe edge-case handling (empty query/doc, OOV terms, non-string `TypeError`, float clamping) | `vsm_engine.py`, `test_vsm.py` | Verified |
| FR9 | Interactive command-line interface with ad-hoc queries | `cli.py` | Verified |
| NFR1 | Determinism (sorted vocabulary, ascending `doc_id` tie-breaking) | `vsm_engine.py`, `test_vsm.py` | Verified |
| NFR2 | Standard library only (strict 10-module allowlist) | All `.py` files | Verified |
| NFR3 | Score bounds $[0, 1]$, zero division-by-zero errors | `cosine_similarity`, `search_inverted_index` | Verified |
| NFR4 | Independent unit-testable modules | `test_vsm.py` | Verified |
| NFR5 | Latency (`time`) and memory (`tracemalloc`) reporting | `cli.py`, `benchmark.py` | Verified |

---

## 3. System Architecture & Data Pipeline

1. **Normalization (`normalize_text`):** Case-folds text, strips `string.punctuation`, collapses whitespace via `re.sub`, and splits into tokens. Enforces `I2` and `I6`.
2. **Vocabulary Basis (`build_vocabulary`):** Deduplicates corpus tokens via `set`, sorts alphabetically, and assigns contiguous indices $0 \dots D-1$. Enforces `I3`.
3. **Vector Projection (`vectorize` & `vectorize_sparse`):** Computes $d_i = \text{count}(t_i, d) / |T(d)|$. Unseen query terms are ignored (`I8`). Enforces `I1` and `I4`.
4. **Geometric Scoring (`dot_product`, `euclidean_norm`, `cosine_similarity`):** Computes $\frac{q \cdot d}{\|q\| \|d\|}$, returns `0.0` on zero-norm inputs, and clamps float drift to $[0.0, 1.0]$ (`I5`).
5. **Ranked Retrieval (`search` & `search_inverted_index`):** Extracts top-$k$ results via `heapq.nsmallest` keyed on `(-score, doc_id)` (`I7`).

---

## 4. Section 4.5 Worked Example Verification

* **Corpus:** `d1 = "the cat sat on the mat"`, `d2 = "the dog sat on the log"`
* **Query:** `"cat mat"`
* **Vocabulary ($D = 7$):** `{'cat': 0, 'dog': 1, 'log': 2, 'mat': 3, 'on': 4, 'sat': 5, 'the': 6}`
* **Observed Output (both Dense and Inverted-Index engines):**
  * Rank 1: `doc_id = 0`, `Score = 0.5000`
  * Rank 2: `doc_id = 1`, `Score = 0.0000`

---

## 5. Optimization & Complexity Analysis (Milestone 4)

| Operation / Resource | Dense Engine | Sparse / Inverted Index |
|----------------------|--------------|-------------------------|
| Build Vocabulary | $O(T \log D)$ | $O(T \log D)$ |
| Score One Query | $O(N \times D)$ | $O\left(|q| + \sum_{t \in q} |\text{posting\_list}(t)| + k\right)$ |
| Vector Memory | $O(N \times D)$ | $O(\text{nnz})$ |

### Empirical Benchmark Results (`python3 benchmark.py`, $D = 800$, 20 words/doc, 5-run average)

| N (Docs) | D (\|V\|) | Dense Mem (KB) | Sparse Schema Mem (KB) | Dense Query Time (ms) | Inverted-Index Time (ms) | Speedup |
|---------:|----------:|---------------:|-----------------------:|----------------------:|-------------------------:|--------:|
| 100 | 800 | 680.62 | 562.17 | 122.385 | 0.056 | 2,193.5x |
| 500 | 800 | 3,389.28 | 2,219.53 | 614.420 | 0.097 | 6,303.6x |
| 1000 | 800 | 6,780.53 | 4,309.48 | 1,227.147 | 0.142 | 8,661.8x |
| 2000 | 800 | 13,561.12 | 8,454.21 | 2,469.131 | 0.238 | 10,376.2x |

**Key Findings:**
* **Linear $O(N \times D)$ verification:** Scaling $N$ by $20\times$ (`100` $\rightarrow$ `2000`) scaled dense memory by $19.9\times$ and dense latency by $20.2\times$.
* **Sub-millisecond inverted-index retrieval:** Traversing only query-term posting lists achieved up to a **10,376x speedup** at $N = 2000$ while passing strict numerical equivalence assertions (`rel_tol = 1e-9`) against the dense engine.
* **Memory savings:** At $N = 2000$, the full sparse + inverted-index schema reduced peak traced memory by **37.7%** compared to storing dense vectors alone.

---

## 6. Testing & Verification

Executed `python3 -m unittest test_vsm.py -v` (**10/10 tests passed in 0.226s**), verifying:
* Section 4.5 worked example
* FR1–FR8, NFR1–NFR4, Invariants `I1`–`I8`
* Section 4.4 edge cases (empty inputs, punctuation-only strings, OOV queries, zero norms, float drift clamping, non-string `TypeError`s)
* Section 8 JSON schema serialization/deserialization (full and minimal schemas)
* 200-document synthetic stress test confirming exact numerical equality (`tol = 1e-9`) between Dense and Inverted-Index retrieval.

---

## 7. Future Work

* **TF-IDF / BM25 Weighting:** Down-weight common stopwords (`"the"`, `"on"`) by multiplying TF by Inverse Document Frequency $\log(N / \text{df}_t)$.
* **Stemming / Lemmatization:** Normalize morphological variants (`"cat"` vs. `"cats"`).
* **Disk-Backed Postings:** Persist posting lists in chunked files for corpora exceeding RAM.
