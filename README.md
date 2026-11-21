# Vector Space Information Retrieval Engine (From First Principles)

**Institution:** Madhav Institute of Technology & Science, Gwalior (B.Tech – Mathematics and Computing)  
**Team:** Tanay Shrivastava (`BTMC26O1138`), Tanay Jaiswal (`BTMC26O1137`)  
**Project Guide:** Monika Vyas  

## Overview

A deterministic text search engine built **from mathematical first principles using only Python's standard library** (zero external dependencies—no NumPy, Pandas, Scikit-learn, NLTK, or SciPy). Documents and queries are projected as normalized Term-Frequency (TF) vectors in $\mathbb{R}^D$, ranked via cosine similarity, and accelerated using a sparse vector schema and inverted index.

## Mathematical Model

For corpus $C = \{d_1, \dots, d_N\}$ and alphabetically sorted vocabulary $V = \{t_0, \dots, t_{D-1}\}$:

$$d_i = \frac{\text{count}(t_i, d)}{|T(d)|}, \qquad \text{Score}(q, d) = \frac{q \cdot d}{\|q\| \times \|d\|} = \frac{\sum_{i=0}^{D-1} q_i d_i}{\sqrt{\sum_{i=0}^{D-1} q_i^2}\;\sqrt{\sum_{i=0}^{D-1} d_i^2}}$$

## Enforced Program Invariants (I1–I8)

| ID | Invariant |
|----|-----------|
| **I1** | Every dense vector has length exactly $D = |V|$. |
| **I2** | Every token is a non-empty lowercase string with no punctuation or whitespace. |
| **I3** | Vocabulary indices are unique, contiguous ($0 \dots D-1$), and alphabetically sorted. |
| **I4** | All TF entries are $\ge 0$; non-empty in-vocabulary document vectors sum to $1.0$. |
| **I5** | Every cosine score lies in $[0.0, 1.0]$ (zero-norm guarded, float drift clamped). |
| **I6** | Non-string documents or queries raise `TypeError`. |
| **I7** | Ranking is deterministic: score descending, ties broken by ascending `doc_id`. |
| **I8** | Vocabulary $V$ is built strictly from the corpus; queries never modify $V$. |

## Standard-Library Boundary

Exclusively uses the 10 allowed standard-library modules:  
`re`, `string`, `math`, `json`, `sys`, `time`, `collections`, `heapq`, `unittest`, `tracemalloc`.

## Prerequisites & Setup

* Python 3.8+ (no `pip install` needed).

```bash
git clone <your-repo-url>
cd vsm_ir_engine
