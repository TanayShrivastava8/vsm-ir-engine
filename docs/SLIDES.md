# Presentation Deck Outline (10 Slides)

## Slide 1: Title Slide
* **Title:** Design and Implementation of a Vector Space Information Retrieval Engine from First Principles
* **Team:** Tanay Shrivastava (BTMC26O1138), Tanay Jaiswal (BTMC26O1137)
* **Guide:** Monika Vyas | MITS Gwalior (B.Tech – Mathematics and Computing)

## Slide 2: Problem Statement & Constraints
* Goal: Rank text documents against free-text queries using linear algebra.
* Hard Constraint: **Python Standard Library only** (no NumPy, Scikit-learn, NLTK).
* Why: Full mathematical transparency, explicit invariant enforcement, empirical Big-O analysis.

## Slide 3: Mathematical Model (Vector Space Model)
* Vocabulary $V$ sorted alphabetically $\rightarrow$ $D$-dimensional space $\mathbb{R}^D$.
* Normalized TF: $d_i = \text{count}(t_i, d) / |T(d)|$ (entries $\ge 0$, sum to $1$).
* Cosine Similarity: $\text{Score}(q, d) = \frac{q \cdot d}{\|q\|\,\|d\|} \in [0, 1]$.

## Slide 4: Worked Example (Section 4.5)
* `d1 = "the cat sat on the mat"`, `d2 = "the dog sat on the log"`, `q = "cat mat"`.
* $D = 7$, $q \cdot d_1 = 2$, $\|q\| = \sqrt{2}$, $\|d_1\| = \sqrt{8} \implies \text{Score}(q, d_1) = 0.5$, $\text{Score}(q, d_2) = 0.0$.

## Slide 5: System Architecture (Level 1 Data-Flow)
* Pipeline: `Document Input` $\rightarrow$ `normalize_text` $\rightarrow$ `build_vocabulary` $\rightarrow$ `vectorize` $\rightarrow$ `cosine_similarity` $\rightarrow$ `search` (top-$k$).
* Query follows the exact same normalization & projection path without altering $V$.

## Slide 6: Defensive Engineering & Program Invariants (I1–I8)
* Type safety (`I6`: `TypeError` on non-strings).
* Zero-norm protection (`||q|| = 0` or `||d|| = 0` returns `0.0`, zero crashes).
* Float precision clamping (`1.0000000002` $\rightarrow$ `1.0`, `I5`).
* Deterministic tie-breaking (`(-score, doc_id)` via `heapq.nsmallest`, `I7`).

## Slide 7: Milestone 3 — Interactive CLI & Evaluation
* Live terminal loop (`cli.py`) supporting ad-hoc queries, `:docs`, `:schema`, `:top`, `:save`.
* Reports wall-clock latency (`time.perf_counter()`) and full score distribution (min, max, mean, matches).

## Slide 8: Milestone 4 — Sparse Vectors & Inverted Index
* Problem with Dense VSM: $O(N \times D)$ memory & query time (most entries are `0.0`).
* Solution: JSON schema storing sparse vectors `{term_idx: tf}`, precomputed `doc_norms`, and posting lists `{term: [[doc_id, tf], ...]}`.
* Query scoring collapses to $O(|q| + \sum |\text{posting\_list}| + k)$ with exact numerical equivalence.

## Slide 9: Empirical Verification & Benchmarks ($D = 800$, 20 words/doc)
* Automated `unittest` suite (`test_vsm.py`): **10/10 tests passing** in 0.23s (edge cases, invariants, 200-doc stress test).
* **Measured Benchmark (`python3 benchmark.py`):**
  * `N = 100`: Dense `680.6 KB` / `122.4 ms` vs. Inverted Index `562.2 KB` / `0.056 ms` (**2,193x faster**)
  * `N = 500`: Dense `3,389.3 KB` / `614.4 ms` vs. Inverted Index `2,219.5 KB` / `0.097 ms` (**6,304x faster**)
  * `N = 1000`: Dense `6,780.5 KB` / `1,227.1 ms` vs. Inverted Index `4,309.5 KB` / `0.142 ms` (**8,662x faster**)
  * `N = 2000`: Dense `13,561.1 KB` / `2,469.1 ms` vs. Inverted Index `8,454.2 KB` / `0.238 ms` (**10,376x faster**, **37.7% less RAM**)

## Slide 10: Conclusion & Live Demo
* Summary of deliverables across Milestones 1–5.
* Live CLI demonstration (`python3 cli.py`) + Q&A.
