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
