# Viva Preparation Guide (High-Probability Examiner Questions)

1. **Why use cosine similarity instead of Euclidean distance for text?**
   Cosine similarity measures the *angle* (direction) between vectors rather than magnitude. A 10-word document and a 1,000-word document with identical word proportions have angle $0^\circ$ (cosine = `1.0`), whereas their raw count vectors would be far apart in Euclidean distance.

2. **Why is every score guaranteed to lie in $[0, 1]$ instead of $[-1, 1]$?**
   Term frequencies are always $\ge 0$ (Invariant I4), so all vectors lie in the non-negative orthant of $\mathbb{R}^D$. The angle between any two vectors in that orthant is between $0^\circ$ and $90^\circ$, hence $\cos\theta \in [0, 1]$.

3. **Why did you clamp cosine similarity with `max(0.0, min(1.0, raw_score))` if mathematically $\cos\theta \le 1$?**
   IEEE-754 binary floating-point rounding causes identical vectors such as `[1/3, 1/3, 1/3]` to evaluate slightly above `1.0` (e.g., `1.0000000000000002`). Clamping enforces Invariant I5 (`0.0 <= score <= 1.0`) without distorting valid interior scores the way `round()` would.

4. **Does dividing by $|T(d)|$ (normalized TF) change the cosine similarity compared to raw counts?**
   No—multiplying any vector by a positive scalar $c = 1/|T(d)|$ cancels out in $\frac{q \cdot (cd)}{\|q\|\|cd\|} = \frac{q \cdot d}{\|q\|\|d\|}$. We normalize TF to satisfy FR4 and Invariant I4: every non-empty document vector sums to `1.0`, keeping entries bounded in $[0, 1]$ and directly interpretable as word proportions.

5. **What happens if a user submits an empty query or words not present in the corpus?**
   Out-of-vocabulary words are ignored (Invariant I8: query never modifies $V$). The query vector becomes the zero vector ($\|q\| = 0$). Both `cosine_similarity` and `search_inverted_index` detect `norm == 0.0` before dividing and safely return `0.0` for all documents (no `ZeroDivisionError`).

6. **How do you guarantee deterministic rankings when two documents have identical scores?**
   We rank with key `(-score, doc_id)` using `heapq.nsmallest`. Higher scores come first, and exact ties are always broken by ascending `doc_id` (Invariant I7, NFR1). Likewise, sorting vocabulary keys alphabetically guarantees deterministic coordinates (Invariant I3).

7. **Why does the dense representation scale poorly, and how does the inverted index fix it?**
   Dense vectors store $N \times D$ floats—even when a 20-word document in a 10,000-word vocabulary is 99.8% zeros—requiring $O(N \times D)$ RAM and query time. Sparse vectors store only non-zero entries $O(\text{nnz})$, and the inverted index lets a query visit only documents appearing in the posting lists of the query terms: $O(|q| + \sum |\text{posting\_list}| + k)$.

8. **Why do you precompute and store `doc_norms` in the Section 8 schema?**
   During inverted-index scoring, a document's posting entries are only visited for terms shared with the query, not for all terms in that document. Precomputing $\|d\|$ once during indexing lets us divide `dot_val / (q_norm * d_norm)` in $O(1)$ time without reading the rest of document $d$.

9. **Why did you use `heapq.nsmallest` instead of sorting the entire scored list?**
   Sorting all $M$ scored documents takes $O(M \log M)$, whereas `heapq.nsmallest(k, ...)` extracts the top-$k$ items in $O(M \log k)$—which is faster when $k \ll M$.

10. **What are the main limitations of Normalized TF VSM, and how would you extend it?**
    (a) High-frequency stopwords (`"the"`, `"on"`) carry high weight $\rightarrow$ fix with **TF-IDF** or **BM25**.  
    (b) Bag-of-words ignores word order (`"dog bites man"` = `"man bites dog"`) and synonyms (`"car"` vs. `"automobile"`) $\rightarrow$ fix with n-grams or dense semantic embeddings.
