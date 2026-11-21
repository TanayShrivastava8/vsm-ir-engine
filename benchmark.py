"""
benchmark.py
Milestone 4 & Section 9: Empirical verification of time complexity and memory
consumption comparing Dense O(N * D) vs. Sparse / Inverted Index using only
Python's standard-library modules: heapq, math, time, tracemalloc.
"""

import heapq
import math
import time
import tracemalloc

import vsm_engine


def generate_synthetic_corpus(num_docs, words_per_doc=20, vocab_pool_size=800):
    """Deterministically generate a synthetic corpus of N clean lowercase documents."""
    corpus = []
    for doc_id in range(num_docs):
        words = [
            f"term{(doc_id * 17 + pos * 31) % vocab_pool_size}"
            for pos in range(words_per_doc)
        ]
        corpus.append(" ".join(words))
    return corpus


def profile_corpus_size(num_docs, words_per_doc=20, vocab_pool_size=800, repeats=5):
    """Profile memory usage (KB) and average query scoring latency (ms)."""
    corpus = generate_synthetic_corpus(
        num_docs, words_per_doc=words_per_doc, vocab_pool_size=vocab_pool_size
    )
    tokenized = [vsm_engine.normalize_text(d) for d in corpus]
    vocab = vsm_engine.build_vocabulary(tokenized)
    d_dim = len(vocab)

    # 1. Profile Dense Vector Matrix Memory: O(N * D)
    tracemalloc.start()
    dense_vectors = [vsm_engine.vectorize(tokens, vocab) for tokens in tokenized]
    _, dense_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    query = "term5 term42 term105"
    q_vec = vsm_engine.vectorize(vsm_engine.normalize_text(query), vocab)

    # 2. Measure Dense Pre-Built Query Scoring Latency: O(N * D)
    t0 = time.perf_counter()
    dense_top_k = []
    for _ in range(repeats):
        scored = [
            (doc_id, vsm_engine.cosine_similarity(q_vec, d_vec))
            for doc_id, d_vec in enumerate(dense_vectors)
        ]
        dense_top_k = heapq.nsmallest(10, scored, key=lambda item: (-item[1], item[0]))
    dense_query_ms = ((time.perf_counter() - t0) / float(repeats)) * 1000.0

    # Release dense matrix before profiling sparse structures
    del dense_vectors

    # 3. Profile Sparse Vectors + Inverted Index Memory: O(nnz)
    tracemalloc.start()
    schema = vsm_engine.build_index_schema(corpus)
    _, sparse_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 4. Measure Inverted Index Query Scoring Latency: O(sum |posting_list|)
    t1 = time.perf_counter()
    inv_top_k = []
    for _ in range(repeats):
        inv_top_k = vsm_engine.search_inverted_index(query, schema, k=10)
    inv_query_ms = ((time.perf_counter() - t1) / float(repeats)) * 1000.0

    # Verify exact equivalence between dense and inverted-index top-10 results
    assert len(dense_top_k) == len(inv_top_k)
    for (d_id, d_sc), (i_id, i_sc) in zip(dense_top_k, inv_top_k):
        assert d_id == i_id and math.isclose(d_sc, i_sc, rel_tol=1e-9, abs_tol=1e-9)

    return {
        "N": num_docs,
        "D": d_dim,
        "dense_kb": dense_peak_bytes / 1024.0,
        "sparse_kb": sparse_peak_bytes / 1024.0,
        "dense_ms": dense_query_ms,
        "inv_ms": inv_query_ms,
        "speedup": dense_query_ms / inv_query_ms if inv_query_ms > 0.0 else 0.0,
    }


def main():
    print("=" * 98)
    print("SECTION 9 & MILESTONE 4: DENSE O(N * D) vs. SPARSE / INVERTED-INDEX BENCHMARK")
    print("=" * 98)
    header = (
        f"{'N (Docs)':>9} | {'D (|V|)':>7} | {'Dense Mem (KB)':>14} | "
        f"{'Sparse Mem (KB)':>15} | {'Dense Time (ms)':>15} | "
        f"{'InvIdx Time (ms)':>16} | {'Speedup':>8}"
    )
    print(header)
    print("-" * 98)
    for n in [100, 500, 1000, 2000]:
        stats = profile_corpus_size(n, words_per_doc=20, vocab_pool_size=800, repeats=5)
        print(
            f"{stats['N']:>9d} | {stats['D']:>7d} | {stats['dense_kb']:>14.2f} | "
            f"{stats['sparse_kb']:>15.2f} | {stats['dense_ms']:>15.3f} | "
            f"{stats['inv_ms']:>16.3f} | {stats['speedup']:>7.1f}x"
        )
    print("=" * 98)
    print("Equivalence check passed: Dense and Inverted-Index top-10 rankings match identically.")


if __name__ == "__main__":
    main()
