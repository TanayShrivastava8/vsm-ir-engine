"""
cli.py
Milestone 3: Interactive Command-Line Interface for ad-hoc search queries (FR9),
relevance ranking, score distribution statistics, and latency measurements (NFR5).

Allowed standard-library modules used: json, sys, time
"""

import json
import sys
import time

import vsm_engine

DEFAULT_CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are domestic animals",
    "linear algebra and vector space models power information retrieval",
    "cosine similarity measures the angle between two document vectors",
    "discrete mathematics and graph theory form the foundation of computer science",
    "python standard library provides data structures without external dependencies",
    "inverted index optimization reduces query scoring complexity in search engines",
]


def run_worked_example():
    """Verify and display the exact worked example from Section 4.5 of the report."""
    print("=" * 74)
    print("SECTION 4.5 WORKED EXAMPLE VERIFICATION")
    print("=" * 74)
    corpus = ["the cat sat on the mat", "the dog sat on the log"]
    query = "cat mat"

    tokenized = [vsm_engine.normalize_text(d) for d in corpus]
    vocab = vsm_engine.build_vocabulary(tokenized)
    results = vsm_engine.search(query, corpus, k=2)

    assert results[0][0] == 0 and abs(results[0][1] - 0.5) < 1e-9
    assert results[1][0] == 1 and abs(results[1][1] - 0.0) < 1e-9

    print(f"d1 (doc_id=0): {corpus[0]!r}")
    print(f"d2 (doc_id=1): {corpus[1]!r}")
    print(f"Query        : {query!r}")
    print(f"Vocabulary V : {vocab} (D = {len(vocab)})")
    for rank, (doc_id, score) in enumerate(results, start=1):
        print(f"Rank {rank}: doc_id={doc_id} | Score = {score:.4f} | Text = {corpus[doc_id]!r}")
    print("=" * 74 + "\n")


def compute_score_distribution(scores):
    """Compute min, max, mean, and matching (score > 0) count for similarity scores."""
    if not scores:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "non_zero": 0}
    return {
        "min": min(scores),
        "max": max(scores),
        "mean": sum(scores) / float(len(scores)),
        "non_zero": sum(1 for s in scores if s > 0.0),
    }


def load_corpus_from_file(filepath):
    """Load a corpus from a JSON list of strings or a Section 8 schema JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    if isinstance(loaded, list) and all(isinstance(x, str) for x in loaded):
        return loaded
    if isinstance(loaded, dict) and "documents" in loaded and isinstance(loaded["documents"], list):
        return [d["raw_text"] for d in loaded["documents"]]
    raise ValueError(
        "JSON file must be either a list of strings or a Section 8 schema containing 'documents'."
    )


def print_help():
    """Display available interactive commands."""
    print("Commands:")
    print("  <any text>     Run ad-hoc search query")
    print("  :docs          List all loaded documents")
    print("  :schema        Preview Section 8 JSON index schema")
    print("  :top <k>       Change number of displayed top-k results")
    print("  :save <file>   Save full Section 8 index schema to a JSON file")
    print("  :help          Show this help message")
    print("  :quit          Exit the engine\n")


def interactive_loop(corpus, top_k=5):
    """Run the interactive real-time CLI search loop (Milestone 3, FR9)."""
    t_build = time.perf_counter()
    index_schema = vsm_engine.build_index_schema(corpus)
    build_ms = (time.perf_counter() - t_build) * 1000.0
    vocab_size = len(index_schema["vocabulary"])

    print("Vector Space Information Retrieval Engine (First Principles)")
    print(
        f"Loaded Corpus: N = {len(corpus)} documents | "
        f"Vocabulary Dimension: D = {vocab_size} | Index Built in {build_ms:.3f} ms"
    )
    print_help()

    while True:
        try:
            raw_input_str = input("Query> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Retrieval Engine.")
            break

        cmd = raw_input_str.strip()
        if cmd in (":quit", ":exit", ":q"):
            print("Exiting Retrieval Engine.")
            break

        if cmd == ":help":
            print_help()
            continue

        if cmd == ":docs":
            for doc in index_schema["documents"]:
                print(f"  [{doc['doc_id']}] (tokens={doc['token_count']}): {doc['raw_text']}")
            print()
            continue

        if cmd == ":schema":
            preview = {
                "documents": index_schema["documents"][:2],
                "vocabulary": index_schema["vocabulary"],
                "inverted_index": {
                    k: index_schema["inverted_index"][k]
                    for k in list(index_schema["inverted_index"].keys())[:6]
                },
            }
            print(json.dumps(preview, indent=2) + "\n")
            continue

        if cmd.startswith(":top"):
            parts = cmd.split()
            if len(parts) != 2:
                print("Usage: :top <positive_integer>\n")
                continue
            try:
                new_k = int(parts[1])
                if new_k <= 0:
                    raise ValueError
                top_k = new_k
                print(f"Updated top-k to {top_k}.\n")
            except ValueError:
                print("Invalid value for k. Please provide a positive integer.\n")
            continue

        if cmd.startswith(":save"):
            parts = cmd.split(None, 1)
            if len(parts) != 2 or not parts[1].strip():
                print("Usage: :save <filepath.json>\n")
                continue
            out_path = parts[1].strip()
            try:
                vsm_engine.save_schema_to_json(index_schema, out_path)
                print(f"Saved Section 8 index schema to {out_path!r}.\n")
            except OSError as exc:
                print(f"Error saving file: {exc}\n")
            continue

        # Measure Dense search latency (NFR5)
        t0 = time.perf_counter()
        dense_results = vsm_engine.search(raw_input_str, corpus, k=top_k)
        dense_ms = (time.perf_counter() - t0) * 1000.0

        # Measure Inverted-Index search latency (NFR5)
        t1 = time.perf_counter()
        inv_results = vsm_engine.search_inverted_index(raw_input_str, index_schema, k=top_k)
        inv_ms = (time.perf_counter() - t1) * 1000.0

        # Verify dense and inverted-index agreement
        assert len(dense_results) == len(inv_results)
        for (d_id, d_sc), (i_id, i_sc) in zip(dense_results, inv_results):
            assert d_id == i_id and abs(d_sc - i_sc) < 1e-9

        # Full corpus score distribution (Milestone 3 evaluation)
        all_results = vsm_engine.search_inverted_index(
            raw_input_str, index_schema, k=len(corpus)
        )
        dist = compute_score_distribution([score for _, score in all_results])

        print("-" * 74)
        print(
            f"Results (top-{min(top_k, len(corpus))}) | "
            f"Dense Latency: {dense_ms:.3f} ms | "
            f"Inverted-Index Latency: {inv_ms:.3f} ms"
        )
        print(
            f"Score Distribution (N={len(corpus)}) -> "
            f"Max: {dist['max']:.4f}, Mean: {dist['mean']:.4f}, "
            f"Min: {dist['min']:.4f}, Matching Docs (Score > 0): {dist['non_zero']}/{len(corpus)}"
        )
        print("-" * 74)
        if not inv_results:
            print("  (No documents in corpus.)")
        else:
            for rank, (doc_id, score) in enumerate(inv_results, start=1):
                print(f"  #{rank} | doc_id={doc_id:<2d} | Score={score:.4f} | {corpus[doc_id]}")
        print()


def main():
    corpus = list(DEFAULT_CORPUS)
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        try:
            corpus = load_corpus_from_file(filepath)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            sys.stderr.write(f"Failed to load corpus from '{filepath}': {exc}\n")
            sys.exit(1)

    run_worked_example()
    interactive_loop(corpus)


if __name__ == "__main__":
    main()
