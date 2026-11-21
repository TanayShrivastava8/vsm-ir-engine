"""
vsm_engine.py
Design and Implementation of a Vector Space Information Retrieval Engine
from First Principles.

Madhav Institute of Technology & Science, Gwalior
B.Tech - Mathematics and Computing
Team: Tanay Shrivastava (BTMC26O1138), Tanay Jaiswal (BTMC26O1137)
Guide: Monika Vyas

Strictly adheres to the Milestone 1 Report specification:
- Allowed standard-library modules only: collections, heapq, json, math, re, string
- Section 6.3 exact module signatures, preconditions, and postconditions
- Section 7 Program Invariants I1-I8 and Section 4.4 Edge Cases
- Section 8 Data & Index Schema (Dense M2-M3, Sparse & Inverted Index M4)
"""

import collections
import heapq
import json
import math
import re
import string

_PUNCT_TRANSLATOR = str.maketrans("", "", string.punctuation)
_PUNCT_SET = set(string.punctuation)


def _validate_real_number(val, func_name):
    """Validate that val is a finite int/float (excluding bool)."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise TypeError(
            f"{func_name} expected numeric vector entries, got '{type(val).__name__}'."
        )
    if math.isnan(val) or math.isinf(val):
        raise ValueError(f"{func_name} does not allow NaN or infinite values.")


# ==============================================================================
# SECTION 6.3: CORE MODULES AND INTERFACES (MILESTONE 2)
# ==============================================================================


def normalize_text(text):
    """
    FR2: Normalize text by converting to lowercase (case-folding), removing
    punctuation, collapsing extra whitespace, and splitting into tokens.

    Signature: str -> list[str]
    Precondition: text is a str (Invariant I6).
    Postcondition: tokens are non-empty lowercase strings without punctuation (Invariant I2).
    """
    if not isinstance(text, str):
        raise TypeError(
            f"Invariant I6 violated: normalize_text expected 'str', got '{type(text).__name__}'."
        )

    lowered = text.lower()
    no_punct = lowered.translate(_PUNCT_TRANSLATOR)
    collapsed = re.sub(r"\s+", " ", no_punct).strip()
    tokens = collapsed.split(" ") if collapsed else []

    # Postcondition & Invariant I2 verification
    assert isinstance(tokens, list), "Postcondition failed: output must be a list."
    assert all(
        isinstance(t, str)
        and len(t) > 0
        and t == t.lower()
        and not any(ch in _PUNCT_SET or ch.isspace() for ch in t)
        for t in tokens
    ), "Invariant I2 violated: tokens must be non-empty lowercase strings without punctuation."

    return tokens


def build_vocabulary(docs):
    """
    FR3: Build vocabulary V: the set of all unique tokens in the corpus,
    sorted alphabetically, each mapped to a contiguous index 0 .. D-1.

    Signature: list[list[str]] -> dict[str, int]
    Precondition: docs is a list of normalized token lists (list[list[str]]).
    Postcondition: indices are unique and run 0 .. D-1 in sorted-word order (Invariant I3).
    """
    if not isinstance(docs, list):
        raise TypeError(
            f"build_vocabulary expected 'list[list[str]]', got '{type(docs).__name__}'."
        )

    unique_tokens = set()
    for doc_tokens in docs:
        if not isinstance(doc_tokens, list):
            raise TypeError(
                "build_vocabulary expected each document to be a list of token strings."
            )
        for token in doc_tokens:
            if not isinstance(token, str):
                raise TypeError(
                    f"build_vocabulary expected token of type 'str', got '{type(token).__name__}'."
                )
            if (
                len(token) == 0
                or token != token.lower()
                or any(ch in _PUNCT_SET or ch.isspace() for ch in token)
            ):
                raise ValueError(
                    f"Token '{token}' violates Invariant I2 (must be non-empty lowercase without punctuation)."
                )
            unique_tokens.add(token)

    sorted_terms = sorted(unique_tokens)
    vocab = {term: idx for idx, term in enumerate(sorted_terms)}

    # Postcondition & Invariant I3 verification
    d_dim = len(vocab)
    assert list(vocab.keys()) == sorted_terms, (
        "Invariant I3 violated: vocabulary keys must be in sorted alphabetical order."
    )
    assert list(vocab.values()) == list(range(d_dim)), (
        "Invariant I3 violated: vocabulary indices must be unique and contiguous 0..D-1."
    )

    return vocab


def vectorize(tokens, vocab):
    """
    FR4 & FR6: Convert a list of tokens into a normalized term-frequency (TF)
    vector of length D = |V|, where d_i = count(t_i, d) / |T(d)|.
    Words not in V are ignored without modifying V.

    Signature: list[str], dict[str, int] -> list[float]
    Postcondition: length equals D (I1); all values >= 0; values sum to 1 when
    the token list is non-empty and all tokens belong to V (I4).
    """
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        raise TypeError("vectorize expected 'tokens' to be list[str].")
    if not isinstance(vocab, dict):
        raise TypeError("vectorize expected 'vocab' to be dict[str, int].")

    d_dim = len(vocab)
    vec = [0.0] * d_dim
    total_tokens = len(tokens)

    if total_tokens == 0 or d_dim == 0:
        assert len(vec) == d_dim, "Invariant I1 violated: vector length must equal D."
        return vec

    counts = collections.Counter(tokens)
    for term, count in counts.items():
        if term in vocab:
            idx = vocab[term]
            if isinstance(idx, bool) or not isinstance(idx, int) or not (0 <= idx < d_dim):
                raise ValueError(f"Invalid vocabulary index '{idx}' for term '{term}'.")
            vec[idx] = float(count) / float(total_tokens)

    # Postconditions & Invariants I1, I4 verification
    assert len(vec) == d_dim, "Invariant I1 violated: vector length must equal D = |V|."
    assert all(v >= 0.0 for v in vec), "Invariant I4 violated: TF values must be >= 0."
    if total_tokens > 0 and all(t in vocab for t in tokens):
        assert math.isclose(sum(vec), 1.0, rel_tol=1e-9, abs_tol=1e-9), (
            "Invariant I4 violated: TF values for a non-empty in-vocabulary document must sum to 1."
        )

    return vec


def dot_product(a, b):
    """
    FR5: Compute dot product: a . b = sum_{i=0}^{D-1} a_i * b_i.

    Signature: list[float], list[float] -> float
    Precondition: len(a) == len(b).
    """
    if not isinstance(a, list) or not isinstance(b, list):
        raise TypeError("dot_product requires two lists of numbers.")
    if len(a) != len(b):
        raise ValueError(
            f"Precondition failed in dot_product: len(a) ({len(a)}) != len(b) ({len(b)})."
        )

    total = 0.0
    for x, y in zip(a, b):
        _validate_real_number(x, "dot_product")
        _validate_real_number(y, "dot_product")
        total += float(x) * float(y)

    return float(total)


def euclidean_norm(a):
    """
    FR5: Compute Euclidean (L2) norm: ||a|| = sqrt(sum_{i=0}^{D-1} a_i^2).

    Signature: list[float] -> float
    Postcondition: result >= 0.0.
    """
    if not isinstance(a, list):
        raise TypeError("euclidean_norm requires a list of numbers.")

    norm = math.sqrt(dot_product(a, a))
    assert norm >= 0.0, "Postcondition failed: Euclidean norm must be >= 0."
    return float(norm)


def cosine_similarity(a, b):
    """
    FR5 & NFR3: Compute cosine similarity: (a . b) / (||a|| * ||b||).
    Returns 0.0 if either norm is 0 and clamps floating-point drift to [0.0, 1.0].

    Signature: list[float], list[float] -> float
    Precondition: len(a) == len(b), elements >= 0 for TF vectors.
    Postcondition: result in [0.0, 1.0] (Invariant I5).
    """
    if not isinstance(a, list) or not isinstance(b, list):
        raise TypeError("cosine_similarity requires two lists of numbers.")
    if len(a) != len(b):
        raise ValueError(
            f"Precondition failed in cosine_similarity: len(a) ({len(a)}) != len(b) ({len(b)})."
        )
    for x in a:
        _validate_real_number(x, "cosine_similarity")
        if x < 0.0:
            raise ValueError("Invariant I4 violated: TF vector entries cannot be negative.")
    for y in b:
        _validate_real_number(y, "cosine_similarity")
        if y < 0.0:
            raise ValueError("Invariant I4 violated: TF vector entries cannot be negative.")

    norm_a = euclidean_norm(a)
    norm_b = euclidean_norm(b)

    # Edge Case (Section 4.4): ||q|| = 0 or ||d|| = 0 -> Score = 0.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    raw_score = dot_product(a, b) / (norm_a * norm_b)

    # Edge Case (Section 4.4): Clamp floating-point drift (e.g., 1.0000000002) to [0.0, 1.0]
    score = max(0.0, min(1.0, raw_score))

    assert 0.0 <= score <= 1.0, "Invariant I5 violated: cosine score must lie in [0, 1]."
    return float(score)


def search(query, corpus, k=10):
    """
    FR7 & NFR1: Score every document in corpus against query using dense TF
    vectors and return the top-k (doc_id, score) pairs ranked by score descending,
    breaking ties deterministically by doc_id ascending.

    Signature: str, list[str], int -> list[tuple[int, float]]
    Precondition: query is str, corpus is list[str] (I6), k is int.
    Postcondition: sorted by (-score, doc_id), at most k results (I7).
    """
    if not isinstance(query, str):
        raise TypeError(
            f"Invariant I6 violated: query must be 'str', got '{type(query).__name__}'."
        )
    if not isinstance(corpus, list):
        raise TypeError(
            f"Invariant I6 violated: corpus must be 'list[str]', got '{type(corpus).__name__}'."
        )
    for idx, doc in enumerate(corpus):
        if not isinstance(doc, str):
            raise TypeError(
                f"Invariant I6 violated: document at index {idx} must be 'str', "
                f"got '{type(doc).__name__}'."
            )
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"Parameter 'k' must be 'int', got '{type(k).__name__}'.")

    if k <= 0 or len(corpus) == 0:
        return []

    tokenized_docs = [normalize_text(doc) for doc in corpus]
    vocab = build_vocabulary(tokenized_docs)
    vocab_snapshot = dict(vocab)

    query_tokens = normalize_text(query)
    query_vec = vectorize(query_tokens, vocab)

    # Invariant I8: Query never modifies vocabulary V
    assert vocab == vocab_snapshot, (
        "Invariant I8 violated: query processing must not modify vocabulary V."
    )

    scored = []
    for doc_id, doc_tokens in enumerate(tokenized_docs):
        doc_vec = vectorize(doc_tokens, vocab)
        score = cosine_similarity(query_vec, doc_vec)
        scored.append((doc_id, score))

    # Deterministic ranking: score descending, ties broken by doc_id ascending (I7)
    ranked = heapq.nsmallest(k, scored, key=lambda item: (-item[1], item[0]))

    # Postcondition & Invariant I7 verification
    assert len(ranked) <= min(k, len(corpus)), "Postcondition failed: too many results returned."
    for i in range(len(ranked) - 1):
        curr_id, curr_score = ranked[i]
        next_id, next_score = ranked[i + 1]
        assert (curr_score > next_score) or (
            curr_score == next_score and curr_id < next_id
        ), "Invariant I7 violated: ranking is not deterministically ordered."

    return ranked


# ==============================================================================
# SECTION 8: DATA & INDEX SCHEMA AND SPARSE / INVERTED INDEX (MILESTONE 4)
# ==============================================================================


def vectorize_sparse(tokens, vocab):
    """
    Section 8.3 (Milestone 4): Convert a token list into a sparse TF vector
    dict[term_index, tf] storing only strictly non-zero entries.
    """
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        raise TypeError("vectorize_sparse expected 'tokens' to be list[str].")
    if not isinstance(vocab, dict):
        raise TypeError("vectorize_sparse expected 'vocab' to be dict[str, int].")

    total_tokens = len(tokens)
    if total_tokens == 0 or len(vocab) == 0:
        return {}

    counts = collections.Counter(tokens)
    sparse_vec = {}
    for term, count in counts.items():
        if term in vocab and count > 0:
            sparse_vec[vocab[term]] = float(count) / float(total_tokens)

    assert all(v > 0.0 for v in sparse_vec.values()), (
        "Sparse vector must store only strictly positive entries."
    )
    return sparse_vec


def sparse_euclidean_norm(sparse_vec):
    """Compute Euclidean (L2) norm of a sparse vector dict[term_index, tf]."""
    if not isinstance(sparse_vec, dict):
        raise TypeError("sparse_euclidean_norm expected a dictionary.")

    sq_sum = 0.0
    for val in sparse_vec.values():
        _validate_real_number(val, "sparse_euclidean_norm")
        sq_sum += float(val) * float(val)

    norm = math.sqrt(sq_sum)
    assert norm >= 0.0, "Postcondition failed: Euclidean norm must be >= 0."
    return float(norm)


def build_index_schema(raw_documents, titles=None):
    """
    Sections 8.1 - 8.4: Build the complete Data and Index Schema:
    - documents: list of {doc_id, title, raw_text, token_count}
    - vocabulary: {term: index}
    - sparse_vectors: {doc_id: {term_index: tf}}
    - doc_norms: {doc_id: float} (precomputed once and stored per Section 8.4)
    - inverted_index: {term: [[doc_id, tf], ...]}
    """
    if not isinstance(raw_documents, list):
        raise TypeError(
            f"Invariant I6 violated: raw_documents must be 'list[str]', "
            f"got '{type(raw_documents).__name__}'."
        )
    for idx, doc in enumerate(raw_documents):
        if not isinstance(doc, str):
            raise TypeError(
                f"Invariant I6 violated: document at index {idx} must be 'str', "
                f"got '{type(doc).__name__}'."
            )
    if titles is not None:
        if not isinstance(titles, list) or not all(isinstance(t, str) for t in titles):
            raise TypeError("titles must be None or list[str].")

    tokenized_corpus = [normalize_text(doc) for doc in raw_documents]
    vocab = build_vocabulary(tokenized_corpus)
    index_to_term = {idx: term for term, idx in vocab.items()}

    documents_store = []
    sparse_vectors = {}
    doc_norms = {}
    inverted_index = {term: [] for term in vocab}

    for doc_id, (raw_text, tokens) in enumerate(zip(raw_documents, tokenized_corpus)):
        title = titles[doc_id] if (titles is not None and doc_id < len(titles)) else f"doc{doc_id}"
        documents_store.append(
            {
                "doc_id": doc_id,
                "title": title,
                "raw_text": raw_text,
                "token_count": len(tokens),
            }
        )

        sp_vec = vectorize_sparse(tokens, vocab)
        sparse_vectors[doc_id] = sp_vec
        doc_norms[doc_id] = sparse_euclidean_norm(sp_vec)

        for term_idx in sorted(sp_vec.keys()):
            term = index_to_term[term_idx]
            inverted_index[term].append([doc_id, sp_vec[term_idx]])

    return {
        "documents": documents_store,
        "vocabulary": vocab,
        "sparse_vectors": sparse_vectors,
        "doc_norms": doc_norms,
        "inverted_index": inverted_index,
    }


def search_inverted_index(query, index_schema, k=10):
    """
    Section 8.4 & Section 9 (Milestone 4): Score a query using the inverted index.

    Traverses only the posting lists of in-vocabulary query terms (no O(N * D)
    full-corpus scan and no O(D) reverse-vocabulary rebuild), while guaranteeing
    identical scores and deterministic tie-breaking to dense search().
    """
    if not isinstance(query, str):
        raise TypeError(
            f"Invariant I6 violated: query must be 'str', got '{type(query).__name__}'."
        )
    if not isinstance(index_schema, dict):
        raise TypeError("index_schema must be a dictionary.")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"Parameter 'k' must be 'int', got '{type(k).__name__}'.")

    documents = index_schema["documents"]
    vocab = index_schema["vocabulary"]
    inverted_index = index_schema["inverted_index"]
    doc_norms = index_schema["doc_norms"]

    num_docs = len(documents)
    target_count = min(k, num_docs)
    if target_count <= 0:
        return []

    query_tokens = normalize_text(query)
    total_q_tokens = len(query_tokens)
    if total_q_tokens == 0 or len(vocab) == 0:
        return [(doc_id, 0.0) for doc_id in range(target_count)]

    # O(|q|): count only in-vocabulary query terms without mutating vocab (Invariant I8)
    q_counts = collections.Counter(t for t in query_tokens if t in vocab)
    if not q_counts:
        return [(doc_id, 0.0) for doc_id in range(target_count)]

    q_term_tf = {
        term: float(cnt) / float(total_q_tokens) for term, cnt in q_counts.items()
    }
    q_norm = math.sqrt(sum(tf * tf for tf in q_term_tf.values()))
    if q_norm == 0.0:
        return [(doc_id, 0.0) for doc_id in range(target_count)]

    # O(sum of posting-list lengths for query terms): accumulate dot products
    dot_accumulators = collections.defaultdict(float)
    for term, q_tf in q_term_tf.items():
        for doc_id, d_tf in inverted_index.get(term, []):
            dot_accumulators[int(doc_id)] += q_tf * float(d_tf)

    # Score only matching documents
    positive_scored = []
    positive_ids = set()
    for doc_id, dot_val in dot_accumulators.items():
        d_norm = float(
            doc_norms[doc_id] if doc_id in doc_norms else doc_norms[str(doc_id)]
        )
        if d_norm > 0.0 and dot_val > 0.0:
            score = max(0.0, min(1.0, dot_val / (q_norm * d_norm)))
            assert 0.0 <= score <= 1.0, "Invariant I5 violated: cosine score must lie in [0, 1]."
            positive_scored.append((doc_id, score))
            positive_ids.add(doc_id)

    ranked = heapq.nsmallest(
        target_count, positive_scored, key=lambda item: (-item[1], item[0])
    )

    # If fewer than k documents matched, fill remaining slots with 0.0 in doc_id order (I7)
    if len(ranked) < target_count:
        for doc_id in range(num_docs):
            if doc_id not in positive_ids:
                ranked.append((doc_id, 0.0))
                if len(ranked) == target_count:
                    break

    # Invariant I7 verification
    assert len(ranked) <= target_count, "Postcondition failed: too many results returned."
    for i in range(len(ranked) - 1):
        curr_id, curr_score = ranked[i]
        next_id, next_score = ranked[i + 1]
        assert (curr_score > next_score) or (
            curr_score == next_score and curr_id < next_id
        ), "Invariant I7 violated: ranking is not deterministically ordered."

    return ranked


def schema_to_json_str(index_schema):
    """Serialize a Section 8 Data and Index Schema dictionary to a JSON string."""
    return json.dumps(index_schema, indent=2)


def schema_from_json_str(json_str):
    """
    Deserialize a Section 8 JSON string into an index schema dictionary.
    Restores integer keys for doc_norms and sparse_vectors, and safely
    reconstructs them if loading a minimal Section 8.4 snippet.
    """
    data = json.loads(json_str)
    vocab = data["vocabulary"]
    documents = data["documents"]
    inverted_index = data["inverted_index"]

    if "sparse_vectors" in data:
        data["sparse_vectors"] = {
            int(doc_id): {int(t_idx): float(tf) for t_idx, tf in vec.items()}
            for doc_id, vec in data["sparse_vectors"].items()
        }
    else:
        sparse_vectors = {int(doc["doc_id"]): {} for doc in documents}
        for term, postings in inverted_index.items():
            if term in vocab:
                t_idx = vocab[term]
                for doc_id, tf in postings:
                    sparse_vectors.setdefault(int(doc_id), {})[t_idx] = float(tf)
        data["sparse_vectors"] = sparse_vectors

    if "doc_norms" in data:
        data["doc_norms"] = {int(k): float(v) for k, v in data["doc_norms"].items()}
    else:
        data["doc_norms"] = {
            doc_id: sparse_euclidean_norm(sp_vec)
            for doc_id, sp_vec in data["sparse_vectors"].items()
        }

    return data


def save_schema_to_json(index_schema, filepath):
    """Save the Section 8 Data and Index Schema to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(schema_to_json_str(index_schema))


def load_schema_from_json(filepath):
    """Load the Section 8 Data and Index Schema from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return schema_from_json_str(f.read())


if __name__ == "__main__":
    # Milestone 2 Prototype Verification (Section 4.5 Worked Example)
    worked_corpus = ["the cat sat on the mat", "the dog sat on the log"]
    worked_query = "cat mat"
    worked_results = search(worked_query, worked_corpus, k=2)
    print("Section 4.5 Worked Example:")
    for rank, (d_id, sc) in enumerate(worked_results, start=1):
        print(f"  Rank {rank}: doc_id={d_id} | score={sc:.4f} | text={worked_corpus[d_id]!r}")
