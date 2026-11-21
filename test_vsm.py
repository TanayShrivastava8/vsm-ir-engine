"""
test_vsm.py
Comprehensive unit-test suite covering:
- Section 4.5 Worked Example
- Functional Requirements FR1-FR8 & Non-Functional Requirements NFR1-NFR4
- Program Invariants I1-I8 (Section 7)
- Edge Cases (Section 4.4)
- Section 8 Data & Index Schema, JSON Round-Trip, and Dense vs. Inverted-Index Equivalence
- Stress Testing on synthetic corpora (Milestone 4)

Allowed standard-library modules used: math, unittest
"""

import math
import unittest

import vsm_engine


class TestVSMRetrievalEngine(unittest.TestCase):
    def test_worked_example_section_4_5(self):
        """Section 4.5: Verify exact vocabulary, cosine scores (0.5 and 0.0), and ranking."""
        corpus = ["the cat sat on the mat", "the dog sat on the log"]
        query = "cat mat"

        tokenized = [vsm_engine.normalize_text(d) for d in corpus]
        vocab = vsm_engine.build_vocabulary(tokenized)
        expected_vocab = {
            "cat": 0,
            "dog": 1,
            "log": 2,
            "mat": 3,
            "on": 4,
            "sat": 5,
            "the": 6,
        }
        self.assertEqual(vocab, expected_vocab)
        self.assertEqual(len(vocab), 7)

        results = vsm_engine.search(query, corpus, k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], 0)
        self.assertAlmostEqual(results[0][1], 0.5, places=9)
        self.assertEqual(results[1][0], 1)
        self.assertAlmostEqual(results[1][1], 0.0, places=9)

        schema = vsm_engine.build_index_schema(corpus)
        inv_results = vsm_engine.search_inverted_index(query, schema, k=2)
        self.assertEqual(inv_results[0][0], 0)
        self.assertAlmostEqual(inv_results[0][1], 0.5, places=9)
        self.assertEqual(inv_results[1][0], 1)
        self.assertAlmostEqual(inv_results[1][1], 0.0, places=9)

    def test_normalize_text_fr2_and_invariants_i2_i6(self):
        """FR2, I2, I6: Case-folding, punctuation removal, whitespace collapse, TypeError on non-str."""
        raw = "  The   CAT, sat!! On - the\n\tMAT?  "
        tokens = vsm_engine.normalize_text(raw)
        self.assertEqual(tokens, ["the", "cat", "sat", "on", "the", "mat"])

        self.assertEqual(vsm_engine.normalize_text("don't!"), ["dont"])
        self.assertEqual(vsm_engine.normalize_text(""), [])
        self.assertEqual(vsm_engine.normalize_text("   \t\n  "), [])
        self.assertEqual(vsm_engine.normalize_text("!!! ??? ... ---"), [])

        for invalid_input in [None, 123, 3.14, True, ["the", "cat"], {"text": "hi"}]:
            with self.assertRaises(TypeError):
                vsm_engine.normalize_text(invalid_input)

    def test_build_vocabulary_fr3_and_invariant_i3(self):
        """FR3, I3: Unique contiguous indices 0..D-1 in sorted alphabetical order."""
        docs = [["zebra", "apple", "mango"], ["apple", "banana"]]
        vocab = vsm_engine.build_vocabulary(docs)
        self.assertEqual(vocab, {"apple": 0, "banana": 1, "mango": 2, "zebra": 3})

        self.assertEqual(vsm_engine.build_vocabulary([]), {})
        self.assertEqual(vsm_engine.build_vocabulary([[], []]), {})

        with self.assertRaises(TypeError):
            vsm_engine.build_vocabulary("not a list")
        with self.assertRaises(TypeError):
            vsm_engine.build_vocabulary(["not a list of lists"])
        with self.assertRaises(ValueError):
            vsm_engine.build_vocabulary([["UPPER"]])
        with self.assertRaises(ValueError):
            vsm_engine.build_vocabulary([["punct!"]])
        with self.assertRaises(ValueError):
            vsm_engine.build_vocabulary([[""]])

    def test_vectorize_fr4_fr6_and_invariants_i1_i4(self):
        """FR4, FR6, I1, I4: Vector length D, entries >= 0, sum == 1.0 for non-empty in-vocab doc."""
        vocab = {"cat": 0, "mat": 1, "the": 2}
        vec = vsm_engine.vectorize(["the", "cat", "the", "mat"], vocab)
        self.assertEqual(len(vec), 3)
        self.assertAlmostEqual(vec[0], 0.25, places=9)
        self.assertAlmostEqual(vec[1], 0.25, places=9)
        self.assertAlmostEqual(vec[2], 0.50, places=9)
        self.assertAlmostEqual(sum(vec), 1.0, places=9)

        # Empty tokens -> zero vector of length D
        self.assertEqual(vsm_engine.vectorize([], vocab), [0.0, 0.0, 0.0])
        # Empty vocabulary -> empty vector
        self.assertEqual(vsm_engine.vectorize(["cat"], {}), [])
        # Out-of-vocabulary word ignored without mutating vocab
        oov_vec = vsm_engine.vectorize(["cat", "unseen"], vocab)
        self.assertEqual(oov_vec, [0.5, 0.0, 0.0])

    def test_vector_operations_fr5(self):
        """FR5: Verify dot_product, euclidean_norm, and cosine_similarity preconditions."""
        self.assertAlmostEqual(
            vsm_engine.dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0, places=9
        )
        self.assertEqual(vsm_engine.dot_product([], []), 0.0)
        self.assertAlmostEqual(vsm_engine.euclidean_norm([3.0, 4.0, 0.0]), 5.0, places=9)
        self.assertEqual(vsm_engine.euclidean_norm([0.0, 0.0, 0.0]), 0.0)
        self.assertEqual(vsm_engine.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

        with self.assertRaises(ValueError):
            vsm_engine.dot_product([1.0, 2.0], [1.0])
        with self.assertRaises(ValueError):
            vsm_engine.cosine_similarity([1.0, 2.0], [1.0])
        with self.assertRaises(ValueError):
            vsm_engine.cosine_similarity([-0.5, 0.5], [0.5, 0.5])

    def test_edge_cases_section_4_4_and_invariant_i5(self):
        """FR8, NFR3, I5: Zero norms, disjoint vocabs, OOV queries, empty corpus, clamping, TypeErrors."""
        corpus = ["the cat sat on the mat", "the dog sat on the log"]

        # 1. Empty query -> ||q|| = 0 -> Score = 0.0 without division by zero
        self.assertEqual(vsm_engine.search("", corpus, k=2), [(0, 0.0), (1, 0.0)])

        # 2. Punctuation-only query -> ||q|| = 0 -> Score = 0.0
        self.assertEqual(vsm_engine.search("!!! ???", corpus, k=2), [(0, 0.0), (1, 0.0)])

        # 3. Out-of-vocabulary query -> ||q|| = 0 -> Score = 0.0
        self.assertEqual(
            vsm_engine.search("quantum relativity spacecraft", corpus, k=2),
            [(0, 0.0), (1, 0.0)],
        )

        # 4. Empty document in corpus -> ||d|| = 0 -> Score = 0.0
        res_empty_doc = vsm_engine.search("cat", ["", "cat sat"], k=2)
        self.assertEqual(res_empty_doc[0][0], 1)
        self.assertGreater(res_empty_doc[0][1], 0.0)
        self.assertEqual(res_empty_doc[1], (0, 0.0))

        # 5. All-empty documents and empty corpus
        self.assertEqual(vsm_engine.search("cat", ["", ""], k=2), [(0, 0.0), (1, 0.0)])
        self.assertEqual(vsm_engine.search("cat", [], k=5), [])
        self.assertEqual(vsm_engine.search("cat", corpus, k=0), [])
        self.assertEqual(len(vsm_engine.search("cat", corpus, k=100)), 2)

        # 6. Identical direction vectors with potential float drift -> clamped to [0.0, 1.0]
        v = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        sim = vsm_engine.cosine_similarity(v, v)
        self.assertGreaterEqual(sim, 0.0)
        self.assertLessEqual(sim, 1.0)
        self.assertAlmostEqual(sim, 1.0, places=9)

        # 7. Non-string inputs -> TypeError (I6)
        with self.assertRaises(TypeError):
            vsm_engine.search(123, corpus, k=2)
        with self.assertRaises(TypeError):
            vsm_engine.search("cat", ["valid", 456], k=2)
        with self.assertRaises(TypeError):
            vsm_engine.search("cat", corpus, k=True)

    def test_deterministic_tie_breaking_nfr1_and_invariant_i7(self):
        """NFR1, I7: Equal scores are deterministically broken by ascending doc_id."""
        corpus = ["alpha beta", "alpha beta", "alpha beta"]
        dense_res = vsm_engine.search("alpha", corpus, k=3)
        self.assertEqual([doc_id for doc_id, _ in dense_res], [0, 1, 2])

        schema = vsm_engine.build_index_schema(corpus)
        inv_res = vsm_engine.search_inverted_index("alpha", schema, k=3)
        self.assertEqual([doc_id for doc_id, _ in inv_res], [0, 1, 2])

    def test_query_never_modifies_vocabulary_invariant_i8(self):
        """I8: Unseen query words must never modify vocabulary V."""
        corpus = ["cat mat", "dog log"]
        tokenized = [vsm_engine.normalize_text(d) for d in corpus]
        vocab = vsm_engine.build_vocabulary(tokenized)
        vocab_before = dict(vocab)

        q_vec = vsm_engine.vectorize(["cat", "unseenword", "anotherword"], vocab)
        self.assertEqual(vocab, vocab_before)
        self.assertEqual(len(q_vec), 4)

        schema = vsm_engine.build_index_schema(corpus)
        schema_vocab_before = dict(schema["vocabulary"])
        vsm_engine.search_inverted_index("cat unseenword anotherword", schema, k=2)
        self.assertEqual(schema["vocabulary"], schema_vocab_before)

    def test_schema_json_roundtrip_and_inverted_index_equivalence(self):
        """Section 8: Verify schema structure, JSON round-trip, and dense vs. sparse equivalence."""
        corpus = [
            "the cat sat on the mat",
            "the dog sat on the log",
            "the cat and the dog sat together",
            "",
        ]
        schema = vsm_engine.build_index_schema(corpus)

        # Verify Section 8.4 posting entry: "cat" in doc 0 has TF 1/6
        cat_postings = schema["inverted_index"]["cat"]
        self.assertEqual(cat_postings[0][0], 0)
        self.assertAlmostEqual(cat_postings[0][1], 1.0 / 6.0, places=9)

        # Round-trip through JSON string
        json_text = vsm_engine.schema_to_json_str(schema)
        reloaded = vsm_engine.schema_from_json_str(json_text)

        for q in ["cat mat", "the sat", "dog", "unknownword", ""]:
            for k_val in [1, 2, 4, 10]:
                dense_res = vsm_engine.search(q, corpus, k=k_val)
                inv_res = vsm_engine.search_inverted_index(q, reloaded, k=k_val)
                self.assertEqual(len(dense_res), len(inv_res))
                for (id_d, sc_d), (id_i, sc_i) in zip(dense_res, inv_res):
                    self.assertEqual(id_d, id_i)
                    self.assertAlmostEqual(sc_d, sc_i, places=9)

        # Verify minimal Section 8.4 JSON snippet loads safely
        minimal_json = """
        {
          "documents": [{"doc_id": 0, "title": "doc0", "raw_text": "the cat sat on the mat", "token_count": 6}],
          "vocabulary": {"cat": 0, "dog": 1, "log": 2, "mat": 3, "on": 4, "sat": 5, "the": 6},
          "inverted_index": {"cat": [[0, 0.1667]], "mat": [[0, 0.1667]]}
        }
        """
        minimal_schema = vsm_engine.schema_from_json_str(minimal_json)
        min_res = vsm_engine.search_inverted_index("cat mat", minimal_schema, k=1)
        self.assertEqual(len(min_res), 1)
        self.assertEqual(min_res[0][0], 0)
        self.assertAlmostEqual(min_res[0][1], 1.0, places=6)

    def test_stress_equivalence_milestone_4(self):
        """Milestone 4 stress test: 200 synthetic docs must match identically across engines."""
        corpus = [
            " ".join(f"term{(doc_id * 17 + pos * 31) % 120}" for pos in range(15))
            for doc_id in range(200)
        ]
        schema = vsm_engine.build_index_schema(corpus)
        queries = [
            "term0 term15",
            "term42",
            "term5 term99 term119",
            "nonexistentterm",
            "",
        ]
        for query in queries:
            dense_res = vsm_engine.search(query, corpus, k=10)
            inv_res = vsm_engine.search_inverted_index(query, schema, k=10)
            self.assertEqual(len(dense_res), len(inv_res))
            for (d_id, d_sc), (i_id, i_sc) in zip(dense_res, inv_res):
                self.assertEqual(d_id, i_id)
                self.assertTrue(math.isclose(d_sc, i_sc, rel_tol=1e-9, abs_tol=1e-9))


if __name__ == "__main__":
    unittest.main(verbosity=2)
