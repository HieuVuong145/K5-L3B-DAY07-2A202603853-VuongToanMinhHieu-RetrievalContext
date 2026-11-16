"""Unit tests for Checkpoint 3: Lexical Retriever and Hybrid RRF Retriever."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

from agentic import Retriever
from src.lexical_retriever import BM25LexicalRetriever
from src.hybrid_retriever import HybridRRFRetriever


class Checkpoint3RetrievalTests(unittest.TestCase):
    def test_lexical_retriever_protocol(self):
        retriever = BM25LexicalRetriever()
        self.assertTrue(hasattr(retriever, "search"))
        self.assertTrue(callable(getattr(retriever, "search")))

        res = retriever.search("tín chỉ học tập", top_k=2)
        self.assertLessEqual(len(res), 2)
        if res:
            self.assertIn("id", res[0])
            self.assertIn("text", res[0])
            self.assertIn("score", res[0])

    def test_hybrid_rrf_retriever_fusion(self):
        class MockDense:
            def search(self, query, *, filters, top_k):
                return [
                    {"id": "doc_A", "text": "A", "source_id": "s1", "source": "s.pdf", "page": 1, "metadata": {}, "score": 0.9},
                    {"id": "doc_B", "text": "B", "source_id": "s1", "source": "s.pdf", "page": 2, "metadata": {}, "score": 0.8},
                ]

        class MockLexical:
            def search(self, query, *, filters, top_k):
                return [
                    {"id": "doc_B", "text": "B", "source_id": "s1", "source": "s.pdf", "page": 2, "metadata": {}, "score": 15.0},
                    {"id": "doc_C", "text": "C", "source_id": "s1", "source": "s.pdf", "page": 3, "metadata": {}, "score": 10.0},
                ]

        hybrid = HybridRRFRetriever(dense_retriever=MockDense(), lexical_retriever=MockLexical(), candidate_k=5, rrf_k=60)
        res = hybrid.search("test", top_k=3)
        self.assertEqual(len(res), 3)

        # doc_B is present in both dense (rank 2) and lexical (rank 1):
        # rrf_score = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ~ 0.0161 + 0.0164 ~ 0.0325
        # doc_A is only in dense (rank 1): 1/61 ~ 0.0164
        # doc_C is only in lexical (rank 2): 1/62 ~ 0.0161
        # Therefore, doc_B must be rank 1!
        self.assertEqual(res[0]["id"], "doc_B")
        self.assertEqual(res[1]["id"], "doc_A")
        self.assertEqual(res[2]["id"], "doc_C")


if __name__ == "__main__":
    unittest.main()
