"""Unit tests for Checkpoint 2: Dense Retriever and LLM Controller."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

from agentic import Controller, Retriever, run_agent
from src.dense_retriever import PgVectorDenseRetriever
from src.llm_controller import OpenRouterController


class Checkpoint2ProtocolTests(unittest.TestCase):
    def test_retriever_conforms_to_protocol(self):
        # Verify PgVectorDenseRetriever has callable search method
        self.assertTrue(hasattr(PgVectorDenseRetriever, "search"))
        self.assertTrue(callable(getattr(PgVectorDenseRetriever, "search")))

    def test_controller_conforms_to_protocol(self):
        # Verify OpenRouterController has callable decide method
        self.assertTrue(hasattr(OpenRouterController, "decide"))
        self.assertTrue(callable(getattr(OpenRouterController, "decide")))

    def test_agent_abstention_on_unanswerable_evidence(self):
        # Test that agent logic strictly abstains when controller returns abstain or empty citations
        class MockAbstainController:
            def decide(self, question, history):
                if not history:
                    return {"action": "search", "query": "tuition"}
                return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}

        class MockRetriever:
            def search(self, query, *, filters, top_k):
                return [{
                    "id": "c_unrelated",
                    "text": "Điều khoản về tổ chức giảng dạy.",
                    "source_id": "doc1",
                    "source": "doc1.pdf",
                    "page": 1,
                    "metadata": {},
                    "score": 0.3,
                }]

        res = run_agent("Học phí là bao nhiêu?", MockAbstainController(), MockRetriever(), max_searches=2)
        self.assertEqual(res["status"], "abstain")
        self.assertEqual(res["answer"], "Không đủ bằng chứng để trả lời.")
        self.assertEqual(res["citations"], [])

    def test_agent_answer_with_valid_citations(self):
        # Test that agent produces answered status when citations match context
        class MockAnswerController:
            def decide(self, question, history):
                if not history:
                    return {"action": "search", "query": "chuẩn đầu ra"}
                return {
                    "action": "answer",
                    "answer": "Chuẩn đầu ra là TOEIC 500 [c_toeic].",
                    "citations": ["c_toeic"],
                }

        class MockRetriever:
            def search(self, query, *, filters, top_k):
                return [{
                    "id": "c_toeic",
                    "text": "Chuẩn đầu ra ngoại ngữ là chứng chỉ TOEIC đạt từ 500 điểm.",
                    "source_id": "doc1",
                    "source": "doc1.pdf",
                    "page": 5,
                    "metadata": {},
                    "score": 0.95,
                }]

        res = run_agent("Chuẩn đầu ra là gì?", MockAnswerController(), MockRetriever(), max_searches=2)
        self.assertEqual(res["status"], "answered")
        self.assertIn("[c_toeic]", res["answer"])
        self.assertEqual(res["citations"], ["c_toeic"])


if __name__ == "__main__":
    unittest.main()
