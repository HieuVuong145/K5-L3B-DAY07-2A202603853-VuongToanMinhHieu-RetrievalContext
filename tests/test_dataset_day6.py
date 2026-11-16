"""Validate Day 6 converted chunks, benchmark questions, and cases."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Day6DatasetTests(unittest.TestCase):
    def setUp(self):
        self.chunks_path = ROOT / 'data/converted/chunks.json'
        self.questions_path = ROOT / 'data/converted/questions.json'
        self.cases_path = ROOT / 'data/converted/cases.json'

    def test_files_exist(self):
        self.assertTrue(self.chunks_path.exists(), "converted chunks.json must exist")
        self.assertTrue(self.questions_path.exists(), "converted questions.json must exist")
        self.assertTrue(self.cases_path.exists(), "converted cases.json must exist")

    def test_chunk_counts_and_schema(self):
        chunks = json.loads(self.chunks_path.read_text(encoding='utf-8'))
        self.assertEqual(len(chunks), 277, "Total chunks should be 277")
        for c in chunks:
            self.assertIn('id', c)
            self.assertIn('source_id', c)
            self.assertIn('source', c)
            self.assertIn('page', c)
            self.assertIn('text', c)
            self.assertIn('metadata', c)

    def test_questions_gold_resolves_in_chunks(self):
        chunks = json.loads(self.chunks_path.read_text(encoding='utf-8'))
        questions = json.loads(self.questions_path.read_text(encoding='utf-8'))
        self.assertEqual(len(questions), 24, "Should have 24 answerable questions")
        
        for q in questions:
            for g in q['gold']:
                matched = any(
                    c['source_id'] == g['source_id'] and
                    c['page'] == g['page'] and
                    g['evidence'] in c['text']
                    for c in chunks
                )
                self.assertTrue(
                    matched,
                    f"Question {q['id']} gold evidence on page {g['page']} must resolve verbatim in chunks"
                )

    def test_cases_structure_and_split(self):
        cases = json.loads(self.cases_path.read_text(encoding='utf-8'))
        self.assertEqual(len(cases), 30, "Total cases must be 30")

        dev_cases = [c for c in cases if c['split'] == 'dev']
        test_cases = [c for c in cases if c['split'] == 'test']

        self.assertEqual(len(dev_cases), 20, "Dev set must have 20 cases")
        self.assertEqual(len(test_cases), 10, "Test set must have 10 cases")

        dev_answerable = [c for c in dev_cases if c['answerable']]
        dev_unanswerable = [c for c in dev_cases if not c['answerable']]
        test_answerable = [c for c in test_cases if c['answerable']]
        test_unanswerable = [c for c in test_cases if not c['answerable']]

        self.assertEqual(len(dev_answerable), 17)
        self.assertEqual(len(dev_unanswerable), 3)
        self.assertEqual(len(test_answerable), 7)
        self.assertEqual(len(test_unanswerable), 3)


if __name__ == '__main__':
    unittest.main()
