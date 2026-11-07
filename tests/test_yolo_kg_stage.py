"""Offline, network-free tests. Run from the extracted project root with -m unittest."""
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "yolo_kg_stage.py"
spec = importlib.util.spec_from_file_location("yolo_kg_stage", SOURCE)
kg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kg)


class KnowledgeGraphTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        kg.ROOT = self.root
        kg.SOURCE = self.root / "data/datasets/all_chunks_v3.jsonl"
        kg.OUT = self.root / "results/neo4j/knowledge_graph_v1"
        kg.CACHE = kg.OUT / "extractions"
        kg.REVIEW = kg.OUT / "triples_review.csv"
        kg.SOURCE.parent.mkdir(parents=True)
        self.chunk = {
            "chunk_id": "YOLOv9_1", "paper_id": "YOLOv9", "page": 5,
            "text": "We propose programmable gradient information (PGI) to solve the information bottleneck problem."
        }
        kg.SOURCE.write_text(json.dumps(self.chunk) + "\n", encoding="utf-8")
        self.candidate = {
            "head_type": "Paper", "head_name": "YOLOv9", "relation": "PROPOSES",
            "tail_type": "Method", "tail_name": "PGI",
            "quote": "We propose programmable gradient information (PGI) to solve the information bottleneck problem."
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_quote_validation_and_rejection(self):
        obj, err = kg.valid_candidate(self.candidate, self.chunk)
        self.assertIsNone(err)
        self.assertEqual(obj["page"], 5)
        hallucination = dict(self.candidate, quote="This paper achieves absolutely perfect accuracy every time.")
        self.assertEqual(kg.valid_candidate(hallucination, self.chunk)[1], "quote_not_exact_substring")
        bad_rel = dict(self.candidate, relation="INVENTED_LINK")
        self.assertEqual(kg.valid_candidate(bad_rel, self.chunk)[1], "invalid_schema")

    def test_manual_approval_gate(self):
        kg.CACHE.mkdir(parents=True)
        obj, err = kg.valid_candidate(self.candidate, self.chunk)
        self.assertIsNone(err)
        obj["llm_model"] = "fixture-model"
        kg.cache_file(self.chunk["chunk_id"]).write_text(
            json.dumps({"candidates": [obj]}), encoding="utf-8"
        )
        kg.cmd_review(SimpleNamespace())
        self.assertEqual(kg.get_reviewed_triples(), [])
        with kg.REVIEW.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        rows[0]["approved"] = "1"
        with kg.REVIEW.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=kg.FIELDS)
            w.writeheader()
            w.writerows(rows)
        self.assertEqual(len(kg.get_reviewed_triples()), 1)
        kg.cmd_review(SimpleNamespace())
        self.assertEqual(len(kg.get_reviewed_triples()), 1)
        kg.cmd_ingest(SimpleNamespace(dry_run=True))

    def test_key_canonicalization(self):
        self.assertEqual(kg.key_for("Model", "YOLOv9  "), kg.key_for("Model", " yolov9"))
        self.assertNotEqual(kg.key_for("Paper", "YOLOv9"), kg.key_for("Model", "YOLOv9"))


if __name__ == "__main__":
    unittest.main()
