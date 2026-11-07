"""Offline verification without user database, Milvus or cloud keys."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

PATH = Path(__file__).resolve().parents[1] / "scripts" / "yolo_graph_rag.py"
spec = importlib.util.spec_from_file_location("yolo_graph_rag", PATH)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


class GraphRAGOfflineTests(unittest.TestCase):
    def test_graph_query_is_static_and_bounded(self):
        for hops in (1, 2):
            query = g.HOP_QUERY.format(hops=hops)
            self.assertIn(f"[*1..{hops}]", query)
            self.assertIn("{key: $anchor_key}", query)
            self.assertIn("WITH DISTINCT r", query)
            self.assertIn("LIMIT $limit", query)

    def test_lexical_anchor_prefers_method(self):
        nodes = [
            {"key": "paper::yolov9", "name": "YOLOv9", "kind": "Paper"},
            {"key": "model::yolov9", "name": "YOLOv9", "kind": "Model"},
            {"key": "method::pgi", "name": "PGI", "kind": "Method"},
            {"key": "metric::ap", "name": "AP", "kind": "Metric"},
        ]
        chosen = g.heuristic_anchors("YOLOv9 的 PGI 有哪些分支？", nodes)
        self.assertEqual(chosen[0]["key"], "method::pgi")
        self.assertFalse(g.contains_name("AP50", "AP"))

    def test_manual_anchor_only_existing_node(self):
        nodes = [{"key": "method::pgi", "name": "PGI", "kind": "Method"}]
        selected, method = g.choose_anchors("foo", nodes, "PGI", "lexical", 2)
        self.assertEqual((selected[0]["key"], method), ("method::pgi", "explicit"))
        with self.assertRaises(RuntimeError):
            g.choose_anchors("foo", nodes, "GELAN", "lexical", 2)

    def test_deduplicate_semantic_relation_preserve_sources(self):
        base = dict(head_key="method::pgi", head_name="PGI", head_kind="Method",
                    relation="HAS_COMPONENT", tail_key="component::main branch",
                    tail_name="main branch", tail_kind="Component", depth_priority=0)
        rows = [
            dict(base, evidence_id="id-1", paper_id="YOLOv9", page=5, chunk_id="c1", quote="main branch 5"),
            dict(base, evidence_id="id-2", paper_id="YOLOv9", page=9, chunk_id="c2", quote="main branch 9"),
        ]
        facts = g.group_graph_facts(rows)
        self.assertEqual(len(facts), 1)
        self.assertEqual(len(facts[0]["evidence"]), 2)
        self.assertEqual({e["page"] for e in facts[0]["evidence"]}, {5, 9})
        self.assertIn("[G1]", g.graph_context(facts))

    def test_parse_vector_log_and_figure_priority(self):
        sample = """TEXT EVIDENCE:
[T1] YOLOv9 PDF p.5 E5=0.866 source=full_pdf_page_index
    This is a test snippet.
FIGURE EVIDENCE:
[F1] YOLOv9_T5 YOLOv9 Table 5 PDF p.9 RRF=0.08
[F2] YOLOv9_F3 YOLOv9 Figure 3 PDF p.5 RRF=0.07
Result JSON: /tmp/r.json
Image gallery: /tmp/r.html
"""
        parsed = g.parse_vector_stdout(sample)
        self.assertEqual(parsed["text"][0]["id"], "T1")
        self.assertIn("test snippet", parsed["text"][0]["snippet"])
        self.assertEqual(parsed["figure"][1]["id"], "F2")
        self.assertEqual(g.target_figure_id("请分析 YOLOv9 Figure 3 的结构"), "YOLOv9_F3")
        with tempfile.TemporaryDirectory() as tmp:
            old_root = g.ROOT
            try:
                g.ROOT = Path(tmp)
                png = g.ROOT / "data/visual_overrides/YOLOv9_F3.png"
                png.parent.mkdir(parents=True)
                png.write_bytes(b"fixture")
                choices = g.choose_figure_images("YOLOv9 Figure 3", parsed, 2)
                self.assertEqual(choices[0]["asset_id"], "YOLOv9_F3")
            finally:
                g.ROOT = old_root

    def test_read_only_pipeline_offline_fixture(self):
        class Record:
            def __init__(self, data):
                self.obj = data
            def data(self):
                return self.obj
        class FakeDB:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def verify_connectivity(self): pass
            def execute_query(self, query, **kwargs):
                if "RETURN n.key AS key" in query:
                    return [Record({"key": "method::pgi", "name": "PGI", "kind": "Method"})], None, []
                if "UNWIND relationships(p) AS r" in query:
                    return [Record({
                        "head_key": "method::pgi", "head_name": "PGI", "head_kind": "Method",
                        "relation": "HAS_COMPONENT", "tail_key": "component::main branch",
                        "tail_name": "main branch", "tail_kind": "Component",
                        "evidence_id": "example-1", "paper_id": "YOLOv9", "page": 5,
                        "chunk_id": "test-chunk", "quote": "main branch for inference",
                        "depth_priority": 0,
                    })], None, []
                raise AssertionError(query)
        old_driver, old_vector, old_out = g.driver, g.vector_evidence, g.OUT
        with tempfile.TemporaryDirectory() as tmp:
            try:
                g.driver = FakeDB
                g.vector_evidence = lambda question: {
                    "text": [{"id": "T1", "header": "YOLOv9 PDF p.5", "snippet": "PGI has three branches."}],
                    "figure": [{"id": "F1", "header": "YOLOv9_F3 Figure 3"}],
                    "gallery": None, "result_json": None,
                }
                g.OUT = Path(tmp)
                opts = SimpleNamespace(question="PGI 的分支有哪些？", mode="evidence", hops=2,
                                       anchor="PGI", anchor_mode="auto", max_anchors=2,
                                       fetch_limit=120, max_facts=24, vision=False, vision_count=2)
                g.run_ask(opts)
                outputs = list(Path(tmp).glob("*.json"))
                self.assertEqual(len(outputs), 1)
                import json
                result = json.loads(outputs[0].read_text(encoding="utf-8"))
                self.assertEqual(result["unique_graph_fact_count"], 1)
                self.assertEqual(result["anchors"][0]["name"], "PGI")
                self.assertIsNone(result["answer"])
                self.assertFalse(result["neo4j_and_milvus_modified"])
                self.assertTrue(outputs[0].with_suffix(".html").exists())
            finally:
                g.driver, g.vector_evidence, g.OUT = old_driver, old_vector, old_out

    def test_report_html_escape_and_labels(self):
        result = {
            "mode": "evidence", "question": "<test>", "anchors": [],
            "graph_facts": [], "milvus_evidence": {"text": [], "figure": []},
            "answer": None,
        }
        html = g.report_html(result)
        self.assertIn("&lt;test&gt;", html)
        self.assertIn("GraphRAG", html)


if __name__ == "__main__":
    unittest.main()
