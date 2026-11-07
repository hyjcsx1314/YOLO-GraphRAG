import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

APP_DIR = Path(__file__).resolve().parents[1]

def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

bridge = load("bridge_local_test", APP_DIR / "bridge_local.py")
adapter = load("backend_adapter_test", APP_DIR / "backend_adapter.py")

STUB = r'''
import json, sys
from pathlib import Path

def arg(name):
    return sys.argv[sys.argv.index(name)+1]

root = Path(__file__).resolve().parents[1]
is_graph = Path(__file__).name == "yolo_graph_rag.py"
question = arg("--question")
vision = "--vision" in sys.argv
assert arg("--mode") == "cloud"
if is_graph:
    assert arg("--hops") == "2"
    output = root / "results/neo4j/graphrag_v1/test_graph.json"
    record = {
      "question": question, "mode": "cloud", "answer": "PGI 解释 [G1][T1][F1]",
      "cloud_model": "offline-test-llm",
      "graph_facts": [{"citation_id": "G1", "head": "YOLOv9",
          "relation": "PROPOSES", "tail": "PGI",
          "evidence": [{"paper": "YOLOv9", "page": 4, "quote": "original evidence"}]}],
      "milvus_evidence": {
          "text": [{"id": "T1", "header": "YOLOv9 p.4", "snippet": "论文证据"}],
          "figure": [{"id": "F1", "header": "YOLOv9_F3 Figure 3"}]},
      "vision_image_ids": ["YOLOv9_F3"] if vision else [],
      "vision_images": [{
          "citation_id": "F1", "asset_id": "YOLOv9_F3",
          "path": str(root / "data/visual_overrides/YOLOv9_F3.png")
      }] if vision else [],
    }
else:
    assert arg("--text-k") == "5"
    assert arg("--figure-k") == "5"
    output = root / "results/milvus/multimodal_rag/test_vector.json"
    record = {
      "question": question, "mode": "cloud",
      "generation": {"answer": "证据回答 [T1][F1]", "model": "offline-test-llm"},
      "text_evidence": [{
          "citation_id": "T1", "paper_id": "YOLOv9", "pdf_page": 4,
          "source": "full_pdf_page_index", "score": 0.7, "text": "paper text",
      }],
      "figure_evidence": [{
          "citation_id": "F1", "paper_id": "YOLOv9", "pdf_page": 5,
          "asset_id": "YOLOv9_F3", "image_path": "assets/YOLOv9_F3.png",
          "label": "Figure 3", "caption_pdf": "caption", "description_zh": "说明",
          "rrf_score": 0.024
      }],
    }
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
if not is_graph and vision:
    print("VISION IMAGE SOURCE: YOLOv9_F3 -> data/visual_overrides/YOLOv9_F3.png")
print("Result JSON:", output)
'''

class BridgeIntegrationOffline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="yolo_ui_offline_")
        self.root = Path(self.temp.name)
        (self.root / "frontend").mkdir(parents=True)
        (self.root / "scripts").mkdir(parents=True)
        for file in ("yolo_graph_rag.py", "yolo_multimodal_rag.py"):
            (self.root / "scripts" / file).write_text(STUB, encoding="utf-8")
        shutil.copyfile(APP_DIR / "bridge_local.py", self.root / "frontend/bridge_local.py")
        for folder in ("data/visual_overrides", "data/yolo_visual_final_v2/assets"):
            (self.root / folder).mkdir(parents=True)
        for path in (
            "data/visual_overrides/YOLOv9_F3.png",
            "data/yolo_visual_final_v2/assets/YOLOv9_F3.png",
        ):
            (self.root / path).write_bytes(b"offline fixture")

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def req(mode, vision):
        return {"question": "YOLOv9 的 PGI 是什么？", "mode": mode,
                "with_images": vision, "top_k": 5}

    def test_graph_vision(self):
        res = bridge.query_existing_project(self.req("GraphRAG", True), self.root)
        self.assertIn("[G1]", res["answer"])
        self.assertEqual(res["meta"]["vision_images_sent"], 1)
        self.assertEqual({x["id"] for x in res["evidence"]}, {"G1", "T1", "F1"})
        img = next(x for x in res["evidence"] if x["kind"] == "F")
        self.assertEqual(img["image_path"], "data/visual_overrides/YOLOv9_F3.png")
        self.assertTrue(img["sent_to_model"])

    def test_graph_no_vision(self):
        res = bridge.query_existing_project(self.req("GraphRAG", False), self.root)
        img = next(x for x in res["evidence"] if x["kind"] == "F")
        self.assertEqual(img["image_path"], "data/yolo_visual_final_v2/assets/YOLOv9_F3.png")
        self.assertFalse(img["sent_to_model"])

    def test_vector_vision(self):
        res = bridge.query_existing_project(self.req("Vector RAG", True), self.root)
        self.assertIn("[T1]", res["answer"])
        self.assertEqual(res["meta"]["vision_images_sent"], 1)
        img = next(x for x in res["evidence"] if x["kind"] == "F")
        self.assertTrue(img["sent_to_model"])
        self.assertEqual(img["image_path"], "data/visual_overrides/YOLOv9_F3.png")

    def test_vector_no_vision(self):
        res = bridge.query_existing_project(self.req("Vector RAG", False), self.root)
        self.assertEqual(res["meta"]["vision_images_sent"], 0)
        img = next(x for x in res["evidence"] if x["kind"] == "F")
        self.assertFalse(img["sent_to_model"])
        self.assertEqual(img["image_path"], "data/yolo_visual_final_v2/assets/YOLOv9_F3.png")

    def test_bad_input_and_escape_paths(self):
        for req in (
            self.req("Other", False),
            {**self.req("GraphRAG", False), "top_k": 100},
            {**self.req("GraphRAG", False), "question": ""},
        ):
            with self.assertRaises(bridge.IntegrationError):
                bridge.query_existing_project(req, self.root)
        with self.assertRaises(bridge.IntegrationError):
            bridge._result_file("Result JSON: /tmp/outside.json", self.root, "GraphRAG")
        self.assertIsNone(adapter.safe_image_path(self.root, "/etc/passwd"))
        self.assertIsNone(adapter.safe_image_path(self.root, "../secrets.png"))

    def test_full_adapter_subprocess(self):
        env = {
            "RAG_LLM_ENDPOINT": "https://example.test/v1/chat/completions",
            "RAG_LLM_MODEL": "offline-test-llm", "RAG_LLM_KEY": "fake-key",
            "NEO4J_URI": "neo4j+s://example.test",
            "NEO4J_USERNAME": "test", "NEO4J_PASSWORD": "fake-password",
            "YOLO_UI_BRIDGE_COMMAND": f"{sys.executable} {self.root}/frontend/bridge_local.py",
        }
        with patch.dict(os.environ, env, clear=False):
            res = adapter.query_backend(self.root, "YOLOv9 的 PGI 是什么？", "GraphRAG", True, 5)
        self.assertIn("[G1]", res["answer"])
        self.assertEqual(res["meta"]["vision_images_sent"], 1)

    def test_missing_cloud_vars(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ["YOLO_UI_BRIDGE_COMMAND"] = (
                f"{sys.executable} {self.root}/frontend/bridge_local.py"
            )
            with self.assertRaises(adapter.BridgeNotConfigured):
                adapter.query_backend(self.root, "PGI?", "GraphRAG", False, 5)

if __name__ == "__main__":
    unittest.main()
