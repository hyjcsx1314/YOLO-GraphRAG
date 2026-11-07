"""Offline regression tests. No API, Milvus or Neo4j access."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE_FILE = Path(__file__).resolve().parents[1] / "scripts/yolo_rag_eval.py"
spec=importlib.util.spec_from_file_location("yolo_rag_eval",MODULE_FILE)
evalmod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(evalmod)


class EvaluationTests(unittest.TestCase):
    def test_frozen_manifest_and_question_set(self):
        rows, manifest=evalmod.load_items()
        self.assertEqual(len(rows),12)
        self.assertEqual(manifest["count"],12)
        self.assertEqual({x["paper"] for x in rows},{"YOLOX","YOLOv7","YOLOv9","YOLOv10"})
        self.assertEqual(len({x["id"] for x in rows}),12)

    def test_answer_extraction(self):
        vector=("hello\nCLOUD ANSWER (not local QLoRA):\n"
                "测试回答 [T2][F1]\nResult JSON: /tmp/abc.json\nMULTIMODAL EVIDENCE RAG PASSED\n")
        graph=("CLOUD ANSWER (not local QLoRA):\n第一段 [G1]\n第二段 [T2]\n"
               "Vision images sent: none\nResult JSON: /tmp/out.json\nYOLO GRAPHRAG CLOUD PASSED")
        self.assertEqual(evalmod.extract_answer(vector),"测试回答 [T2][F1]")
        self.assertEqual(evalmod.extract_answer(graph),"第一段 [G1]\n第二段 [T2]")
        self.assertEqual(evalmod.extract_answer("missing"),"")

    def test_scores_validate_bad_input(self):
        self.assertEqual(evalmod.number("2"),2)
        self.assertIsNone(evalmod.number(""))
        with self.assertRaises(ValueError):evalmod.number("3")
        with self.assertRaises(ValueError):evalmod.number("A")
        with self.assertRaises(ValueError):evalmod.number("-1")

    def test_run_offline_fake_cloud_and_score_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/"scripts").mkdir()
            (root/"evaluation/holdout_v1").mkdir(parents=True)
            source=MODULE_FILE.parents[1]/"evaluation/holdout_v1"
            for p in ("questions.csv","frozen_manifest.json"):
                shutil.copy(source/p,root/"evaluation/holdout_v1"/p)
            for method,filename in (("VectorRAG","yolo_multimodal_rag.py"),("GraphRAG","yolo_graph_rag.py")):
                marker="MULTIMODAL EVIDENCE RAG PASSED" if method=="VectorRAG" else "YOLO GRAPHRAG CLOUD PASSED"
                (root/"scripts"/filename).write_text(
                    "import sys\nprint('CLOUD ANSWER (not local QLoRA):')\n"
                    f"print('mock answer [{ 'T1' if method=='VectorRAG' else 'G1' }]')\n"
                    f"print('{marker}')\n",encoding="utf-8")
            with mock.patch.multiple(evalmod,ROOT=root,
                    QPATH=root/"evaluation/holdout_v1/questions.csv",
                    MANIFEST=root/"evaluation/holdout_v1/frozen_manifest.json",
                    SCRIPTS={m:root/"scripts"/name for m,name in (("VectorRAG","yolo_multimodal_rag.py"),("GraphRAG","yolo_graph_rag.py"))}):
                output=root/"results/evaluation/holdout_v1"
                args=type("Args",(),{"ids":"","limit":2,"methods":"both","out_dir":str(output),
                                     "dry_run":False,"timeout":20,"cooldown":0,"retry_failed":False})()
                env={"RAG_LLM_ENDPOINT":"https://example.invalid", "RAG_LLM_MODEL":"mock",
                     "RAG_LLM_KEY":"mock-key-not-a-secret", "NEO4J_URI":"neo4j+s://example.invalid",
                     "NEO4J_USERNAME":"mock", "NEO4J_PASSWORD":"mock-password"}
                with mock.patch.dict(os.environ,env):evalmod.run(args)
                first=evalmod.run_status(output,"X01","VectorRAG")
                self.assertEqual(first["status"],"ok")
                self.assertTrue((output/first["answer_file"]).read_text(encoding="utf-8").startswith("mock answer"))
                evalmod.prepare_review(type("Args",(),{"out_dir":str(output),"overwrite":False})())
                self.assertTrue((output/"score_sheet.xlsx").is_file())
                sheet=output/"score_sheet.csv"
                with sheet.open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
                for row in rows:
                    if row["run_status"]=="ok":
                        row.update({"factual_0_2":"2","citation_0_2":"2","completeness_0_2":"2",
                                    "graph_extra_0_2":"2" if row["method"]=="GraphRAG" else "",
                                    "hallucination_0_1":"0"})
                with sheet.open("w",encoding="utf-8-sig",newline="") as f:
                    writer=csv.DictWriter(f,fieldnames=evalmod.SCORE_FIELDS)
                    writer.writeheader();writer.writerows(rows)
                evalmod.summarize(type("Args",(),{"out_dir":str(output),"scores":str(sheet)})())
                result=json.loads((output/"manual_summary.json").read_text(encoding="utf-8"))
                self.assertEqual(result["paired_n"],2)
                self.assertEqual(result["mean_paired_delta_graph_minus_vector"],0)
                self.assertEqual(result["methods"]["VectorRAG"]["mean_common_0_6"],6)


if __name__=="__main__": unittest.main()
