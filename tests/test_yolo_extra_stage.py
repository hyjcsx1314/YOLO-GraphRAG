import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m
    spec.loader.exec_module(m)
    return m

AUTH=module('auth',HERE/'scripts/yolo_author_backfill.py')
AB=module('ab',HERE/'scripts/yolo_vision_ab_eval.py')

class TestExtraStage(unittest.TestCase):
    def test_authorship_counts(self):
        rows=AUTH.rows_for_import()
        self.assertEqual(len(rows),18)
        self.assertEqual(len({r['author_key'] for r in rows}),16)
        self.assertEqual(len({r['evidence_id'] for r in rows}),18)
        self.assertEqual(sum(r['author']=='Chien-Yao Wang' for r in rows),2)

    def test_fixed_questions(self):
        rows=AB.load_questions()
        self.assertEqual(len(rows),8)
        self.assertTrue(all('Figure' in r['question'] for r in rows))

    def test_parse_target_image(self):
        out='''TEXT EVIDENCE:\n[T1] YOLOv9 PDF p.2 E5=0.84\n x\nFIGURE EVIDENCE:\n[F1] YOLOv9_F3 YOLOv9 Figure 3 PDF p.5 RRF=0.08\nVISION TARGET PRIORITIZED: YOLOv9_F3\nVISION IMAGE SOURCE: YOLOv9_F3 -> data/visual_overrides/YOLOv9_F3.png\nCLOUD ANSWER (not local QLoRA):\n这是回答 [F1]\nResult JSON: path\nMULTIMODAL EVIDENCE RAG PASSED\n'''
        self.assertEqual(AB.figure_ids(out),['YOLOv9_F3'])
        self.assertEqual(AB.answer_from_stdout(out),'这是回答 [F1]')
        self.assertEqual(len(AB.evidence_signature(out)),64)

    def test_command_is_only_vision_toggle(self):
        q=AB.load_questions()[0]
        no=AB.command_for(q,'NoImage')
        yes=AB.command_for(q,'WithImage')
        self.assertEqual(no,yes[:-3])
        self.assertEqual(yes[-3:],['--vision','--vision-count','1'])

if __name__=='__main__':unittest.main()
