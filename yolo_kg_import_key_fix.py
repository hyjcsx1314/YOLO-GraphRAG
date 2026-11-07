#!/usr/bin/env python3
"""Safe idempotent patch for YOLO KG v1 importer.
Run at ~/projects/yolo-graphrag project root before initial graph ingest.
Only affects graph entity keys (aliases and paper-scoped figures/tables).
Does not edit candidate CSV, SHA-based evidence IDs, Milvus or source datasets.
"""
from pathlib import Path
import ast

p = Path('scripts/yolo_kg_stage.py')
if not p.is_file():
    raise SystemExit('ERROR: run from ~/projects/yolo-graphrag; scripts/yolo_kg_stage.py missing')
s = p.read_text(encoding='utf-8')
if 'PAPER_SCOPED_FIGURE_KEYS_V2' in s:
    print('PATCH ALREADY INSTALLED; no changes made')
    raise SystemExit(0)

old = '''def key_for(kind: str, name: str) -> str:
    return kind.lower() + "::" + norm_text(name).casefold()
'''
new = '''def key_for(kind: str, name: str, paper_id: str | None = None) -> str:
    # PAPER_SCOPED_FIGURE_KEYS_V2: preserve provenance and normalize common aliases.
    key = norm_text(name).casefold()
    if kind == "Method":
        key = {
            "programmable gradient information (pgi)": "pgi",
            "consistent dual assignments strategy": "consistent dual assignments",
            "compound model scaling method": "compound scaling method",
            "rank-guided block design scheme": "rank-guided block design",
            "coarse for assistant and fine for lead label assignment strategy":
                "coarse-to-fine lead guided label assignment",
        }.get(key, key)
    elif kind == "Model":
        key = {
            "generalized efficient layer aggregation network (gelan)": "gelan",
            "yolov7e6e": "yolov7-e6e",
        }.get(key, key)
    if kind in ("Figure", "Table"):
        if not paper_id:
            raise ValueError("Figure/Table entity requires paper_id for safe key generation")
        if kind == "Figure":
            m = re.fullmatch(r"(?:figure|fig\\.?)\\s*(\\d+)\\s*\\.?\\s*(?:\\(\\s*([a-z])\\s*\\)|([a-z]))?", key)
            if m:
                key = "figure " + m.group(1) + (" (" + (m.group(2) or m.group(3)) + ")" if (m.group(2) or m.group(3)) else "")
        else:
            m = re.fullmatch(r"(?:table|tab\\.?)\\s*(\\d+)", key)
            if m:
                key = "table " + m.group(1)
        key = norm_text(paper_id).casefold() + "::" + key
    return kind.lower() + "::" + key
'''
old_head = '"head_key": key_for(x["head_type"], x["head_name"]),'
new_head = '"head_key": key_for(x["head_type"], x["head_name"], x["paper_id"]),'
old_tail = '"tail_key": key_for(x["tail_type"], x["tail_name"]),'
new_tail = '"tail_key": key_for(x["tail_type"], x["tail_name"], x["paper_id"]),'
if any(s.count(needle) != 1 for needle in (old, old_head, old_tail)):
    raise SystemExit('ERROR: script differs from v1 source; no changes made')
updated = s.replace(old, new, 1).replace(old_head, new_head, 1).replace(old_tail, new_tail, 1)
ast.parse(updated)
backup = p.with_name('yolo_kg_stage.py.bak.before_keys_v2')
if backup.exists():
    raise SystemExit('ERROR: backup already exists; no changes made')
backup.write_text(s, encoding='utf-8')
p.write_text(updated, encoding='utf-8')
print('NEO4J ENTITY KEY PATCH PASSED')
print('Backup:', backup)
print('Figure/Table now paper-scoped; common PGI/GELAN aliases unified')
