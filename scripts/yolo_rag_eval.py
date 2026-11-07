#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen 12-question human-scored VectorRAG/GraphRAG evaluation.

- Read-only with respect to Milvus and Neo4j.
- Never sends answer key/reference_key.json to an LLM.
- Uses installed, unchanged scripts/yolo_multimodal_rag.py and scripts/yolo_graph_rag.py.
- Cloud calls happen ONLY with the explicit `run` subcommand (not dry-run).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "evaluation/holdout_v1"
QPATH = TEST_DIR / "questions.csv"
MANIFEST = TEST_DIR / "frozen_manifest.json"
DEFAULT_OUT = ROOT / "results/evaluation/holdout_v1"
METHODS = ("VectorRAG", "GraphRAG")
SCRIPTS = {
    "VectorRAG": ROOT / "scripts/yolo_multimodal_rag.py",
    "GraphRAG": ROOT / "scripts/yolo_graph_rag.py",
}
SCORES = ("factual_0_2", "citation_0_2", "completeness_0_2")
SCORE_FIELDS = ["test_id", "paper", "method", "question", "run_status", "answer_file",
                *SCORES, "graph_extra_0_2", "hallucination_0_1", "reviewer_notes"]


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_items() -> tuple[list[dict], dict]:
    if not QPATH.is_file() or not MANIFEST.is_file():
        raise RuntimeError("缺少 evaluation/holdout_v1/questions.csv 或 frozen_manifest.json")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual = sha256(QPATH)
    if actual != manifest["questions_sha256"]:
        raise RuntimeError("固定题集 SHA256 不匹配。不要编辑问题；另建版本评测。")
    with QPATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or len(rows) != manifest["count"]:
        raise RuntimeError("题集数量异常")
    ids = [x["id"] for x in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("重复测试题 ID")
    return rows, manifest


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def safe_text(s: str) -> str:
    for secret_key in ("RAG_LLM_KEY", "NEO4J_PASSWORD", "HF_TOKEN"):
        token = os.getenv(secret_key, "")
        if token and len(token) >= 8:
            s = s.replace(token, "[REDACTED]")
    return s


def command_for(method: str, question: str) -> list[str]:
    if method == "VectorRAG":
        return [sys.executable, str(SCRIPTS[method]), "ask", "--question", question,
                "--mode", "cloud"]
    return [sys.executable, str(SCRIPTS[method]), "ask", "--question", question,
            "--mode", "cloud", "--hops", "2"]


def extract_answer(stdout: str) -> str:
    marker = "CLOUD ANSWER (not local QLoRA):"
    if marker not in stdout:
        return ""
    part = stdout.split(marker, 1)[1]
    # Result artifact paths are emitted by both versions after the answer.
    part = re.split(r"(?m)^\s*(?:Vision images sent:|Result JSON:|Image gallery:|Evidence HTML:|"
                    r"YOLO GRAPHRAG CLOUD PASSED|MULTIMODAL EVIDENCE RAG PASSED)",
                    part, maxsplit=1)[0]
    return part.strip()


def record_file(out: Path, qid: str, method: str) -> Path:
    return out / "records" / f"{qid}_{method}.json"


def run_status(out: Path, qid: str, method: str) -> dict:
    r = read_json(record_file(out, qid, method))
    return r if isinstance(r, dict) else {}


def config_for(manifest: dict) -> dict:
    return {
        "version": manifest["version"], "questions_sha256": manifest["questions_sha256"],
        "script_sha256": {method: sha256(path) for method, path in SCRIPTS.items()},
        "cloud_model": os.getenv("RAG_LLM_MODEL", ""),
        "generation_modes": {"VectorRAG": "cloud", "GraphRAG": "cloud"},
        "graph_hops": 2, "graph_anchor_mode": "auto (LLM selection)", "vision": False,
        "note": "These are two independently implemented prompts; do not interpret delta as graph-only causal effect.",
    }


def check_config(out: Path, manifest: dict, *, dry_run: bool):
    for name, script in SCRIPTS.items():
        if not script.is_file():
            raise RuntimeError(f"找不到现有 {name} 程序：{script}")
    cfg = config_for(manifest)
    old = read_json(out / "run_config.json")
    if old and isinstance(old, dict):
        old_static = {k: old[k] for k in cfg if k in old}
        if old_static != cfg:
            raise RuntimeError("该输出目录已有不同模型/脚本版本的评测，请用 --out-dir 新建独立目录，不能混合计分")
    elif not dry_run:
        if any(out.glob("records/*.json")):
            raise RuntimeError("存在旧记录但没有 run_config.json，请另建 --out-dir")
        write_json(out / "run_config.json", {**cfg, "created_at_utc": stamp()})
    return cfg


def selected_items(items: list[dict], args):
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        unknown = wanted - {x["id"] for x in items}
        if unknown:
            raise RuntimeError("题集没有这些 ID：" + ", ".join(sorted(unknown)))
        items = [x for x in items if x["id"] in wanted]
    if args.limit:
        items = items[:args.limit]
    return items


def run(args):
    items, manifest = load_items()
    items = selected_items(items, args)
    out = Path(args.out_dir).resolve()
    cfg = check_config(out, manifest, dry_run=args.dry_run)
    methods = list(METHODS if args.methods == "both" else (args.methods,))
    plan = []
    # Alternate method order by question to reduce a systematic order effect.
    for idx, item in enumerate(items):
        ordering = methods[::-1] if idx % 2 else methods
        for method in ordering:
            existing = run_status(out, item["id"], method)
            plan.append((item, method, existing))
    if args.dry_run:
        print("DRY RUN: no cloud requests; questions SHA256:", manifest["questions_sha256"])
        print("LLM model:", cfg["cloud_model"] or "(not set)")
        for item, method, existing in plan:
            print(item["id"], method, "SKIP OK" if existing.get("status") == "ok" else "WOULD RUN")
        print("PLANNED:", len(plan), "calls (GraphRAG may issue an additional LLM anchor-selection request each question)")
        return
    if not (os.getenv("RAG_LLM_ENDPOINT") and os.getenv("RAG_LLM_MODEL") and os.getenv("RAG_LLM_KEY")):
        raise RuntimeError("缺少 RAG_LLM_ENDPOINT / RAG_LLM_MODEL / RAG_LLM_KEY")
    if "GraphRAG" in methods and not all(os.getenv(k) for k in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")):
        raise RuntimeError("GraphRAG 需要 NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD")
    total = 0
    for item, method, existing in plan:
        if existing.get("status") == "ok":
            print("SKIP PASSED:", item["id"], method)
            continue
        if existing and not args.retry_failed:
            print("STOP: 已有失败记录", item["id"], method, "请修复后使用 --retry-failed")
            break
        qid = item["id"]
        command = command_for(method, item["question"])
        print("RUN:", qid, method, "UTC:", stamp(), flush=True)
        start = time.monotonic()
        exception = ""
        try:
            completed = subprocess.run(command, cwd=ROOT, capture_output=True,
                                       text=True, encoding="utf-8", errors="replace", timeout=args.timeout)
            stdout, stderr, code = completed.stdout, completed.stderr, completed.returncode
        except subprocess.TimeoutExpired as ex:
            stdout = (ex.stdout or b"").decode("utf-8", "replace") if isinstance(ex.stdout, bytes) else (ex.stdout or "")
            stderr = (ex.stderr or b"").decode("utf-8", "replace") if isinstance(ex.stderr, bytes) else (ex.stderr or "")
            code = 124
            exception = "timeout"
        except OSError as ex:
            stdout, stderr, code, exception = "", type(ex).__name__, 125, type(ex).__name__
        elapsed = round(time.monotonic() - start, 2)
        answer = extract_answer(stdout)
        status = "ok" if code == 0 and len(answer) > 0 and (
            ("YOLO GRAPHRAG CLOUD PASSED" in stdout) if method == "GraphRAG" else
            ("MULTIMODAL EVIDENCE RAG PASSED" in stdout)) else "error"
        for folder in ("logs", "answers", "records"):
            (out / folder).mkdir(parents=True, exist_ok=True)
        prefix = qid + "_" + method
        (out / "logs" / (prefix + ".stdout.txt")).write_text(safe_text(stdout), encoding="utf-8")
        (out / "logs" / (prefix + ".stderr.txt")).write_text(safe_text(stderr), encoding="utf-8")
        answer_path = out / "answers" / (prefix + ".txt")
        answer_path.write_text(safe_text(answer), encoding="utf-8")
        record = {
            "question_id": qid, "paper": item["paper"], "method": method,
            "status": status, "exit_code": code, "exception": exception,
            "elapsed_seconds": elapsed, "finished_at_utc": stamp(), "question_sha256": hashlib.sha256(item["question"].encode("utf-8")).hexdigest(),
            "answer_file": str(answer_path.relative_to(out)),
            "stdout_file": str(Path("logs") / (prefix + ".stdout.txt")),
            "stderr_file": str(Path("logs") / (prefix + ".stderr.txt")),
            "citation_tokens": len(re.findall(r"\[(?:G|T|F)\d+\]", answer)),
            "citation_types": {letter: len(set(re.findall(r"\[" + letter + r"\d+\]", answer))) for letter in "GTF"},
        }
        write_json(record_file(out, qid, method), record)
        print("RESULT:", qid, method, status, "elapsed:", elapsed, "s", "answer_chars:", len(answer), flush=True)
        total += 1
        if status != "ok":
            print("STOP ON ERROR: 查看", out / "logs" / (prefix + ".stderr.txt"), flush=True)
            break
        if args.cooldown > 0:
            time.sleep(args.cooldown)
    print("THIS RUN COMPLETED REQUESTS:", total)
    print("NEXT: python scripts/yolo_rag_eval.py prepare-review")


def score_rows(items: list[dict], out: Path) -> list[dict]:
    rows = []
    for x in items:
        for method in METHODS:
            record = run_status(out, x["id"], method)
            row = {k: "" for k in SCORE_FIELDS}
            row.update({"test_id": x["id"], "paper": x["paper"], "method": method,
                        "question": x["question"], "run_status": record.get("status", "pending"),
                        "answer_file": record.get("answer_file", "")})
            rows.append(row)
    return rows


def create_review_xlsx(rows: list[dict], file: Path):
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
        from openpyxl.worksheet.datavalidation import DataValidation
        from openpyxl.formatting.rule import CellIsRule
    except ImportError:
        raise RuntimeError("请先安装 openpyxl: python -m pip install openpyxl") from None
    wb=Workbook()
    overview=wb.active;overview.title="Overview"
    overview.sheet_view.showGridLines=False
    overview.merge_cells("A1:F1");overview["A1"]="YOLO RAG HOLDOUT · Manual Evaluation"
    overview["A1"].font=Font(name="Aptos Display",size=20,bold=True,color="173F54")
    overview.row_dimensions[1].height=42
    overview["A3"]="方法";overview["B3"]="完成评分 (n)";overview["C3"]="平均共同指标 (0–6)";overview["D3"]="平均准确性";overview["E3"]="平均引用";overview["F3"]="平均完整性"
    nrow=4+len(rows)
    for idx, method in enumerate(METHODS,4):
        overview.cell(idx,1,method)
        overview.cell(idx,2,f'=COUNTIFS(Scoring!$C$5:$C${nrow},A{idx},Scoring!$M$5:$M${nrow},">=0")')
        overview.cell(idx,3,f'=IFERROR(AVERAGEIFS(Scoring!$M$5:$M${nrow},Scoring!$C$5:$C${nrow},A{idx}),"")')
        for col,src in [(4,"G"),(5,"H"),(6,"I")]:
            overview.cell(idx,col,f'=IFERROR(AVERAGEIFS(Scoring!${src}$5:${src}${nrow},Scoring!$C$5:$C${nrow},A{idx}),"")')
    overview.merge_cells("A8:F8");overview["A8"]="先对照原论文为每个回答填写 G/H/I 三项 (0–2)，GraphRAG 额外关系价值填 J，事实幻觉填 K。"
    overview.merge_cells("A9:F9");overview["A9"]="总分仅统计 G/H/I 三项 (0–6)；图谱增量价值不用于跨方法总分，以免偏袒 GraphRAG。"
    overview.merge_cells("A10:F10");overview["A10"]="评分后运行 summarize --scores results/evaluation/holdout_v1/score_sheet.xlsx 输出配对汇总。"
    for col in "ABCDEF":overview.column_dimensions[col].width=25 if col!="A" else 30
    for r in (8,9,10):overview.cell(r,1).alignment=Alignment(wrap_text=True,vertical="center");overview.row_dimensions[r].height=34

    ws=wb.create_sheet("Scoring")
    ws.sheet_view.showGridLines=False
    ws.merge_cells("A1:M1");ws["A1"]="Independent QA Evaluation · 12 Questions × 2 Methods"
    ws["A1"].font=Font(size=17,bold=True,color="173F54")
    ws.merge_cells("A2:M2");ws["A2"]="只对原文核对后的答案评分；不要以答案长短或引用数量代替准确性。"
    ws["A2"].font=Font(italic=True,color="657782")
    headings=["Test ID","Paper","Method","Question","Run status","Answer file",
              "Factual 0–2","Citation 0–2","Complete 0–2","Graph extra 0–2",
              "Hallucination 0/1","Review notes","Common total 0–6"]
    for c,title in enumerate(headings,1):ws.cell(4,c,title)
    for index, entry in enumerate(rows,5):
        vals=[entry["test_id"],entry["paper"],entry["method"],entry["question"],
              entry["run_status"],entry["answer_file"],None,None,None,None,None,None,
              f'=IF(COUNT(G{index}:I{index})=3,SUM(G{index}:I{index}),"")']
        for col,value in enumerate(vals,1):ws.cell(index,col,value)
        ws.row_dimensions[index].height=39
        for col in range(1,14):ws.cell(index,col).alignment=Alignment(wrap_text=True,vertical="center")
        for col in (7,8,9,10,11):ws.cell(index,col).font=Font(color="1768B2",bold=True)
        if entry["method"]=="VectorRAG": ws.cell(index,10).fill=PatternFill("solid",fgColor="EEF0F3")
    ws.freeze_panes="G5"
    ws.auto_filter.ref=f"A4:M{nrow}"
    widths={"A":11,"B":13,"C":16,"D":68,"E":13,"F":44,"G":14,"H":14,"I":14,"J":18,"K":18,"L":56,"M":18}
    for k,v in widths.items():ws.column_dimensions[k].width=v
    validation=DataValidation(type="whole",operator="between",formula1=0,formula2=2,allow_blank=True)
    validation.error="必须填写0、1或2";validation.showErrorMessage=True
    ws.add_data_validation(validation);validation.add(f"G5:J{nrow}")
    v1=DataValidation(type="whole",operator="between",formula1=0,formula2=1,allow_blank=True)
    v1.showErrorMessage=True;v1.error="请填0或1";ws.add_data_validation(v1);v1.add(f"K5:K{nrow}")
    ws.conditional_formatting.add(f"G5:I{nrow}",CellIsRule(operator="equal",formula=["0"],fill=PatternFill("solid",fgColor="FBE7E5")))
    ws.conditional_formatting.add(f"G5:I{nrow}",CellIsRule(operator="equal",formula=["2"],fill=PatternFill("solid",fgColor="DCF2E5")))

    ref=wb.create_sheet("Rubric")
    ref.sheet_view.showGridLines=False
    criteria=[
        ("Factual (事实正确)","0: 关键事实错误/明显幻觉；1: 核心正确但有遗漏或较小错误；2: 与原文关键要点一致"),
        ("Citation (引用准确)","0: 缺失/错引关键事实；1: 有引用但覆盖不完整；2: 主要事实的 [T][F][G] 引用与对应来源相符"),
        ("Complete (回答完整)","0: 基本未回答；1: 满足部分 gold 要点；2: 满足全部关键要点、不混淆指标"),
        ("Graph extra (图谱增量)","仅GraphRAG填写：0无用/错误关系；1有相关关系但增量有限；2可验证的实体关联明显帮助理解本题"),
        ("Hallucination","1: 存在明显无依据/与原文相反的断言；0: 未发现明显幻觉（不表示模型永远无误）"),
        ("Reference","对照 evaluation/holdout_v1/reference_key.json 中的原文页码与要点评分。该文件绝不能作为问答模型的输入。"),
        ("Comparison","共同指标只比较 factual + citation + complete (0–6)；Graph extra 仅单独报告。")]
    ref.append(["Dimension","Scoring guidance"])
    for i,(a,b) in enumerate(criteria,2):ref.cell(i,1,a);ref.cell(i,2,b);ref.row_dimensions[i].height=45
    ref.column_dimensions["A"].width=31;ref.column_dimensions["B"].width=105
    for row in ref:
        for cell in row: cell.alignment=Alignment(vertical="center",wrap_text=True)
    # Simple consistent headers and subtle alternating rows.
    for sheet, header in [(overview,3),(ws,4),(ref,1)]:
        for cell in sheet[header]:
            cell.fill=PatternFill("solid",fgColor="1E5268")
            cell.font=Font(color="FFFFFF",bold=True)
            cell.alignment=Alignment(vertical="center",wrap_text=True)
        sheet.row_dimensions[header].height=31
    for idx in range(5,nrow+1):
        if idx%2==0:
            for col in (1,2,3,4,5,6,12,13):ws.cell(idx,col).fill=PatternFill("solid",fgColor="F5F8FA")
    file.parent.mkdir(parents=True, exist_ok=True)
    wb.save(file)


def prepare_review(args):
    items,_=load_items();out=Path(args.out_dir).resolve()
    rows=score_rows(items,out)
    csvpath=out/"score_sheet.csv";xlsx=out/"score_sheet.xlsx"
    if (csvpath.exists() or xlsx.exists()) and not args.overwrite:
        print("评分文件已存在，禁止覆盖人工分数：",csvpath,xlsx)
        print("如确实需要重新生成，请在备份文件后使用 --overwrite")
        return
    out.mkdir(parents=True,exist_ok=True)
    with csvpath.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=SCORE_FIELDS);writer.writeheader();writer.writerows(rows)
    create_review_xlsx(rows,xlsx)
    print("REVIEW PREPARED:",len(rows),"rows")
    print("Score CSV:",csvpath)
    print("Score Excel:",xlsx)
    print("Use reference_key.json as human answer key; do not send it to either RAG script.")


def number(value, *, maximum=2):
    if value is None or str(value).strip()=="":return None
    if isinstance(value,bool):raise ValueError("布尔值不是分数")
    v=str(value).strip()
    if v.endswith(".0"):v=v[:-2]
    if not re.fullmatch(r"[0-9]+",v):raise ValueError("分数不是整数: "+v)
    n=int(v)
    if n<0 or n>maximum:raise ValueError("分数超出范围: "+v)
    return n


def load_scores(path: Path):
    if not path.is_file():raise RuntimeError("找不到评分文件: "+str(path))
    if path.suffix.lower()==".xlsx":
        try:from openpyxl import load_workbook
        except ImportError:raise RuntimeError("请安装 openpyxl") from None
        ws=load_workbook(path,read_only=True,data_only=False)["Scoring"]
        rows=[]
        for cells in ws.iter_rows(min_row=5,values_only=True):
            if not cells or not cells[0]:continue
            rows.append(dict(zip(SCORE_FIELDS,[cells[i] if i<len(cells) else None for i in range(12)])))
        return rows
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def summarize(args):
    items,_=load_items();out=Path(args.out_dir).resolve()
    default=out/"score_sheet.xlsx"
    path=Path(args.scores).resolve() if args.scores else (default if default.is_file() else out/"score_sheet.csv")
    scores=load_scores(path)
    allowed={(it["id"],method) for it in items for method in METHODS}
    if len(scores)!=len(allowed): raise RuntimeError(f"评分行数应是{len(allowed)}，实际{len(scores)}")
    seen=set();valid={};incomplete=[]
    for row in scores:
        key=(str(row.get("test_id")),str(row.get("method")))
        if key not in allowed or key in seen:raise RuntimeError("评分表ID/方法无效或重复: "+repr(key))
        seen.add(key)
        values=[number(row.get(col)) for col in SCORES]
        actual_status = run_status(out, key[0], key[1]).get("status")
        if actual_status != "ok":
            if any(x is not None for x in values):
                raise RuntimeError(f"{key} 未成功生成答案却已有评分，请勿将模拟/未运行结果计入正式报告")
            incomplete.append(key)
            continue
        if any(x is None for x in values):
            incomplete.append(key);continue
        val={"factual":values[0],"citation":values[1],"completeness":values[2],"total":sum(values),
             "graph_extra":number(row.get("graph_extra_0_2")),
             "hallucination":number(row.get("hallucination_0_1"),maximum=1),
             "notes":str(row.get("reviewer_notes") or "")}
        valid[key]=val
    pair=[]
    for it in items:
        a=valid.get((it["id"],"VectorRAG"));b=valid.get((it["id"],"GraphRAG"))
        if a and b:
            pair.append({"id":it["id"],"paper":it["paper"],"vector":a["total"],"graph":b["total"],
                         "delta_graph_minus_vector":b["total"]-a["total"],"graph_extra":b["graph_extra"]})
    method_summary={}
    for method in METHODS:
        vals=[x for (qid,m),x in valid.items() if m==method]
        method_summary[method]={"scored_n":len(vals),"mean_common_0_6":round(mean(x["total"] for x in vals),3) if vals else None,
            "mean_factual_0_2":round(mean(x["factual"] for x in vals),3) if vals else None,
            "mean_citation_0_2":round(mean(x["citation"] for x in vals),3) if vals else None,
            "mean_completeness_0_2":round(mean(x["completeness"] for x in vals),3) if vals else None,
            "hallucinations_flagged":sum(x["hallucination"]==1 for x in vals),
            "hallucination_rows_marked":sum(x["hallucination"] is not None for x in vals)}
    graph_extra=[v["graph_extra"] for (q,m),v in valid.items() if m=="GraphRAG" and v["graph_extra"] is not None]
    summary={"generated_at_utc":stamp(),"scoring_file":str(path),"questions":len(items),"scored_rows":len(valid),
             "unscored_rows":len(incomplete),"paired_n":len(pair),"methods":method_summary,
             "mean_paired_delta_graph_minus_vector":round(mean(x["delta_graph_minus_vector"] for x in pair),3) if pair else None,
             "mean_graph_extra_0_2":round(mean(graph_extra),3) if graph_extra else None,
             "paired_comparisons":pair,
             "limitation":"Small domain-specific holdout, not evidence of statistical significance or isolated graph causality; differing prompts/token limits and GraphRAG anchor LLM calls are confounds."}
    write_json(out/"manual_summary.json",summary)
    with (out/"paired_scores.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["id","paper","vector","graph","delta_graph_minus_vector","graph_extra"]);writer.writeheader();writer.writerows(pair)
    md=["# Vector RAG / GraphRAG 评分汇总", "",f"- 完成评分：{len(valid)}/{len(allowed)} 条回答",f"- 可配对问题：{len(pair)}/{len(items)} 道", "", "| 方法 | 已评分 | 平均共同得分（0–6） | 事实（0–2） | 引用（0–2） | 完整性（0–2） |", "|---|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        v=method_summary[m]
        fmt=lambda n:'待评分' if n is None else f'{n:.2f}'
        md.append(f"| {m} | {v['scored_n']} | {fmt(v['mean_common_0_6'])} | {fmt(v['mean_factual_0_2'])} | {fmt(v['mean_citation_0_2'])} | {fmt(v['mean_completeness_0_2'])} |")
    md.extend(["",f"同题配对平均差值 GraphRAG−VectorRAG：{summary['mean_paired_delta_graph_minus_vector'] if pair else '待评分'}（0–6分）",
               f"GraphRAG 图谱额外关系平均评分（仅单独报告）：{summary['mean_graph_extra_0_2'] if graph_extra else '待评分'}（0–2分）",
               "", "## 配对结果", "", "| 题号 | 论文 | Vector | Graph | 差值 |", "|---|---|---:|---:|---:|"])
    for v in pair:md.append(f"| {v['id']} | {v['paper']} | {v['vector']} | {v['graph']} | {v['delta_graph_minus_vector']:+d} |")
    md.extend(["", "## 实验局限", "", "这是开发后新固定的问题集，规模有限，不能据此称统计显著或未见领域泛化。两份原有脚本的系统提示词和输出 token 预算并不完全相同；GraphRAG 还额外调用 LLM 做锚点定位，因此分数差异不能被单独归因为图谱的因果作用。实际得分由人工对照原 PDF 给出。"])
    (out/"manual_summary.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print("SUMMARY WRITTEN:",out/"manual_summary.md")
    print("SCORED:",len(valid),"/",len(allowed),"PAIRS:",len(pair),"/",len(items))
    print("No scores were invented. Incomplete answers excluded from means.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest="task",required=True)
    p=subs.add_parser("run",help="执行固定问题的两个系统的云端问答；要花API费用")
    p.add_argument("--limit",type=int,default=0,help="只测试前N道，用于小规模烟测")
    p.add_argument("--ids",default="",help="逗号分隔的题目ID，如X01,V701")
    p.add_argument("--methods",choices=("both",)+METHODS,default="both")
    p.add_argument("--out-dir",default=str(DEFAULT_OUT))
    p.add_argument("--timeout",type=int,default=900)
    p.add_argument("--cooldown",type=float,default=2.0)
    p.add_argument("--retry-failed",action="store_true",help="明确允许对失败请求再次收费重试")
    p.add_argument("--dry-run",action="store_true",help="打印将执行的命令但不做任何云端调用")
    p=subs.add_parser("prepare-review",help="准备评分表（CSV+Excel），默认不覆盖旧分数")
    p.add_argument("--out-dir",default=str(DEFAULT_OUT))
    p.add_argument("--overwrite",action="store_true")
    p=subs.add_parser("summarize",help="按人工填写的分数汇总，绝不自动打分")
    p.add_argument("--out-dir",default=str(DEFAULT_OUT))
    p.add_argument("--scores",default="",help="CSV或XLSX的人工评分路径")
    args=parser.parse_args()
    try:
        if args.task=="run":
            if args.limit<0 or args.timeout<10 or args.cooldown<0:raise RuntimeError("运行参数非法")
            run(args)
        elif args.task=="prepare-review":prepare_review(args)
        else:summarize(args)
    except (RuntimeError,ValueError,OSError) as exc:
        print("EVALUATION ERROR:",exc,file=sys.stderr);sys.exit(1)


if __name__=="__main__":main()
