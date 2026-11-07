#!/usr/bin/env python3
# 依据本地页面分组清单，把种子样本放入训练/验证/测试池。
# 仅对照英文摘录和本地已清洗文本做机械验证，不能代替 PDF 原页核对。
import argparse
import json
import re
from pathlib import Path
from collections import Counter

def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]

def norm(text):
    return re.sub(r"\s+", " ", text).strip().casefold()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--project", default=".")
    args=parser.parse_args()
    project=Path(args.project).expanduser().resolve()
    data=project/"data"/"datasets"
    manifest_path=data/"split_manifest_v3.json"
    chunks_path=data/"all_chunks_v3.jsonl"
    seed_path=Path(__file__).resolve().parent/"yolo_seed_37_with_sources.jsonl"

    for item in (manifest_path,chunks_path,seed_path):
        if not item.exists():
            raise SystemExit(f"缺少文件: {item}\n请先运行 split_chunks_by_page_safe.py 并解压完整种子数据包")

    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    source_rows={r["chunk_id"]:r for r in read_jsonl(chunks_path)}
    seeds=read_jsonl(seed_path)
    result={"train":[],"val":[],"test":[]}
    skipped=[]

    for seed in seeds:
        ident=seed["chunk_id"]
        paper=seed["paper_id"]
        page=seed["page"]
        src=source_rows.get(ident)
        if src is None:
            skipped.append((ident,"V3 原文块不存在"))
            continue
        missing=[e for e in seed["evidence_spans"] if norm(e) not in norm(src["text"])]
        if missing:
            skipped.append((ident,"本地原文中未逐字找到部分证据"))
            continue
        matches=[name for name in ("train","val","test") if page in manifest.get(paper,{}).get(name,[])]
        if len(matches)!=1:
            skipped.append((ident,"页面未唯一归属数据划分清单（可能是已排除的低价值片段）"))
            continue
        split=matches[0]
        local=seed.copy()
        # 本地全文只用于证据核验；训练输入保留精选短摘录，减少致谢和冗余
        local["input"]=seed["input"]
        local["split"]=split
        result[split].append(local)

    for split,items in result.items():
        with_meta=data/f"seed_{split}_with_sources.jsonl"
        bare=data/f"seed_{split}_alpaca.jsonl"
        with_meta.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in items),encoding="utf-8")
        bare.write_text("".join(json.dumps({k:r[k] for k in ("instruction","input","output")},ensure_ascii=False)+"\n" for r in items),encoding="utf-8")
        print(f"{split:5s}  {len(items):2d} 条；来源页面 {len({(r['paper_id'],r['page']) for r in items})} 个")

    report=data/"seed_alignment_report.txt"
    report.write_text(
        f"源种子数量：{len(seeds)}\n已匹配：{sum(map(len,result.values()))}\n未匹配：{len(skipped)}\n"
        +"\n".join(f"{k} | {why}" for k,why in skipped),
        encoding="utf-8")
    print("未匹配样本:",len(skipped),"(可能多条属于同一个来源片段)")
    print("对齐报告:",report)
    print("注意：数据均未进行原 PDF 原页最终人工复核。")
    if not result["train"]:
        raise SystemExit("训练组为空，请检查 split_manifest_v3.json 和对齐报告")

if __name__=="__main__":
    main()
