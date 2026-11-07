#!/usr/bin/env python3
"""按用户已冻结的论文页面清单，将人工审订的 12 条留出数据归入验证/测试，并生成不可变指纹。

本脚本不修改任何学术答案，也不自动评判正确性。
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default=".")
    ap.add_argument("--verify",action="store_true")
    args=ap.parse_args()

    project=Path(args.project).expanduser().resolve()
    data=project/"data"/"datasets"
    locked=data/"eval_frozen_v1"
    freeze_record=locked/"freeze_record.json"
    manifest=data/"split_manifest_v3.json"

    if args.verify:
        record=json.loads(freeze_record.read_text(encoding="utf-8"))
        for filename, wanted in record["sha256"].items():
            target=(locked/filename) if filename.endswith(".jsonl") else manifest
            assert target.exists(), f"文件已缺失: {target}"
            assert digest(target)==wanted, f"冻结后内容被改变: {target}"
        print("冻结指纹校验通过：验证、测试题目及原数据划分清单未改变。")
        return

    if locked.exists() and any(locked.iterdir()):
        raise SystemExit("已存在评测冻结目录，请勿覆盖。若需要检查使用 --verify。")

    assert manifest.is_file(), f"找不到本地页面划分: {manifest}"
    splits=json.loads(manifest.read_text(encoding="utf-8"))
    pool=load_jsonl(Path(__file__).resolve().parent/"reviewed_holdout_pool_12.jsonl")

    train_pages=set()
    for paper,groups in splits.items():
        for page in groups.get("train",[]):
            train_pages.add((paper,int(page)))

    grouped={"val":[],"test":[]}
    for item in pool:
        key=(item["paper_id"],int(item["page"]))
        matches=[
            split for split in ("train","val","test")
            if int(item["page"]) in splits.get(item["paper_id"],{}).get(split,[])
        ]
        assert len(matches)==1, f"缺少或重复分组信息: {key}"
        assert matches[0]!="train", f"评测样本被分入训练集，停止: {key}"
        item["split"]=matches[0]
        item["freeze_version"]="v1"
        grouped[matches[0]].append(item)

    assert len(pool)==12
    assert len(grouped["val"])==6 and len(grouped["test"])==6, (
        f"与本机此前报告的 val/test 各6条不符: "
        f"{len(grouped['val'])}/{len(grouped['test'])}"
    )
    holdout_pages={(r["paper_id"],int(r["page"])) for r in pool}
    assert train_pages.isdisjoint(holdout_pages), "验证/测试页与训练页面交叉"

    # 二次确认实际 train_chunks_v3.jsonl 中不存在评测来源
    train_source=data/"train_chunks_v3.jsonl"
    if train_source.exists():
        train_real={(r["paper_id"],int(r["page"])) for r in load_jsonl(train_source)}
        assert train_real.isdisjoint(holdout_pages), "实际训练块与评测页发生泄漏"

    locked.mkdir(parents=True,exist_ok=False)
    sha={}
    for split,items in grouped.items():
        path=locked/f"eval_{split}_locked_v1.jsonl"
        path.write_text(
            "".join(json.dumps(r,ensure_ascii=False)+"\n" for r in items),
            encoding="utf-8"
        )
        sha[path.name]=digest(path)
        pages=sorted({(r["paper_id"],r["page"]) for r in items})
        print(f"{split}: {len(items)} 条，来源页面 {pages}")

    sha[manifest.name]=digest(manifest)
    record={
        "purpose":"YOLO QLoRA evaluation freeze",
        "version":"v1",
        "status":"frozen_before_formal_training",
        "counts":{"val":6,"test":6},
        "sha256":sha,
        "paper_page_isolation":True,
        "note":"冻结数据只用于评测，禁止合入训练集；学术问答需人工打分"
    }
    freeze_record.write_text(
        json.dumps(record,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8"
    )
    print("冻结记录:",freeze_record)
    print("之后请使用 --verify 校验，不要修改已冻结文件。")


if __name__=="__main__":
    main()
