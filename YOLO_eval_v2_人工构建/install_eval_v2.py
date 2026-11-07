#!/usr/bin/env python3
"""将人工构建的 V2 评测集安装并冻结，不修改既有 eval_frozen_v1。"""
import argparse
import hashlib
import json
from pathlib import Path


def load_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--project",default=".")
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    project=Path(args.project).expanduser().resolve()
    data=project/"data"/"datasets"
    target=data/"eval_frozen_v2"
    package=Path(__file__).resolve().parent

    if args.verify:
        record_file=target/"freeze_record_v2.json"
        if not record_file.exists():
            raise SystemExit("尚未创建 V2 冻结记录")
        record=json.loads(record_file.read_text(encoding="utf-8"))
        for relative, expected in record["files_sha256"].items():
            if relative.startswith("eval_frozen_v2/"):
                file=data/relative
            else:
                file=data/relative
            if not file.exists():
                raise SystemExit(f"冻结数据缺失: {file}")
            if sha(file)!=expected:
                raise SystemExit(f"冻结指纹发生变化: {file}")
        print("V2 数据冻结及页面划分指纹验证通过。")
        return

    if target.exists() and any(target.iterdir()):
        raise SystemExit("eval_frozen_v2 已存在，禁止覆盖。请使用 --verify 核查。")

    local_val=data/"val_chunks_v3.jsonl"
    local_test=data/"test_chunks_v3.jsonl"
    local_train=data/"train_chunks_v3.jsonl"
    manifest=data/"split_manifest_v3.json"
    reviewed_train=data/"qlora_train_reviewed.jsonl"
    for f in (local_val,local_test,local_train,manifest,reviewed_train):
        if not f.exists():
            raise SystemExit(f"缺少本机来源文件: {f}")

    val_chunks={r["chunk_id"]:r for r in load_jsonl(local_val)}
    test_chunks={r["chunk_id"]:r for r in load_jsonl(local_test)}
    train_chunks=load_jsonl(local_train)
    train_pages={(r["paper_id"],int(r["page"])) for r in train_chunks}
    page_manifest=json.loads(manifest.read_text(encoding="utf-8"))
    candidates={
        "val":load_jsonl(package/"eval_val_candidate_v2.jsonl"),
        "test":load_jsonl(package/"eval_test_candidate_v2.jsonl")
    }
    assert len(candidates["val"])==23 and len(candidates["test"])==37
    ids=set()
    pages_by_split={}

    for split, rows in candidates.items():
        sources=val_chunks if split=="val" else test_chunks
        pages=set()
        for row in rows:
            id=row["review_id"]
            if id in ids:
                raise SystemExit(f"重复评测编号: {id}")
            ids.add(id)
            chunk=sources.get(row["chunk_id"])
            if chunk is None:
                raise SystemExit(f"评测题不属于{split}来源文件: {row['chunk_id']}")
            page=(row["paper_id"],int(row["page"]))
            if page!=(chunk["paper_id"],int(chunk["page"])):
                raise SystemExit(f"评测题元数据与原始文本不一致: {id}")
            if int(row["page"]) not in page_manifest[row["paper_id"]][split]:
                raise SystemExit(f"评测页与原页面划分清单不一致: {id}")
            if page in train_pages:
                raise SystemExit(f"评测来源页面出现在训练数据中: {id}")
            for quote in row["evidence_spans"]:
                if " ".join(quote.split()).casefold() not in " ".join(row["input"].split()).casefold():
                    raise SystemExit(f"评测题证据不在输入中: {id}")
            if row["origin"].startswith("V2基于") and row["input"] != chunk["text"]:
                raise SystemExit(f"V2新增题输入与本机V3正文不一致: {id}")
            if row["task_type"] in ("structured_json","entity_json"):
                try:
                    if not isinstance(json.loads(row["output"]),dict):
                        raise ValueError()
                except (ValueError,TypeError):
                    raise SystemExit(f"JSON标准答案不合法: {id}")
            pages.add(page)
        pages_by_split[split]=pages

    if pages_by_split["val"] & pages_by_split["test"]:
        raise SystemExit("验证与测试使用了同一论文页面，已停止冻结")

    target.mkdir(parents=True,exist_ok=False)
    files_sha256={}
    for split,rows in candidates.items():
        file=target/f"eval_{split}_locked_v2.jsonl"
        file.write_text(
            "".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows),
            encoding="utf-8"
        )
        files_sha256[f"eval_frozen_v2/{file.name}"]=sha(file)

    for name in ["split_manifest_v3.json","val_chunks_v3.jsonl","test_chunks_v3.jsonl",
                 "train_chunks_v3.jsonl","qlora_train_reviewed.jsonl"]:
        files_sha256[name]=sha(data/name)

    record={
       "version":"V2",
       "status":"frozen_prior_to_formal_training",
       "counts":{"val":23,"test":37,"total":60},
       "origin":"V1 12 reviewed + 48 newly manually authored",
       "pages_by_split":{
            split:[{"paper_id":p,"page":page} for p,page in sorted(pages)]
            for split,pages in pages_by_split.items()
       },
       "files_sha256":files_sha256,
       "note":"不要把冻结题目的答案加入训练集；不要覆盖 V1 冻结基线。"
    }
    (target/"freeze_record_v2.json").write_text(
        json.dumps(record,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8"
    )
    print("V2 验证题:",len(candidates["val"]))
    print("V2 测试题:",len(candidates["test"]))
    print("冻结目录:",target)
    print("V1 的 eval_frozen_v1 和原模型已有结果未修改。")
    print("运行本程序 --verify 可以核对 SHA256 指纹。")


if __name__=="__main__":
    main()
