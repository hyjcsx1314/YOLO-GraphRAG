#!/usr/bin/env python3
"""YOLO PDF figure retrieval: existing CLIP image vectors + new E5 metadata vectors.

Never modifies the QLoRA training data, the existing E5 text collection,
or the existing CLIP image collection. Tested with the stage-4 data schema.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import gc
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import sys
from urllib.parse import quote

# Stage-4 script must be available in the same scripts/ directory.
import yolo_clip_milvus as clip

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "milvus" / "hybrid_retrieval"
META_COLLECTION = "yolo_visual_meta_e5_v1"
META_MODEL = "intfloat/multilingual-e5-small"
META_DIM = 384
PAPERS = ("YOLOX", "YOLOv7", "YOLOv9", "YOLOv10")

# Fixed engineering configuration. NOT fitted to the 10 diagnostic questions.
WEIGHTS = {"clip_image": 0.20, "e5_caption": 0.55, "keyword": 0.25}
RRF_K = 10


def checks():
    rows = clip.source_records()  # SHA256 + path + A-grade verification.
    if len(rows) != 20:
        raise RuntimeError("Expected 20 A-grade figures/tables")
    print("Existing image collection:", clip.COLLECTION)
    print("New metadata collection:", META_COLLECTION)
    return rows


def table_text(rec):
    relative = rec.get("table_text_path")
    if not relative:
        return ""
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or rel.parts[0] != "table_text":
        raise ValueError(f"Unsafe table_text_path: {relative}")
    return (clip.DATA / rel).read_text(encoding="utf-8").strip()


def textual_document(rec):
    """Evidence fields are from the audited manifest, NOT from evaluation labels."""
    prefix = [
        f"Paper: {rec['paper_id']}. {rec['label']}. Type: {rec['type']}.",
        f"中文描述: {rec['description_zh']}",
        f"Keywords: {rec.get('keywords', '')}",
        f"Original caption: {rec['caption_pdf']}",
    ]
    table = table_text(rec)
    if table:
        prefix.append("PDF table extraction (retrieval only, verify numbers in source PDF):\n" + table)
    return "\n".join(prefix)


def e5_document(rec, tokenizer, max_tokens=440):
    raw = textual_document(rec)
    ids = tokenizer.encode(raw, add_special_tokens=False)
    truncated = len(ids) > max_tokens
    if truncated:
        raw = tokenizer.decode(
            ids[:max_tokens],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    doc = "passage: " + raw
    if len(tokenizer.encode(doc, add_special_tokens=True)) > 512:
        raise RuntimeError(f"Unexpected E5 input length for {rec['asset_id']}")
    if len(doc.encode("utf-8")) > 8192:
        raise RuntimeError(f"Milvus metadata field too long: {rec['asset_id']}")
    return doc, truncated


def validate_e5(vec):
    a = [float(x) for x in vec]
    if len(a) != META_DIM or not all(math.isfinite(x) for x in a):
        raise ValueError("Invalid E5 384-d vector")
    length = math.sqrt(sum(x*x for x in a))
    if abs(length - 1) > 0.005:
        raise ValueError(f"E5 vector not normalized: {length}")
    return a


def ensure_collection(client):
    from pymilvus import MilvusClient, DataType
    if client.has_collection(collection_name=META_COLLECTION):
        desc = client.describe_collection(collection_name=META_COLLECTION)
        names = {item["name"] for item in desc["fields"]}
        required = {"id", "asset_id", "paper_id", "pdf_page", "metadata_text",
                    "png_sha256", "text_vector"}
        if not required.issubset(names):
            raise RuntimeError("Metadata collection schema mismatch. Refusing overwrite")
        vec_field = next(x for x in desc["fields"] if x["name"] == "text_vector")
        dim = vec_field.get("params", {}).get("dim", vec_field.get("dim"))
        if dim is not None and int(dim) != META_DIM:
            raise RuntimeError("Metadata collection must be 384-dimensional")
        print("Metadata collection already exists; using idempotent upsert")
        return
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="asset_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="paper_id", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="pdf_page", datatype=DataType.INT64)
    schema.add_field(field_name="metadata_text", datatype=DataType.VARCHAR, max_length=8192)
    schema.add_field(field_name="png_sha256", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="text_vector", datatype=DataType.FLOAT_VECTOR, dim=META_DIM)
    params = client.prepare_index_params()
    params.add_index(field_name="text_vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(collection_name=META_COLLECTION, schema=schema, index_params=params)
    print("CREATED:", META_COLLECTION)


def verify_metadata(client, records):
    count_res = client.query(collection_name=META_COLLECTION, filter="",
                             output_fields=["count(*)"], consistency_level="Strong")
    total = int(count_res[0]["count(*)"])
    print("Metadata Milvus count:", total)
    if total != 20:
        raise RuntimeError(f"Expected 20 metadata vectors, found {total}")
    example = records[0]
    res = client.query(collection_name=META_COLLECTION,
                       filter=f'asset_id == "{example["asset_id"]}"',
                       output_fields=["asset_id", "png_sha256", "text_vector"],
                       limit=1, consistency_level="Strong")
    if len(res) != 1 or res[0]["png_sha256"] != example["png_sha256"]:
        raise RuntimeError("Metadata SHA256 readback mismatch")
    validate_e5(res[0]["text_vector"])
    print("E5 READBACK PASSED: stored vector dim=384")


def ingest():
    rows = checks()
    # Preserve the stage-4 real image collection and verify readback before proceeding.
    clip.verify_db()
    from sentence_transformers import SentenceTransformer
    print("Loading cached E5 on CPU:", META_MODEL)
    model = SentenceTransformer(META_MODEL, device="cpu")
    tokenizer = model.tokenizer
    docs, was_truncated = [], []
    for rec in rows:
        doc, truncated = e5_document(rec, tokenizer)
        docs.append(doc)
        if truncated:
            was_truncated.append(rec["asset_id"])
    vectors = model.encode(docs, batch_size=8, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True)
    dbrows = [dict(id=clip.stable_id(rec["asset_id"]), asset_id=rec["asset_id"],
                   paper_id=rec["paper_id"], pdf_page=int(rec["pdf_page"]),
                   metadata_text=doc, png_sha256=rec["png_sha256"],
                   text_vector=validate_e5(vec))
              for rec, doc, vec in zip(rows, docs, vectors)]
    client = clip.connect()
    try:
        ensure_collection(client)
        client.load_collection(collection_name=META_COLLECTION)
        for start in range(0, len(dbrows), 5):
            client.upsert(collection_name=META_COLLECTION, data=dbrows[start:start+5])
        verify_metadata(client, rows)
    finally:
        client.close()
    RESULTS.mkdir(parents=True, exist_ok=True)
    settings = {"image_collection": clip.COLLECTION,
                "metadata_collection": META_COLLECTION,
                "image_model": clip.IMG_MODEL, "multilingual_clip_query_model": clip.TXT_MODEL,
                "metadata_embedding_model": META_MODEL,
                "image_dim": 512, "metadata_dim": 384,
                "metadata_count": len(dbrows), "table_text_truncated": was_truncated,
                "manifest_sha256": clip.sha256_file(clip.MANIFEST),
                "weights_predeclared": WEIGHTS, "rrf_k": RRF_K,
                "built_local_time": datetime.now().astimezone().isoformat()}
    (RESULTS / "index_manifest.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    print("METADATA INDEX BUILD PASSED: 20 real 384-d E5 vectors in Milvus")


def _tokens(s: str) -> list[str]:
    """Small zero-dependency bilingual term tokenizer for 20 audited records."""
    chunks = re.findall(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*|[\u4e00-\u9fff]+", s.lower())
    tokens = []
    for chunk in chunks:
        if "\u4e00" <= chunk[0] <= "\u9fff":
            # Chinese 2/3-grams avoid external tokenizers and preserve named phrases.
            for n in (2, 3):
                tokens.extend(chunk[i:i+n] for i in range(max(0, len(chunk)-n+1)))
        else:
            tokens.append(chunk)
            if "-" in chunk or "_" in chunk:
                tokens.extend(x for x in re.split("[-_]", chunk) if len(x) >= 2)
    return tokens


def lexical_index(records):
    # Keep captions and curated bilingual descriptions; NO evaluation expected IDs.
    docs = [_tokens(" ".join([
        r["paper_id"], str(r["label"]), r["description_zh"],
        r.get("keywords", ""), r["caption_pdf"]
    ])) for r in records]
    n = len(docs)
    df = Counter()
    for doc in docs:
        df.update(set(doc))
    def rank(question):
        query = set(_tokens(question))
        scored = []
        for r, doc in zip(records, docs):
            counts = Counter(doc)
            total = 0.0
            for t in query:
                tf = counts.get(t, 0)
                if tf:
                    idf = math.log(1.0 + (n - df[t] + 0.5)/(df[t] + 0.5))
                    total += idf * (2.2 * tf) / (tf + 1.2)
            if total > 0.0:
                scored.append((r["asset_id"], total))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored
    return rank


def explicit_paper(question: str) -> str | None:
    # A generic metadata constraint, NEVER a mapping from questions to expected assets.
    text = question.lower()
    named = [p for p in PAPERS if re.search(
        r"(?<![a-z0-9])" + re.escape(p.lower()) + r"(?![a-z0-9])", text)]
    return named[0] if len(named) == 1 else None


def fusion(clip_rank: list[dict], e5_rank: list[dict], lex_rank: list[tuple],
           records: list[dict], paper: str | None, topk: int):
    allowed = {r["asset_id"] for r in records if paper is None or r["paper_id"] == paper}
    image = [x for x in clip_rank if x["asset_id"] in allowed]
    text = [x for x in e5_rank if x["asset_id"] in allowed]
    lexical = [(aid, score) for aid, score in lex_rank if aid in allowed]
    ranks = {
        "clip_image": {x["asset_id"]: i+1 for i, x in enumerate(image)},
        "e5_caption": {x["asset_id"]: i+1 for i, x in enumerate(text)},
        "keyword": {aid: i+1 for i, (aid, _) in enumerate(lexical)},
    }
    img_score = {x["asset_id"]: x["score"] for x in image}
    meta_score = {x["asset_id"]: x["score"] for x in text}
    lookup = {r["asset_id"]: r for r in records}
    final = []
    for aid in allowed:
        rr = {name: ranks[name].get(aid) for name in WEIGHTS}
        score = sum(WEIGHTS[name] / (RRF_K + rank)
                    for name, rank in rr.items() if rank is not None)
        final.append({"asset_id": aid, "paper_id": lookup[aid]["paper_id"],
                      "pdf_page": lookup[aid]["pdf_page"],
                      "label": lookup[aid]["label"],
                      "description_zh": lookup[aid]["description_zh"],
                      "caption_pdf": lookup[aid]["caption_pdf"],
                      "image_path": lookup[aid]["image_path"],
                      "image_clip_cosine": img_score.get(aid),
                      "metadata_e5_cosine": meta_score.get(aid),
                      "rrf_score": score, "channel_ranks": rr})
    final.sort(key=lambda x: (-x["rrf_score"], x["asset_id"]))
    for pos, hit in enumerate(final[:topk], 1):
        hit["rank"] = pos
    return final[:topk]


def retrieval(questions: list[str], topk: int = 5):
    if not 1 <= topk <= 20:
        raise ValueError("topk must be between 1 and 20")
    records = checks()
    lexical = lexical_index(records)
    probe = clip.connect()
    try:
        if not probe.has_collection(collection_name=META_COLLECTION):
            raise RuntimeError("Metadata E5 collection missing. Run ingest first")
        if not probe.has_collection(collection_name=clip.COLLECTION):
            raise RuntimeError("CLIP image collection missing. Run prior stage ingest first")
    finally:
        probe.close()
    print("Retrieving existing 512-d image vectors with multilingual CLIP")
    clip_rankings = clip.search_many(questions, topk=20)
    gc.collect()
    from sentence_transformers import SentenceTransformer
    print("Retrieving 384-d figure captions with E5 on CPU")
    model = SentenceTransformer(META_MODEL, device="cpu")
    evecs = model.encode(["query: " + q for q in questions], batch_size=8,
                         normalize_embeddings=True, convert_to_numpy=True)
    vectors = [validate_e5(vec) for vec in evecs]
    client = clip.connect()
    try:
        if not client.has_collection(collection_name=META_COLLECTION):
            raise RuntimeError("Metadata E5 collection missing. Run ingest first")
        client.load_collection(collection_name=META_COLLECTION)
        found = client.search(collection_name=META_COLLECTION, data=vectors,
                              anns_field="text_vector", search_params={"metric_type": "COSINE"},
                              limit=20, output_fields=["asset_id", "paper_id", "pdf_page"],
                              consistency_level="Strong")
    finally:
        client.close()
    e5_rankings = [
        [{"asset_id": x["entity"]["asset_id"], "score": float(x["distance"])}
         for x in group] for group in found
    ]
    reports = []
    for q, cr, er in zip(questions, clip_rankings, e5_rankings):
        kw = lexical(q)
        paper = explicit_paper(q)
        general = fusion(cr, er, kw, records, None, topk)
        routed = fusion(cr, er, kw, records, paper, topk)
        reports.append({"question": q, "paper_route": paper,
                        "clip_global_ids": [x["asset_id"] for x in cr],
                        "e5_global_ids": [x["asset_id"] for x in er],
                        "lexical_ids": [aid for aid, _ in kw],
                        "fused_global": general, "fused_routed": routed})
    return reports


def count_metrics(expected_groups, rankings):
    n = len(expected_groups)
    ks = [1, 3, 5]
    scores = {}
    for k in ks:
        v = sum(bool(set(expected).intersection(ids[:k])) for expected, ids in
                zip(expected_groups, rankings))
        scores[f"Hit@{k}"] = f"{v}/{n} ({v/n:.1%})"
    return scores


def results_report(records, name, comparisons=None):
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    jf = RESULTS / (stem + ".json")
    hf = RESULTS / (stem + ".html")
    output = {
        "methods": {
            "clip": "pre-existing, real 512-d CLIP image vectors in Milvus",
            "e5": "new real 384-d caption/description/table-text E5 vectors in Milvus",
            "lexical": "BM25-like token scoring from audited captions and descriptions",
            "rrf_weights": WEIGHTS, "rrf_k": RRF_K,
            "routed": "when exactly one paper name occurs, filter results to that paper",
        },
        "results": records,
        "comparison": comparisons,
        "time_local": datetime.now().astimezone().isoformat(),
    }
    jf.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    css = """body{max-width:1100px;margin:26px auto;padding:0 16px;font-family:system-ui,'Noto Sans SC',sans-serif;background:#f7fafc;color:#203342}article{display:grid;grid-template-columns:265px 1fr;gap:20px;border:1px solid #d7e3e9;border-radius:12px;background:white;padding:12px;margin:12px 0}img{width:100%;max-height:230px;object-fit:contain;background:white;border:1px solid #dce4e8}h2{margin-top:30px}p{line-height:1.5}code{background:#e5edf2;border-radius:4px;padding:2px 4px}.muted{color:#536777}small{color:#536777}@media(max-width:680px){article{display:block}}"""
    parts = ["<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>",
             "<title>YOLO hybrid retrieval</title><style>"+css+"</style></head><body>",
             "<h1>YOLO Hybrid Visual Retrieval</h1>",
             "<p class='muted'>真实 Milvus CLIP 图像向量 + E5 图表文本向量；RRF 分数不是概率，表格数字须核对原文 PDF。</p>"]
    if comparisons:
        parts.append("<h2>10-case development diagnostics (not blind test)</h2><pre>"+
                     html.escape(json.dumps(comparisons, ensure_ascii=False, indent=2))+"</pre>")
    for case in records:
        parts.append("<h2>"+html.escape(case["question"])+"</h2>")
        if case.get("expected_asset_ids"):
            parts.append("<p class='muted'>人工标注期望图表："+html.escape(", ".join(case["expected_asset_ids"]))+"</p>")
        parts.append("<p class='muted'>论文路由："+html.escape(case.get("paper_route") or "none")+"</p>")
        for hit in case["fused_routed"]:
            img = quote(os.path.relpath(clip.DATA / hit["image_path"], hf.parent).replace(os.sep,"/"),safe="/")
            title = f"Top {hit['rank']} {hit['asset_id']} | {hit['paper_id']} {hit['label']} | PDF p.{hit['pdf_page']}"
            scores = (f"RRF={hit['rrf_score']:.6f} | CLIP={hit['image_clip_cosine']:.4f} | "
                      f"E5={hit['metadata_e5_cosine']:.4f} | "
                      f"ranks={hit['channel_ranks']}")
            parts.append("<article><a href='"+img+"'><img src='"+img+"'></a><div><b>"+
                         html.escape(title)+"</b><p><code>"+html.escape(scores)+"</code></p><p>"+
                         html.escape(hit["description_zh"])+"</p><details><summary>Original caption</summary><small>"+
                         html.escape(hit["caption_pdf"])+"</small></details></div></article>")
    parts.append("</body></html>")
    hf.write_text("".join(parts), encoding="utf-8")
    print("Result JSON:", jf)
    print("Image gallery:", hf)


def search(question, topk, routing):
    rows = retrieval([question], topk=topk)
    case = rows[0]
    selected = case["fused_routed"] if routing == "auto" else case["fused_global"]
    print("Question:", question)
    print("Auto paper route:", case["paper_route"] or "none",
          "| active:", routing == "auto")
    for h in selected:
        print(f"Top {h['rank']}: {h['asset_id']} | {h['paper_id']} {h['label']} | "
              f"PDF p.{h['pdf_page']} | RRF={h['rrf_score']:.6f} | "
              f"CLIP={h['image_clip_cosine']:.4f} | E5={h['metadata_e5_cosine']:.4f} | "
              f"ranks={h['channel_ranks']}")
    case["fused_routed"] = selected
    results_report(rows, "search")


def evaluate(topk):
    if topk < 5 or topk > 20:
        raise ValueError("For Hit@1/3/5 evaluation, choose --topk 5..20")
    tests = json.loads(clip.TESTS.read_text(encoding="utf-8"))
    if len(tests) != 10:
        raise ValueError("Expected 10 fixed diagnostic questions")
    rows = retrieval([c["question_zh"] for c in tests], topk=topk)
    expected, rankings = [], {k: [] for k in
                            ("CLIP only", "E5 caption only", "Fusion global", "Fusion + paper routing")}
    for c, case in zip(tests, rows):
        gold = c["expected_asset_ids"]
        expected.append(gold)
        case["expected_asset_ids"] = gold
        rankings["CLIP only"].append(case["clip_global_ids"])
        rankings["E5 caption only"].append(case["e5_global_ids"])
        rankings["Fusion global"].append([h["asset_id"] for h in case["fused_global"]])
        rankings["Fusion + paper routing"].append([h["asset_id"] for h in case["fused_routed"]])
        print(f"{c['question_zh']} | target={gold} | image={case['clip_global_ids'][0]} | "
              f"e5={case['e5_global_ids'][0]} | fusion={case['fused_global'][0]['asset_id']} | "
              f"route={case['fused_routed'][0]['asset_id']}")
    metrics = {name: count_metrics(expected, ranks) for name, ranks in rankings.items()}
    print("EVALUATION:", json.dumps(metrics, ensure_ascii=False, indent=2))
    results_report(rows, "evaluation", metrics)
    print("These 10 questions are development diagnostics already inspected during CLIP failure analysis;")
    print("do not claim unseen-test generalization or statistical significance.")


def main():
    parser = argparse.ArgumentParser(description="YOLO image+caption Milvus hybrid retrieval")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("check")
    sub.add_parser("ingest")
    sp = sub.add_parser("search")
    sp.add_argument("--question", required=True)
    sp.add_argument("--topk", type=int, default=5)
    sp.add_argument("--routing", choices=["auto", "off"], default="auto")
    ep = sub.add_parser("evaluate")
    ep.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    if args.action == "check":
        checks()
        print("CHECK PASSED")
    elif args.action == "ingest":
        ingest()
    elif args.action == "search":
        search(args.question, args.topk, args.routing)
    elif args.action == "evaluate":
        evaluate(args.topk)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
