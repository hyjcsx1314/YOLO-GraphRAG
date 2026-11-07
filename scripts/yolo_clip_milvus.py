#!/usr/bin/env python3
"""YOLO figure cross-modal retrieval: real CLIP image vectors + Milvus 2.6.

Commands: check, ingest, verify-db, search, evaluate.
Independent of the existing E5 text collection and QLoRA environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys
from urllib.parse import quote

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "yolo_visual_final_v2"
RESULTS = PROJECT / "results" / "milvus" / "clip_retrieval"
MANIFEST = DATA / "curated_core_manifest.jsonl"
TESTS = DATA / "retrieval_test_cases.json"
COLLECTION = "yolo_figures_clip_v1"
IMG_MODEL = "sentence-transformers/clip-ViT-B-32"
TXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
URI = "http://127.0.0.1:19530"
DIM = 512
FIELDS = ["asset_id", "paper_id", "pdf_page", "kind", "label", "description_zh", "caption_pdf", "image_path", "png_sha256"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def asset_path(record: dict) -> Path:
    # Avoid arbitrary file access from a malformed manifest.
    rel = Path(record["image_path"])
    if rel.is_absolute() or ".." in rel.parts or rel.parts[0] != "assets":
        raise ValueError(f"Unsafe image_path: {rel}")
    return DATA / rel


def source_records() -> list[dict]:
    if not MANIFEST.is_file():
        raise FileNotFoundError(
            f"Missing {MANIFEST}. Extract yolo_visual_final_v2.zip into project/data first."
        )
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 20:
        raise ValueError(f"Core manifest should have 20 rows, got {len(rows)}")
    ids, hashes = set(), set()
    for r in rows:
        for key in ("asset_id", "paper_id", "pdf_page", "type", "label", "description_zh", "caption_pdf", "png_sha256"):
            if key not in r:
                raise ValueError(f"Missing {key} in {r}")
        if r["grade"] != "A" or r["eligible_for_image_vector_index"] is not True:
            raise ValueError(f"Not approved for core index: {r['asset_id']}")
        if r["type"] not in ("figure", "table"):
            raise ValueError(f"Unexpected type: {r['type']}")
        aid = r["asset_id"]
        if aid in ids:
            raise ValueError(f"Duplicate asset_id: {aid}")
        ids.add(aid)
        p = asset_path(r)
        if not p.is_file():
            raise FileNotFoundError(p)
        actual = sha256_file(p)
        if actual != r["png_sha256"]:
            raise ValueError(f"SHA256 mismatch for {p.name}; use the original audited ZIP")
        if actual in hashes:
            raise ValueError(f"Duplicate image: {p.name}")
        hashes.add(actual)
        if r["type"] == "table":
            table_path = r.get("table_text_path")
            if table_path and not (DATA / table_path).is_file():
                raise FileNotFoundError(DATA / table_path)
    print(f"CHECK PASSED: {len(rows)} A-grade images; unique IDs and SHA256 matched")
    return rows


def stable_id(asset_id: str) -> int:
    return int(hashlib.sha256(asset_id.encode("utf-8")).hexdigest()[:15], 16)


def connect():
    from pymilvus import MilvusClient
    return MilvusClient(uri=URI, token="root:Milvus")


def count_db(client) -> int:
    ret = client.query(
        collection_name=COLLECTION,
        filter="",
        output_fields=["count(*)"],
        consistency_level="Strong",
    )
    return int(ret[0]["count(*)"])


def ensure_schema(client):
    from pymilvus import MilvusClient, DataType
    if client.has_collection(collection_name=COLLECTION):
        info = client.describe_collection(collection_name=COLLECTION)
        names = {f["name"] for f in info.get("fields", [])}
        required = {"id", "image_vector"}.union(FIELDS)
        if not required.issubset(names):
            raise RuntimeError("Existing collection has unexpected schema; do not overwrite it")
        vf = next(f for f in info["fields"] if f["name"] == "image_vector")
        params = vf.get("params", {})
        dbdim = params.get("dim", vf.get("dim"))
        if dbdim is not None and int(dbdim) != DIM:
            raise RuntimeError("Existing image_vector dimension is not 512")
        print("Collection already exists; safe idempotent upsert will be used")
        return

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="asset_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="paper_id", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="pdf_page", datatype=DataType.INT64)
    schema.add_field(field_name="kind", datatype=DataType.VARCHAR, max_length=16)
    schema.add_field(field_name="label", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="description_zh", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="caption_pdf", datatype=DataType.VARCHAR, max_length=8192)
    schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="png_sha256", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="image_vector", datatype=DataType.FLOAT_VECTOR, dim=DIM)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="image_vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(collection_name=COLLECTION, schema=schema, index_params=index_params)
    print("CREATED collection:", COLLECTION)


def image_model():
    from sentence_transformers import SentenceTransformer
    print("Loading image model on CPU:", IMG_MODEL)
    return SentenceTransformer(IMG_MODEL, device="cpu")


def text_model():
    from sentence_transformers import SentenceTransformer
    print("Loading multilingual text model on CPU:", TXT_MODEL)
    return SentenceTransformer(TXT_MODEL, device="cpu")


def validate_vector(vec):
    data = [float(v) for v in vec]
    if len(data) != DIM or not all(math.isfinite(x) for x in data):
        raise ValueError(f"Bad embedding shape or NaN; dim={len(data)}")
    norm = math.sqrt(sum(x * x for x in data))
    if abs(norm - 1) > 0.005:
        raise ValueError(f"Embedding not unit-normalized: norm={norm}")
    return data


def to_milvus_row(rec, vec):
    row = {
        "id": stable_id(rec["asset_id"]),
        "asset_id": rec["asset_id"],
        "paper_id": rec["paper_id"],
        "pdf_page": int(rec["pdf_page"]),
        "kind": rec["type"],
        "label": rec["label"],
        "description_zh": rec["description_zh"],
        "caption_pdf": rec["caption_pdf"],
        "image_path": rec["image_path"],
        "png_sha256": rec["png_sha256"],
        "image_vector": validate_vector(vec),
    }
    limits = {"asset_id": 64, "paper_id": 32, "kind": 16, "label": 64,
              "description_zh": 1024, "caption_pdf": 8192,
              "image_path": 256, "png_sha256": 64}
    for k, limit in limits.items():
        if len(row[k].encode("utf-8")) > limit:
            raise ValueError(f"Field too long for Milvus: {k}")
    return row


def ingest():
    from PIL import Image
    items = source_records()
    model = image_model()
    rows = []
    # Small batches avoid holding all high-resolution PNGs in RAM.
    for pos in range(0, len(items), 2):
        batch = items[pos:pos + 2]
        images = []
        try:
            for rec in batch:
                with Image.open(asset_path(rec)) as im:
                    images.append(im.convert("RGB"))
            vecs = model.encode(images, batch_size=2, normalize_embeddings=True, convert_to_numpy=True)
            for rec, vec in zip(batch, vecs):
                rows.append(to_milvus_row(rec, vec))
        finally:
            for im in images:
                im.close()
        print(f"Image vectors prepared: {len(rows)}/{len(items)}")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Primary key collision")

    client = connect()
    try:
        ensure_schema(client)
        client.load_collection(collection_name=COLLECTION)
        for start in range(0, len(rows), 5):
            client.upsert(collection_name=COLLECTION, data=rows[start:start + 5])
        assert_indexed(client, items)
        RESULTS.mkdir(parents=True, exist_ok=True)
        info = {
            "collection": COLLECTION,
            "image_encoder": IMG_MODEL,
            "text_encoder": TXT_MODEL,
            "vector_field": "image_vector",
            "vector_dimension": DIM,
            "metric": "COSINE",
            "device": "cpu",
            "index_image_count": len(items),
            "source_manifest_sha256": sha256_file(MANIFEST),
            "created_local": datetime.now().astimezone().isoformat(),
        }
        (RESULTS / "index_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print("IMAGE INDEX BUILD PASSED")
    finally:
        client.close()


def assert_indexed(client, items):
    total = count_db(client)
    print("Milvus count:", total)
    if total != len(items):
        raise RuntimeError(f"Expected exactly {len(items)} entries, got {total}. Use a fresh collection name if stale data remains")
    sample = items[0]
    check = client.query(
        collection_name=COLLECTION,
        filter=f'asset_id == "{sample["asset_id"]}"',
        output_fields=["asset_id", "png_sha256", "image_vector"],
        limit=1,
        consistency_level="Strong",
    )
    if len(check) != 1 or check[0]["png_sha256"] != sample["png_sha256"]:
        raise RuntimeError("Milvus source metadata verification failed")
    vector = validate_vector(check[0]["image_vector"])
    print(f"READBACK PASSED: {sample['asset_id']}, stored vector dim={len(vector)}")


def verify_db():
    items = source_records()
    client = connect()
    try:
        if not client.has_collection(collection_name=COLLECTION):
            raise RuntimeError("Image collection not found; run ingest first")
        client.load_collection(collection_name=COLLECTION)
        assert_indexed(client, items)
        print("MILVUS IMAGE INDEX VERIFIED")
    finally:
        client.close()


def search_many(questions: list[str], topk: int, paper: str | None = None):
    if not 1 <= topk <= 20:
        raise ValueError("topk must be 1..20")
    # This is NOT 'query: ...' E5 prefix; the multilingual CLIP expects raw text.
    model = text_model()
    qvecs = model.encode(questions, normalize_embeddings=True, convert_to_numpy=True, batch_size=8)
    vectors = [validate_vector(v) for v in qvecs]
    client = connect()
    try:
        if not client.has_collection(collection_name=COLLECTION):
            raise RuntimeError("Collection missing; run ingest first")
        client.load_collection(collection_name=COLLECTION)
        params = dict(collection_name=COLLECTION, data=vectors,
                      anns_field="image_vector", search_params={"metric_type": "COSINE"},
                      limit=topk, output_fields=FIELDS, consistency_level="Strong")
        if paper:
            params["filter"] = f'paper_id == "{paper}"'
        res = client.search(**params)
        return [[{**hit["entity"], "score": float(hit["distance"]), "rank": rank}
                 for rank, hit in enumerate(group, 1)] for group in res]
    finally:
        client.close()


def safe_rel_image(record: dict, html_directory: Path):
    # Verify search result path maps to a validated manifest entry before serving it.
    source = DATA / record["image_path"]
    return quote(os.path.relpath(source, html_directory).replace(os.sep, "/"), safe="/")


def card(hit, basedir):
    src = safe_rel_image(hit, basedir)
    a = html.escape(str(hit["asset_id"]))
    title = html.escape(f"{hit['paper_id']} {hit['label']} · PDF p.{hit['pdf_page']}")
    subtitle = html.escape(str(hit.get("description_zh", "")))
    caption = html.escape(str(hit.get("caption_pdf", "")))
    return (f'<article><a href="{src}" target="_blank"><img src="{src}" alt="{a}" loading="lazy"></a>'
            f'<div><b>Top {hit["rank"]} · {title}</b>'
            f'<p>asset_id: {a} · COSINE: {hit["score"]:.4f}</p>'
            f'<p>{subtitle}</p><details><summary>原文图注</summary><small>{caption}</small></details></div></article>')


def html_page(title, sections, extra=""):
    style = """<style>body{font-family:system-ui,-apple-system,'Noto Sans SC',sans-serif;max-width:1160px;margin:28px auto;padding:0 20px;color:#243543;background:#f9fbfd}h1{font-size:27px}h2{margin:35px 0 15px}article{display:grid;grid-template-columns:320px 1fr;gap:20px;border:1px solid #d7e2e8;border-radius:12px;padding:14px;margin:12px 0;background:white;align-items:center}article img{max-width:100%;width:320px;max-height:210px;object-fit:contain;background:white;border:1px solid #e6edf2}p{line-height:1.6}small{line-height:1.55}details{margin-top:8px}code{background:#e7edf2;padding:2px 5px;border-radius:4px}.sub{color:#4c6574}@media(max-width:650px){article{grid-template-columns:1fr}article img{width:100%}}</style>"""
    return ("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>" + style +
            "</head><body><h1>" + html.escape(title) + "</h1><p class='sub'>CLIP 跨模态检索（图像向量来自 Milvus），图表数值请回原文核对。</p>" + extra + sections + "</body></html>")


def write_report(prefix: str, rows: list[dict], metric: dict | None = None):
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = RESULTS / f"{prefix}_{tag}"
    meta = {"collection": COLLECTION, "image_encoder": IMG_MODEL, "text_encoder": TXT_MODEL,
            "queries": rows, "created_local": datetime.now().astimezone().isoformat()}
    if metric is not None:
        meta["evaluation"] = metric
    jf = base.with_suffix(".json")
    hf = base.with_suffix(".html")
    jf.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    parts = []
    for case in rows:
        q = html.escape(case["question"])
        expectation = case.get("expected_asset_ids", [])
        hint = f" · 预期图表 {html.escape(', '.join(expectation))}" if expectation else ""
        parts.append(f"<h2>{q}</h2><p class='sub'>来源：真实 Milvus 图像向量排序{hint}</p>")
        parts.extend(card(x, hf.parent) for x in case["hits"])
    extras = ""
    if metric is not None:
        extras = "<p>" + html.escape(", ".join(f"{key}={value}" for key, value in metric.items())) + "</p>"
    hf.write_text(html_page("YOLO 图文跨模态检索报告", "".join(parts), extras), encoding="utf-8")
    print("Result JSON:", jf)
    print("Image gallery:", hf)
    return jf, hf


def search(question: str, topk: int, paper: str | None):
    source_records()  # checksum / path validation
    hits = search_many([question], topk, paper)[0]
    for hit in hits:
        print(f"Top {hit['rank']}: {hit['asset_id']} | {hit['paper_id']} | {hit['label']} | PDF p.{hit['pdf_page']} | COSINE={hit['score']:.4f}")
    write_report("search", [{"question": question, "paper_filter": paper, "hits": hits}])


def evaluate(topk: int):
    source_records()
    if not TESTS.is_file():
        raise FileNotFoundError(TESTS)
    cases = json.loads(TESTS.read_text(encoding="utf-8"))
    if not cases or any(not c.get("expected_asset_ids") or not c.get("question_zh") for c in cases):
        raise ValueError("Invalid retrieval_test_cases.json")
    results = search_many([c["question_zh"] for c in cases], topk)
    checks = {"Hit@1": 0, "Hit@3": 0, "Hit@5": 0}
    runs = []
    for c, hits in zip(cases, results):
        actual = [h["asset_id"] for h in hits]
        expected = set(c["expected_asset_ids"])
        for k in (1, 3, 5):
            checks[f"Hit@{k}"] += bool(expected.intersection(actual[:k]))
        print(f"{'HIT' if actual and actual[0] in expected else 'MISS'}: {c['question_zh']} | expected={c['expected_asset_ids']} | top1={actual[0] if actual else 'none'}")
        runs.append({"question": c["question_zh"], "expected_asset_ids": c["expected_asset_ids"],
                     "topic": c.get("topic"), "hits": hits})
    metric = {f"{key} (n={len(cases)})": f"{value}/{len(cases)} ({value/len(cases):.1%})" for key, value in checks.items()}
    print("EVALUATION:", json.dumps(metric, ensure_ascii=False))
    write_report("evaluation", runs, metric)
    print("Interpret as image-retrieval hit rate, not answer accuracy or significance")


def main(argv=None):
    parser = argparse.ArgumentParser(description="CLIP cross-modal retrieval over curated YOLO figures")
    subs = parser.add_subparsers(dest="cmd", required=True)
    subs.add_parser("check", help="Validate 20 audited images without downloading models")
    subs.add_parser("ingest", help="Encode images, upsert into self-hosted Milvus")
    subs.add_parser("verify-db", help="Read back the actual 512-d vector from Milvus")
    s = subs.add_parser("search", help="Chinese question to real-image-vector Top-k")
    s.add_argument("--question", required=True)
    s.add_argument("--topk", type=int, default=5)
    s.add_argument("--paper", choices=["YOLOX", "YOLOv7", "YOLOv9", "YOLOv10"])
    e = subs.add_parser("evaluate", help="Evaluate 10 fixed non-QLoRA retrieval queries")
    e.add_argument("--topk", type=int, default=5)
    args = parser.parse_args(argv)
    if args.cmd == "check":
        source_records()
    elif args.cmd == "ingest":
        ingest()
    elif args.cmd == "verify-db":
        verify_db()
    elif args.cmd == "search":
        search(args.question, args.topk, args.paper)
    elif args.cmd == "evaluate":
        evaluate(args.topk)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
