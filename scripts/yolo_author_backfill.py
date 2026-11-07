#!/usr/bin/env python3
"""Add verified authorships of four YOLO papers to existing Neo4j Entity graph.

No LLM calls, no Milvus write. Dry-run by default. Uses the default Neo4j
home database unless NEO4J_DATABASE is explicitly set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import unicodedata
from collections import defaultdict

# Paper first-page author names checked against supplied PDFs and arXiv abstracts.
PAPERS = {
    "YOLOX": {
        "arxiv": "2107.08430",
        "authors": ["Zheng Ge", "Songtao Liu", "Feng Wang", "Zeming Li", "Jian Sun"],
    },
    "YOLOv7": {
        "arxiv": "2207.02696",
        "authors": ["Chien-Yao Wang", "Alexey Bochkovskiy", "Hong-Yuan Mark Liao"],
    },
    "YOLOv9": {
        "arxiv": "2402.13616",
        "authors": ["Chien-Yao Wang", "I-Hau Yeh", "Hong-Yuan Mark Liao"],
    },
    "YOLOv10": {
        "arxiv": "2405.14458",
        "authors": ["Ao Wang", "Hui Chen", "Lihao Liu", "Kai Chen",
                    "Zijia Lin", "Jungong Han", "Guiguang Ding"],
    },
}


def author_key(name: str) -> str:
    return "author::" + " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def rows_for_import() -> list[dict]:
    rows = []
    for paper, info in PAPERS.items():
        for order, author in enumerate(info["authors"], 1):
            evidence_id = hashlib.sha256(
                (paper + "|AUTHORED_BY|" + author).encode("utf-8")
            ).hexdigest()[:24]
            rows.append({
                "paper": paper,
                "author": author,
                "author_key": author_key(author),
                "order": order,
                "evidence_id": evidence_id,
                "arxiv": info["arxiv"],
                "url": "https://arxiv.org/abs/" + info["arxiv"],
                "page": 1,
            })
    return rows


def driver_for_env():
    from neo4j import GraphDatabase
    uri = os.environ.get("NEO4J_URI", "").strip()
    user = os.environ.get("NEO4J_USERNAME", "").strip()
    password = os.environ.get("NEO4J_PASSWORD", "").strip()
    if not (uri and user and password):
        raise RuntimeError("请设置 NEO4J_URI、NEO4J_USERNAME、NEO4J_PASSWORD")
    if not uri.startswith(("neo4j+s://", "neo4j+ssc://", "bolt+s://", "bolt+ssc://")):
        raise RuntimeError("Neo4j URI 必须使用加密连接")
    return GraphDatabase.driver(uri, auth=(user, password))


def database_name() -> str | None:
    return os.environ.get("NEO4J_DATABASE", "").strip() or None


def get_existing_papers(driver) -> dict[str, str]:
    records, _, _ = driver.execute_query(
        "MATCH (p:Entity:Paper) RETURN p.name AS name, p.key AS key",
        database_=database_name(),
    )
    found = {}
    for r in records:
        if r["name"] in found:
            raise RuntimeError("Paper 名称重复：" + r["name"])
        found[r["name"]] = r["key"]
    absent = set(PAPERS) - set(found)
    if absent:
        raise RuntimeError("数据库缺少预期 Paper 节点：" + ", ".join(sorted(absent)))
    return found


def stats(driver):
    q = {
        "nodes": "MATCH (n:Entity) RETURN count(n) AS n",
        "edges": "MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS n",
        "authors": "MATCH (a:Entity:Author) RETURN count(a) AS n",
        "authorships": "MATCH (:Entity:Paper)-[r:AUTHORED_BY]->(:Entity:Author) RETURN count(r) AS n",
    }
    result = {}
    for k, statement in q.items():
        rows, _, _ = driver.execute_query(statement, database_=database_name())
        result[k] = rows[0]["n"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ingest", action="store_true", help="实际写入；幂等")
    group.add_argument("--check", action="store_true", help="只读取 Neo4j 并显示现状")
    args = parser.parse_args()
    rows = rows_for_import()
    unique_authors = len({x["author_key"] for x in rows})
    print(f"VERIFIED FIRST-PAGE AUTHORSHIPS: {len(rows)}")
    print(f"UNIQUE AUTHOR NAMES: {unique_authors}")
    print("SOURCES:", ", ".join(f"{k}={v['arxiv']}" for k, v in PAPERS.items()))
    with driver_for_env() as driver:
        driver.verify_connectivity()
        papers = get_existing_papers(driver)
        before = stats(driver)
        print("EXISTING PAPER NODES:", ", ".join(sorted(papers)))
        print("BEFORE:", json.dumps(before, ensure_ascii=False))
        if args.check:
            return
        if not args.ingest:
            print("DRY RUN: no database changes made; re-run with --ingest after checking authors")
            for x in rows:
                print(f"{x['paper']}  --AUTHORED_BY-->  {x['author']}  (order={x['order']})")
            return
        # All four Paper nodes were matched before any write. This step creates
        # only Author nodes and AUTHORED_BY relationships. Never creates Papers.
        statement = """
        MATCH (p:Entity:Paper {key: $paper_key})
        MERGE (a:Entity:Author {key: $author_key})
          ON CREATE SET a.name = $author_name, a.kind = 'Author'
        MERGE (p)-[r:AUTHORED_BY {evidence_id: $evidence_id}]->(a)
          ON CREATE SET r.paper_id = $paper_id,
                        r.page = 1,
                        r.author_order = $author_order,
                        r.source = 'paper_first_page',
                        r.source_url = $source_url,
                        r.quote = $author_name,
                        r.chunk_id = $paper_id + '_PDF_P1_AUTHOR',
                        r.llm_model = 'manual_verified'
        RETURN a.name AS author_name
        """
        for x in rows:
            result, _, _ = driver.execute_query(
                statement, paper_key=papers[x["paper"]], author_key=x["author_key"],
                author_name=x["author"], evidence_id=x["evidence_id"],
                paper_id=x["paper"], author_order=x["order"], source_url=x["url"],
                database_=database_name(),
            )
            if len(result) != 1 or result[0]["author_name"] != x["author"]:
                raise RuntimeError("导入结果异常: " + x["paper"] + " / " + x["author"])
        after = stats(driver)
        assert after["authors"] >= unique_authors
        assert after["authorships"] >= len(rows)
        print("AFTER:", json.dumps(after, ensure_ascii=False))
        print("CHANGES: nodes=", after["nodes"]-before["nodes"],
              "edges=", after["edges"]-before["edges"])
        print("NEO4J AUTHOR BACKFILL PASSED")


if __name__ == "__main__":
    main()
