"""Neo4j 只读图谱展示：匹配原项目 (:Entity {kind, name, key}) Schema。"""
from __future__ import annotations

import os
from typing import Any


def configured() -> bool:
    return all([
        os.getenv("NEO4J_URI"),
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD"),
    ])


def read_live_graph(limit: int = 70) -> dict[str, Any]:
    if not configured():
        raise RuntimeError(
            "缺少 NEO4J_URI、NEO4J_USERNAME 或 NEO4J_PASSWORD"
        )

    from neo4j import GraphDatabase, READ_ACCESS

    uri = os.environ["NEO4J_URI"]
    if not uri.startswith(("neo4j+s://", "neo4j+ssc://", "bolt+s://", "bolt+ssc://")):
        raise RuntimeError("Neo4j URI 必须使用 TLS 加密连接")
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    database = os.getenv("NEO4J_DATABASE", "").strip() or None

    def read_query(tx, statement: str, **params):
        return [row.data() for row in tx.run(statement, **params)]

    with GraphDatabase.driver(
        uri, auth=(username, password), connection_timeout=12
    ) as driver:
        driver.verify_connectivity()
        with driver.session(
            database=database,
            default_access_mode=READ_ACCESS,
        ) as session:
            node_types = session.execute_read(
                read_query,
                """MATCH (n:Entity)
                   RETURN coalesce(n.kind, 'Unknown') AS type,
                          count(*) AS count
                   ORDER BY count DESC""",
            )
            rel_types = session.execute_read(
                read_query,
                """MATCH (:Entity)-[r]->(:Entity)
                   RETURN type(r) AS type, count(*) AS count
                   ORDER BY count DESC""",
            )
            edges = session.execute_read(
                read_query,
                """MATCH (s:Entity)-[r]->(t:Entity)
                   RETURN
                     elementId(s) AS source_id,
                     elementId(t) AS target_id,
                     coalesce(toString(s.name), toString(s.key),
                              elementId(s)) AS source,
                     coalesce(toString(t.name), toString(t.key),
                              elementId(t)) AS target,
                     coalesce(s.kind, 'Unknown') AS source_type,
                     coalesce(t.kind, 'Unknown') AS target_type,
                     type(r) AS relation
                   LIMIT $limit""",
                limit=max(1, min(200, int(limit))),
            )
    return {
        "node_types": node_types,
        "rel_types": rel_types,
        "edges": edges,
        "nodes": sum(int(item["count"]) for item in node_types),
        "relationships": sum(int(item["count"]) for item in rel_types),
    }
