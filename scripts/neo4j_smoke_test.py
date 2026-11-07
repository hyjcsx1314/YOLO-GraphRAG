import os
from neo4j import GraphDatabase

uri = os.environ["NEO4J_URI"]
user = os.environ["NEO4J_USERNAME"]
password = os.environ["NEO4J_PASSWORD"]

with GraphDatabase.driver(
    uri, auth=(user, password)
) as driver:

    driver.verify_connectivity()
    print("NEO4J CONNECTION PASSED")

    with driver.session() as session:
        result = session.run(
            "RETURN 1 AS connected"
        ).single()

        assert result["connected"] == 1
        print("CYPHER QUERY PASSED")

        test_id = "yolo_graphrag_connectivity_test"

        result = session.run(
            """
            MERGE (n:RAGConnectivityTest {id: $id})
            RETURN n.id AS id
            """,
            id=test_id
        ).single()

        assert result["id"] == test_id
        print("NODE WRITE PASSED")

        session.run(
            """
            MATCH (n:RAGConnectivityTest {id: $id})
            DELETE n
            """,
            id=test_id
        ).consume()

        print("TEST NODE CLEANUP PASSED")

print("NEO4J SMOKE TEST PASSED")
