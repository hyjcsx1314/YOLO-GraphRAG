// 1. Count entity types (no destructive queries)
MATCH (n:Entity) RETURN n.kind AS kind, count(*) AS count ORDER BY count DESC;

// 2. Count relationship types
MATCH (:Entity)-[r]->(:Entity) RETURN type(r) AS relation, count(*) AS count ORDER BY count DESC;

// 3. Show actual evidence on edges
MATCH (a:Entity)-[r]->(b:Entity)
RETURN a.name AS from_entity, type(r) AS relation, b.name AS to_entity,
       r.paper_id AS paper, r.page AS pdf_page, r.quote AS supporting_quote
LIMIT 30;

// 4. Graph visualization for paper or method anchor. Change 'PGI' to an existing entity.
MATCH path = (anchor:Entity)-[*1..2]-(other:Entity)
WHERE anchor.name = 'PGI'
RETURN path LIMIT 25;
