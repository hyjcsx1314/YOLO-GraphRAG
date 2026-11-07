// 1. 整体规模（返回的结果不是局部 LIMIT 图谱数量）
MATCH (n:Entity) RETURN count(n) AS EntityCount;
MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS EvidenceRelationCount;

// 2. 四篇论文与其提出的模型/方法：可在 Graph 视图截图
MATCH (p:Paper)-[r:PROPOSES]->(m:Entity)
RETURN p,r,m LIMIT 60;

// 3. PGI 一跳：去除多个来源页码形成的重复语义边，仅用于可视化
MATCH (n:Entity)-[r]-(m:Entity)
WHERE toLower(n.name) = 'pgi'
WITH n,m,type(r) AS relation_type, collect(r) AS evidence_rels
RETURN n,head(evidence_rels) AS r,m;

// 4. PGI 一至二跳：真正返回 1～2 跳路径，图表用于课程报告
MATCH p=(n:Entity)-[*1..2]-(m:Entity)
WHERE toLower(n.name) = 'pgi'
RETURN p LIMIT 20;

// 5. 同一 PGI 语义关系关联的多个原文证据（页码不会丢失）
MATCH (n:Entity)-[r:HAS_COMPONENT]->(c:Component)
WHERE toLower(n.name) = 'pgi'
RETURN c.name AS component,
       collect(DISTINCT {paper:r.paper_id,page:r.page,quote:r.quote}) AS sources
ORDER BY component;
