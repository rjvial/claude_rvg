// Surgical wipe of LLM-entity-extraction data from Neo4j.
// Run once, after deleting extract.py and entities.jsonl (2026-05-25).
//
// Preserves:
//   - Layer 1  Messages / Threads / Persons / SENT / RECEIVED_BY / IN_THREAD
//   - Layer 1b REPLY_TO, NEXT_IN_THREAD, STARTS_WITH
//   - Layer 2  domain-derived Orgs + (:Person)-[:WORKS_AT]->(:Org)
//   - Layer 3b Concepts and (:Message|:Thread)-[:MENTIONS]->(:Concept)
//   - Layer 4  Events
//   - Matters, embeddings, full-text + vector indexes
//
// Deletes:
//   - All [:MENTIONS] edges whose target is :Org (Layer 3 fan-out)
//   - All [:DISCUSSES] edges (Layer 3 only)
//   - All :Topic nodes (already deprecated, but cleans any stragglers)
//   - :Org nodes that have no :WORKS_AT in/out (= pure-LLM orgs)
//   - The deprecated topic_name constraint
//
// Run via:
//   cypher-shell -u neo4j -p $NEO4J_PASSWORD -f scripts/wipe_llm_extraction.cypher
// Or paste into Neo4j Browser. All statements are idempotent.

// 1. Drop MENTIONS edges to Orgs (Concept MENTIONS are untouched — different target).
MATCH ()-[r:MENTIONS]->(:Org) DELETE r;

// 2. Drop DISCUSSES edges (only ever written by extract.py).
MATCH ()-[r:DISCUSSES]->() DELETE r;

// 3. Drop Topic nodes (the LLM Topic layer was already unloaded; this cleans any residue).
MATCH (t:Topic) DETACH DELETE t;

// 4. Drop Org nodes with no :WORKS_AT edges (= no longer attached to any Person; pure LLM extractions).
MATCH (o:Org)
WHERE NOT (:Person)-[:WORKS_AT]->(o)
  AND NOT (o)-[:WORKS_AT]->()
DETACH DELETE o;

// 5. Drop the topic_name constraint if a prior schema left it behind.
DROP CONSTRAINT topic_name IF EXISTS;

// Sanity check — should return zero rows for Topic and zero MENTIONS->Org edges.
MATCH (t:Topic) RETURN 'Topic residue' AS check, count(t) AS n
UNION ALL
MATCH ()-[r:MENTIONS]->(:Org) RETURN 'MENTIONS->Org residue' AS check, count(r) AS n
UNION ALL
MATCH ()-[r:DISCUSSES]->() RETURN 'DISCUSSES residue' AS check, count(r) AS n;
