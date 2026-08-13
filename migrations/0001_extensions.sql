-- Extensions.
--
-- pgvector backs the MEMORY plane: `memories.embedding` and `run_summaries.embedding`,
-- both vector(1536) with an HNSW index. It is created first because 0006 cannot parse
-- its own column types without it.
--
-- The design spec writes `CREATE EXTENSION vector;` bare. That form raises
-- "extension already exists" on the second run, which would make this file un-rerunnable
-- against a database created before the migration runner existed. IF NOT EXISTS is the
-- correction; the runner's own version ledger already skips applied files, so this only
-- matters for a database bootstrapped by hand.
--
-- Note the extension is database-scoped, not schema-scoped: it lands in the first schema
-- of search_path and its types are visible from anywhere that schema is on the path.

CREATE EXTENSION IF NOT EXISTS vector;
