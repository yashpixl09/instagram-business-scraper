"""Postgres edge of the lead engine.

Everything in this package talks to a database and is therefore excluded from the purity
rule that governs `lead_engine/*.py`. The dependency only ever points one way: the pure
core never imports from here, and nothing here imports the core in order to compute a
value the database is supposed to store. `businesses.dedupe_key` is the standing example --
`lead_engine.dedupe` derives it, this layer stores and constrains it.
"""
