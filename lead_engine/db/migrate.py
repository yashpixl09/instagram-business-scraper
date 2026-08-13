"""Apply `migrations/*.sql` in sorted order, once each.

One transaction per file: a migration either lands whole or not at all, and the row
recording it lands in that same transaction, so the ledger cannot disagree with the schema
even if the process is killed mid-run.

There are no down-migrations. Until this system is in production `docker compose down -v`
is the rollback, and a reversible migration that is never run is untested code carrying the
authority of a rollback plan.

    python -m lead_engine.db.migrate      # DSN from LEAD_ENGINE_DSN
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg

MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"

# Session-level, so one lock spans every per-file transaction below. Two workers booting
# together would otherwise both read an empty ledger and both apply 0002, and the loser
# fails on "relation already exists" with half a schema behind it. The waiter blocks here
# and then, under READ COMMITTED, reads the winner's committed ledger and applies nothing.
LOCK = "SELECT pg_advisory_lock(hashtext('lead_engine_migrations'))"
UNLOCK = "SELECT pg_advisory_unlock(hashtext('lead_engine_migrations'))"

LEDGER = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  version    text PRIMARY KEY,
  applied_at timestamptz NOT NULL DEFAULT now()
)
"""


def migration_files(directory: Path | None = None) -> list[Path]:
    """Every migration, in application order. Zero-padded names make sorted() the order."""
    return sorted((directory or MIGRATIONS).glob("*.sql"))


def apply_migrations(conn: psycopg.Connection, directory: Path | None = None) -> list[str]:
    """Apply whatever is not yet recorded. Returns the versions applied by this call."""
    conn.execute(LOCK)
    try:
        conn.execute(LEDGER)
        conn.commit()
        done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        conn.commit()

        applied: list[str] = []
        for path in migration_files(directory):
            if path.stem in done:
                continue
            try:
                # No BEGIN in the files themselves: the connection is not in autocommit,
                # so the whole file plus its ledger row is already one transaction.
                conn.execute(path.read_text(encoding="utf-8"))
                conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.stem,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            applied.append(path.stem)
            print(f"applied {path.stem}")

        if not applied:
            print(f"nothing to apply -- {len(done)} migration(s) already recorded")
        return applied
    finally:
        conn.execute(UNLOCK)
        conn.commit()


def main() -> int:
    dsn = os.environ.get("LEAD_ENGINE_DSN")
    if not dsn:
        print("LEAD_ENGINE_DSN is not set (see env.example)", file=sys.stderr)
        return 2
    with psycopg.connect(dsn) as conn:
        apply_migrations(conn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
