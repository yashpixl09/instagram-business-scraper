"""The search budget ledger.

SearchAPI's searches are granted once per key and never renew, so the interesting failures
are not "the counter is off by one". They are:

  * two workers both spent the last credit (`SpendIsAtomic`), and
  * the ledger was seeded with more credits than the key actually has (`AllowanceIsCut`).

Both are invisible to a single-threaded test against a fake connection, because both are
properties of Postgres rather than of `budget.py`. Those classes use a real database and
skip outright when there is none. Versions of them that could pass without one would be
worthless.

The rest of the file splits by what it actually needs:

  * `BudgetContract` uses a stub connection. It checks argument validation and the shape of
    the SQL -- specifically that the limit check is IN the UPDATE, which is the whole safety
    property and is checkable without a database.
  * `SplitArithmetic` is pure: cutting a total into whole searches.
  * everything else is integration, skipped unless LEAD_ENGINE_TEST_DSN is set, following
    tests/test_migrations.py: a throwaway schema per test, dropped afterwards.

WHAT THE CHECK CONSTRAINT IS FOR
`WhatTheConstraintCatches` exists because "the CHECK does not stop a lost update" is true
and is one half of a sentence. Quoted alone it reads as an argument for deleting the
constraint, which does catch a different mistake. Both mutations are pinned there.
"""

from __future__ import annotations

import os
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier

from lead_engine.providers.budget import (
    DEFAULT_SPLIT,
    DISCOVER,
    REFRESH,
    AllowanceAlreadySplit,
    BudgetExhausted,
    BudgetNotConfigured,
    BudgetState,
    SearchBudget,
    _divide,
    fingerprint,
)

try:
    import psycopg
    from psycopg_pool import ConnectionPool

    from lead_engine.db.migrate import apply_migrations
except ImportError:  # pragma: no cover - the pure suite runs without a database driver
    psycopg = None

DSN = os.environ.get("LEAD_ENGINE_TEST_DSN")
NEEDS_DB = unittest.skipIf(
    psycopg is None or not DSN,
    "needs psycopg and LEAD_ENGINE_TEST_DSN (see env.example)",
)

PROVIDER = "searchapi"

#: Two keys, as the operator's rotation produces them: the current one, and the replacement
#: that arrives when its allowance is gone.
KEY_A = fingerprint("searchapi-live-key-one")
KEY_B = fingerprint("searchapi-live-key-two")

#: The allowance actually in hand.
ALLOWANCE = 50


# --- the stub, for the parts that do not need Postgres --------------------------------------


class StubCursor:
    def __init__(self, rows):
        self._rows = list(rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def __iter__(self):
        return iter(self._rows)


class StubConnection:
    """Records every statement and hands back canned result sets, one per execute()."""

    def __init__(self, results=None):
        self.statements: list[tuple[str, tuple]] = []
        self.commits = 0
        self._results = list(results or [])

    def execute(self, query, params=None):
        self.statements.append((query, params))
        return StubCursor(self._results.pop(0) if self._results else [])

    def commit(self):
        self.commits += 1

    def factory(self):
        connection = self

        @contextmanager
        def connect():
            yield connection

        return connect


class BudgetContract(unittest.TestCase):
    """What can be pinned without a database: arguments, and the shape of the SQL."""

    def test_the_limit_check_lives_inside_the_update(self):
        # THE safety property. A read-then-write implementation passes every other test in
        # this file when run single-threaded, and loses the allocation in production. The
        # concurrency tests below are the real proof, but they need Postgres; this one runs
        # everywhere and fails loudly the moment the check moves out of the statement.
        stub = StubConnection(results=[[(1, ALLOWANCE)]])
        SearchBudget(stub.factory()).spend(PROVIDER)

        query, params = stub.statements[0]
        collapsed = " ".join(query.split()).lower()

        self.assertTrue(collapsed.startswith("update search_budget"), collapsed)
        self.assertIn("set used = used + %s", collapsed)
        self.assertRegex(collapsed, r"where .*used \+ %s <= limit_total")
        self.assertIn("returning", collapsed)
        # One statement, then a commit. Not a SELECT followed by an UPDATE.
        self.assertEqual(len(stub.statements), 1)
        self.assertEqual(params, (1, PROVIDER, DISCOVER, "", 1))

    def test_the_spend_is_keyed_on_all_three_columns(self):
        # Missing one of these does not fail loudly -- it charges the wrong row. Leaving out
        # key_fingerprint would spend a retired key's leftovers; leaving out purpose would
        # let discovery eat refresh's share.
        stub = StubConnection(results=[[(1, ALLOWANCE)]])
        SearchBudget(stub.factory()).spend(
            PROVIDER, purpose=REFRESH, key_fingerprint=KEY_A
        )

        query, params = stub.statements[0]
        collapsed = " ".join(query.split()).lower()
        for column in ("provider = %s", "purpose = %s", "key_fingerprint = %s"):
            self.assertIn(column, collapsed)
        self.assertEqual(params, (1, PROVIDER, REFRESH, KEY_A, 1))

    def test_no_select_precedes_the_spend(self):
        stub = StubConnection(results=[[(5, ALLOWANCE)]])
        SearchBudget(stub.factory()).spend(PROVIDER, 4)

        self.assertNotIn("select", stub.statements[0][0].lower())

    def test_the_spend_is_committed(self):
        # An uncommitted spend that a crash rolls back is a search that was paid for and
        # not recorded -- the exact leak the table exists to prevent.
        stub = StubConnection(results=[[(1, ALLOWANCE)]])
        SearchBudget(stub.factory()).spend(PROVIDER)
        self.assertEqual(stub.commits, 1)

    def test_ensure_never_resets_used(self):
        # `ensure` runs on every worker boot. ON CONFLICT DO UPDATE here would hand each
        # restart a fresh allowance.
        stub = StubConnection(results=[[(PROVIDER,)]])
        SearchBudget(stub.factory()).ensure(PROVIDER, ALLOWANCE)

        collapsed = " ".join(stub.statements[0][0].split()).lower()
        self.assertIn("on conflict (provider, purpose, key_fingerprint) do nothing", collapsed)
        self.assertNotIn("do update", collapsed)
        self.assertNotIn("used =", collapsed)

    def test_ensure_allowance_writes_every_row_in_one_statement(self):
        # A crash between two INSERTs would fund discovery and lose refresh, leaving a key
        # whose rows sum to less than the allowance it was cut from.
        stub = StubConnection(results=[[], [(DISCOVER, 35), (REFRESH, 15)]])
        SearchBudget(stub.factory()).ensure_allowance(
            PROVIDER, 50, key_fingerprint=KEY_A, split={DISCOVER: 0.7, REFRESH: 0.3}
        )

        insert, params = stub.statements[0]
        collapsed = " ".join(insert.split()).lower()
        self.assertTrue(collapsed.startswith("insert into search_budget"), collapsed)
        self.assertIn("values (%s, %s, %s, %s), (%s, %s, %s, %s)", collapsed)
        self.assertIn("on conflict (provider, purpose, key_fingerprint) do nothing", collapsed)
        # Both rows, both limits, in the one statement.
        self.assertEqual(
            list(params),
            [PROVIDER, DISCOVER, KEY_A, 35, PROVIDER, REFRESH, KEY_A, 15],
        )
        # Then a read-back, then one commit for the pair.
        self.assertEqual(len(stub.statements), 2)
        self.assertEqual(stub.commits, 1)

    def test_ensure_refuses_to_add_a_purpose_beside_an_existing_split(self):
        # The seeding path that doubles an allowance: ensure_allowance cuts 50 into 35/15,
        # then someone adds refresh=50 with a plain ensure().
        stub = StubConnection(results=[[], [(DISCOVER,), (REFRESH,)]])
        with self.assertRaises(AllowanceAlreadySplit):
            SearchBudget(stub.factory()).ensure(
                PROVIDER, ALLOWANCE, purpose=REFRESH, key_fingerprint=KEY_A
            )

    def test_the_refusal_shares_a_statement_with_the_insert(self):
        # A check-then-insert could be raced past. The NOT EXISTS has to be in the INSERT.
        stub = StubConnection(results=[[(PROVIDER,)]])
        SearchBudget(stub.factory()).ensure(PROVIDER, ALLOWANCE, key_fingerprint=KEY_A)

        collapsed = " ".join(stub.statements[0][0].split()).lower()
        self.assertTrue(collapsed.startswith("insert into search_budget"), collapsed)
        self.assertIn("where not exists", collapsed)
        self.assertIn("purpose <> %s", collapsed)

    def test_a_non_positive_spend_is_rejected(self):
        # Charging zero and reporting success hides a miscomputed page count.
        budget = SearchBudget(StubConnection().factory())
        for n in (0, -1, -100):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    budget.spend(PROVIDER, n)

    def test_a_negative_limit_is_rejected(self):
        budget = SearchBudget(StubConnection().factory())
        with self.assertRaises(ValueError):
            budget.ensure(PROVIDER, -1)

    def test_an_unknown_purpose_is_rejected_before_it_reaches_the_database(self):
        # 'refesh' would otherwise be a third budget that nothing sums and nobody spends.
        budget = SearchBudget(StubConnection().factory())
        for purpose in ("refesh", "Discover", "", "all"):
            with self.subTest(purpose=purpose):
                with self.assertRaises(ValueError):
                    budget.spend(PROVIDER, purpose=purpose)

    def test_purpose_and_key_cannot_be_passed_positionally(self):
        # They are both plain strings with defaults, side by side. A positional call site
        # that swapped them would write a legal-looking row that nothing else ever reads.
        budget = SearchBudget(StubConnection().factory())
        with self.assertRaises(TypeError):
            budget.spend(PROVIDER, 1, REFRESH)
        with self.assertRaises(TypeError):
            budget.ensure(PROVIDER, 10, REFRESH)

    def test_the_module_imports_no_database_driver(self):
        # The ledger talks to whatever connection it is handed. Importing psycopg here
        # would make the seam a lie and drag a driver into anything that reads a budget.
        import lead_engine.providers.budget as module

        source = module.__file__ or ""
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        self.assertNotRegex(text, r"(?m)^\s*(import|from)\s+psycopg")


class FingerprintsNeverCarryTheKey(unittest.TestCase):
    """The column exists so the ledger can tell keys apart without ever holding one."""

    #: Credential-shaped, and planted so that any leak into a message is visible.
    SENTINEL_KEY = "sk-live-searchapi-DO-NOT-LEAK-7c21"

    def test_it_is_twelve_lowercase_hex_characters(self):
        digest = fingerprint(self.SENTINEL_KEY)
        self.assertRegex(digest, r"^[0-9a-f]{12}$")

    def test_it_does_not_contain_the_key(self):
        self.assertNotIn(self.SENTINEL_KEY, fingerprint(self.SENTINEL_KEY))

    def test_it_is_stable_and_distinguishes_keys(self):
        # Stability is what makes a restart reuse the same row rather than mint a new
        # allowance; distinctness is what makes rotation work at all.
        self.assertEqual(fingerprint(self.SENTINEL_KEY), fingerprint(self.SENTINEL_KEY))
        self.assertNotEqual(fingerprint("key-one"), fingerprint("key-two"))

    def test_an_empty_key_is_a_configuration_bug(self):
        # Hashing "" would give every misconfigured deployment the same ledger row, and
        # each of them a full allowance on a key that cannot authenticate.
        with self.assertRaises(ValueError):
            fingerprint("")

    def test_a_raw_key_is_rejected_as_a_fingerprint(self):
        # The mistake this guards: passing the key where the digest belongs.
        budget = SearchBudget(StubConnection().factory())
        with self.assertRaises(ValueError):
            budget.spend(PROVIDER, key_fingerprint=self.SENTINEL_KEY)

    def test_rejecting_a_raw_key_does_not_echo_it(self):
        # The value is rejected precisely when it might BE the credential, so putting it in
        # the message would write the key into the traceback, the log aggregator, and the
        # ticket someone pastes it into.
        budget = SearchBudget(StubConnection().factory())
        with self.assertRaises(ValueError) as caught:
            budget.ensure(PROVIDER, 10, key_fingerprint=self.SENTINEL_KEY)
        self.assertNotIn(self.SENTINEL_KEY, str(caught.exception))

    def test_an_uppercase_digest_is_rejected(self):
        # Same key, different primary key -- which the ledger would read as a new key and
        # hand a second full allowance. That is rotation firing when nothing rotated.
        budget = SearchBudget(StubConnection().factory())
        with self.assertRaises(ValueError):
            budget.spend(PROVIDER, key_fingerprint=fingerprint("k").upper())


class SplitArithmetic(unittest.TestCase):
    """Cutting a total into whole searches. Pure, and worth pinning precisely.

    Every credit matters here: the allowance does not renew, so a part that rounds down
    strands money and a part that rounds up spends money that does not exist.
    """

    def test_the_default_gives_everything_to_discovery(self):
        # At fifty lifetime credits, re-checking swept ground is not worth paying for.
        self.assertEqual(_divide(ALLOWANCE, DEFAULT_SPLIT), {DISCOVER: 50, REFRESH: 0})

    def test_the_parts_always_sum_to_the_total(self):
        for total in range(0, 101):
            for split in (
                {DISCOVER: 0.7, REFRESH: 0.3},
                {DISCOVER: 1 / 3, REFRESH: 2 / 3},
                {DISCOVER: 1.0, REFRESH: 0.0},
                {DISCOVER: 0.5, REFRESH: 0.5},
            ):
                with self.subTest(total=total, split=split):
                    self.assertEqual(sum(_divide(total, split).values()), total)

    def test_binary_floating_point_does_not_lose_a_credit(self):
        # 0.7 * 50 is 34.99999999999999, so a plain int() would produce 34/15 and quietly
        # strand a search. This is the case that makes largest-remainder necessary.
        self.assertEqual(
            _divide(50, {DISCOVER: 0.7, REFRESH: 0.3}), {DISCOVER: 35, REFRESH: 15}
        )

    def test_a_split_that_does_not_sum_to_one_is_rejected(self):
        for split in (
            {DISCOVER: 0.5, REFRESH: 0.3},   # strands credits
            {DISCOVER: 1.0, REFRESH: 1.0},   # invents them -- the doubling bug, as a split
            {DISCOVER: 0.0, REFRESH: 0.0},
        ):
            with self.subTest(split=split):
                with self.assertRaises(ValueError):
                    _divide(ALLOWANCE, split)

    def test_the_spec_split_is_accepted_despite_float_arithmetic(self):
        # 0.7 + 0.3 == 0.9999999999999999, so an exact comparison against 1.0 would reject
        # the split the spec itself asks for.
        self.assertEqual(sum(_divide(100, {DISCOVER: 0.7, REFRESH: 0.3}).values()), 100)

    def test_an_unknown_purpose_in_the_split_is_rejected(self):
        with self.assertRaises(ValueError):
            _divide(ALLOWANCE, {DISCOVER: 0.5, "enrich": 0.5})

    def test_a_negative_share_is_rejected(self):
        with self.assertRaises(ValueError):
            _divide(ALLOWANCE, {DISCOVER: 1.5, REFRESH: -0.5})

    def test_ties_break_deterministically(self):
        # The same split must cut the same way regardless of dict ordering, or two workers
        # disagree about the ceiling.
        forward = _divide(51, {DISCOVER: 0.5, REFRESH: 0.5})
        backward = _divide(51, {REFRESH: 0.5, DISCOVER: 0.5})
        self.assertEqual(forward, backward)
        self.assertEqual(sum(forward.values()), 51)


class ExhaustionIsACleanStop(unittest.TestCase):
    """BudgetExhausted is a loop condition, not a failure. Pin what that means in code."""

    def test_it_is_not_a_provider_error(self):
        # If it inherited ProviderError it would carry an HTTP status, and the API layer
        # would dutifully turn "we decided to stop" into a 5xx.
        from lead_engine.providers.errors import ProviderError

        self.assertFalse(issubclass(BudgetExhausted, ProviderError))
        self.assertFalse(hasattr(BudgetExhausted("p", 50, 50, 1), "status"))

    def test_the_docstring_says_it_is_not_an_error(self):
        # The instruction to treat this as a clean stop has to survive contact with the
        # next person to read the class, so it lives in the docstring and is pinned here.
        doc = (BudgetExhausted.__doc__ or "").lower()
        self.assertIn("not an error", doc)
        self.assertIn("clean stop", doc)

    def test_it_reports_the_numbers_a_run_summary_needs(self):
        exhausted = BudgetExhausted(
            PROVIDER, limit_total=50, used=50, requested=3,
            purpose=REFRESH, key_fingerprint=KEY_A,
        )
        self.assertEqual(exhausted.provider, PROVIDER)
        self.assertEqual(exhausted.purpose, REFRESH)
        self.assertEqual(exhausted.key_fingerprint, KEY_A)
        self.assertEqual(exhausted.limit_total, 50)
        self.assertEqual(exhausted.used, 50)
        self.assertEqual(exhausted.requested, 3)
        self.assertEqual(exhausted.remaining, 0)

    def test_the_message_names_which_allowance_ran_out(self):
        # During a rotation several rows are in play, and "searchapi is exhausted" sends
        # the operator to look at the wrong one.
        message = str(
            BudgetExhausted(PROVIDER, 50, 50, 1, purpose=REFRESH, key_fingerprint=KEY_A)
        )
        self.assertIn(REFRESH, message)
        self.assertIn(KEY_A, message)

    def test_a_missing_row_is_not_reported_as_exhaustion(self):
        # The dangerous confusion: a worker whose `except BudgetExhausted` also swallowed a
        # missing row would exit zero having done nothing and look like a clean run.
        self.assertFalse(issubclass(BudgetNotConfigured, BudgetExhausted))
        self.assertFalse(issubclass(BudgetExhausted, BudgetNotConfigured))

    def test_an_over_allocation_is_not_reported_as_exhaustion_either(self):
        self.assertFalse(issubclass(AllowanceAlreadySplit, BudgetExhausted))

    def test_budget_state_arithmetic(self):
        state = BudgetState(PROVIDER, limit_total=50, used=47)
        self.assertEqual(state.remaining, 3)
        self.assertFalse(state.exhausted)
        self.assertTrue(BudgetState(PROVIDER, limit_total=50, used=50).exhausted)


# --- integration ----------------------------------------------------------------------------


class PostgresSchema:
    """A throwaway schema per test, migrated from empty. Mirrors tests/test_migrations.py.

    A mixin rather than a base TestCase on purpose. Several classes below need this fixture,
    and if the second inherited the first it would inherit its test methods too -- running
    them twice under pytest, which ignores the `load_tests` protocol that would otherwise
    filter them out.
    """

    #: Enough for one connection per worker in the concurrency tests. A pool smaller than
    #: the thread count would make threads queue for a connection instead of racing for
    #: the row, and the race is the entire point.
    POOL_SIZE = 24

    def setUp(self):
        self.schema = "test_" + uuid.uuid4().hex
        with psycopg.connect(DSN, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(f'CREATE SCHEMA "{self.schema}"')
        self.addCleanup(self._drop_schema)

        # A pool, not a connect-per-call factory, for two reasons. It is what production
        # uses -- `SearchBudget(pool.connection)` is the documented construction -- so the
        # tests exercise the real shape. And opening a connection to this database costs
        # ~300ms, which a per-operation factory pays on every single spend; the ledger
        # tests alone were spending a minute of wall clock on TCP handshakes.
        self.pool = ConnectionPool(
            DSN,
            min_size=1,
            max_size=self.POOL_SIZE,
            open=False,
            # Runs once per physical connection, so every checkout is already pinned to
            # this test's schema no matter which thread gets it.
            configure=self._pin_schema,
        )
        self.pool.open(wait=True, timeout=30)
        self.addCleanup(self.pool.close)

        with self.connect() as conn:
            apply_migrations(conn)

        self.budget = SearchBudget(self.pool.connection)

    def _pin_schema(self, conn) -> None:
        conn.execute(f'SET search_path = "{self.schema}", public')
        conn.commit()

    def _drop_schema(self):
        with psycopg.connect(DSN, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA "{self.schema}" CASCADE')

    def connect(self):
        """A connection pinned to this test's schema, from the pool."""
        return self.pool.connection()

    def used(self, purpose=DISCOVER, key_fingerprint=""):
        with self.connect() as conn:
            return conn.execute(
                "SELECT used FROM search_budget"
                " WHERE provider = %s AND purpose = %s AND key_fingerprint = %s",
                (PROVIDER, purpose, key_fingerprint),
            ).fetchone()[0]

    def rows(self):
        """Every ledger row, as {(purpose, fingerprint): (limit_total, used)}."""
        with self.connect() as conn:
            return {
                (r[0], r[1]): (r[2], r[3])
                for r in conn.execute(
                    "SELECT purpose, key_fingerprint, limit_total, used"
                    " FROM search_budget WHERE provider = %s",
                    (PROVIDER,),
                )
            }


@NEEDS_DB
class TheLedgerTable(PostgresSchema, unittest.TestCase):
    """The schema itself: what 0010 and 0011 build, and what the database refuses."""

    def test_the_ledger_is_keyed_on_three_columns(self):
        with self.connect() as conn:
            columns = {
                row[0]: row[1]
                for row in conn.execute(
                    "SELECT column_name, data_type FROM information_schema.columns"
                    " WHERE table_schema = %s AND table_name = 'search_budget'",
                    (self.schema,),
                )
            }
            primary_key = [
                row[0]
                for row in conn.execute(
                    "SELECT a.attname FROM pg_index i"
                    " JOIN pg_attribute a ON a.attrelid = i.indrelid"
                    "  AND a.attnum = ANY(i.indkey)"
                    " WHERE i.indrelid = (quote_ident(%s) || '.search_budget')::regclass"
                    "   AND i.indisprimary ORDER BY a.attnum",
                    (self.schema,),
                )
            ]

        self.assertEqual(
            set(columns),
            {"provider", "limit_total", "used", "updated_at", "purpose", "key_fingerprint"},
        )
        self.assertEqual(set(primary_key), {"provider", "purpose", "key_fingerprint"})

    def test_the_database_refuses_an_overspend_even_by_hand(self):
        # Defence in depth: the CHECK constraint, not the WHERE clause. This is what stops
        # a hand-written UPDATE at 2am from spending an allocation that cannot be refilled.
        self.budget.ensure(PROVIDER, 10)
        with self.assertRaises(psycopg.errors.CheckViolation):
            with self.connect() as conn:
                conn.execute(
                    "UPDATE search_budget SET used = 11 WHERE provider = %s", (PROVIDER,)
                )
                conn.commit()

    def test_the_database_refuses_an_unknown_purpose(self):
        with self.assertRaises(psycopg.errors.CheckViolation):
            with self.connect() as conn:
                conn.execute(
                    "INSERT INTO search_budget (provider, purpose, limit_total)"
                    " VALUES (%s, 'refesh', 10)",
                    (PROVIDER,),
                )
                conn.commit()

    def test_the_database_refuses_to_store_anything_that_is_not_a_digest(self):
        # The constraint that turns "we never store the key" from a convention into a
        # guarantee. A raw API key does not match, so it is rejected rather than stored.
        for bad in ("sk-live-searchapi-DO-NOT-LEAK-7c21", "ABCDEF012345", "abc", "a" * 13):
            with self.subTest(value=bad[:6]):
                with self.assertRaises(psycopg.errors.CheckViolation):
                    with self.connect() as conn:
                        conn.execute(
                            "INSERT INTO search_budget"
                            " (provider, purpose, key_fingerprint, limit_total)"
                            " VALUES (%s, 'discover', %s, 10)",
                            (PROVIDER, bad),
                        )
                        conn.commit()

    def test_the_empty_fingerprint_is_legal(self):
        # Providers where rotation is not modelled.
        self.assertTrue(self.budget.ensure("geoapify", 10))


@NEEDS_DB
class WhatTheConstraintCatches(PostgresSchema, unittest.TestCase):
    """The CHECK is not sufficient on its own, and it is not inert. Both, pinned.

    Two ways to break `spend()`, with two different outcomes. A note recording only the
    first would read as an argument for dropping a constraint that does catch the second;
    a note recording only the second would justify trusting a constraint that cannot see
    the first.
    """

    def test_a_sql_side_increment_that_forgot_its_guard_is_caught(self):
        # The second mutation: keep `used = used + n` in SQL, drop the predicate. Under
        # EvalPlanQual the loser re-reads the winner's committed row, computes past the
        # limit, and the CHECK fires. An ugly failure -- but a failure, not a silent
        # overspend. This is the case that makes the constraint worth keeping.
        self.budget.ensure(PROVIDER, 1)
        barrier = Barrier(2)

        def unguarded():
            barrier.wait(timeout=30)
            try:
                with self.connect() as conn:
                    conn.execute(
                        "UPDATE search_budget SET used = used + 1 WHERE provider = %s",
                        (PROVIDER,),
                    )
                    conn.commit()
                return "applied"
            except psycopg.errors.CheckViolation:
                return "refused"

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = [f.result(timeout=60) for f in [pool.submit(unguarded) for _ in range(2)]]

        self.assertEqual(outcomes.count("applied"), 1)
        self.assertEqual(outcomes.count("refused"), 1)
        self.assertEqual(self.used(), 1)

    def test_but_it_cannot_see_a_value_that_was_overwritten(self):
        # The first mutation, in miniature and without threads: two read-then-write spends
        # interleaved by hand. Both write a legal value; the second erases the first. The
        # column ends at 1 having been charged twice, and no constraint objects -- which is
        # exactly why `spend()` must not be written this way.
        self.budget.ensure(PROVIDER, 10)

        with self.connect() as first, self.connect() as second:
            seen_by_first = first.execute(
                "SELECT used FROM search_budget WHERE provider = %s", (PROVIDER,)
            ).fetchone()[0]
            seen_by_second = second.execute(
                "SELECT used FROM search_budget WHERE provider = %s", (PROVIDER,)
            ).fetchone()[0]

            first.execute(
                "UPDATE search_budget SET used = %s WHERE provider = %s",
                (seen_by_first + 1, PROVIDER),
            )
            first.commit()
            second.execute(
                "UPDATE search_budget SET used = %s WHERE provider = %s",
                (seen_by_second + 1, PROVIDER),
            )
            second.commit()  # no CheckViolation: 1 is a perfectly legal value

        self.assertEqual(self.used(), 1, "two charges, one recorded -- the lost update")


@NEEDS_DB
class AllowanceIsCut(PostgresSchema, unittest.TestCase):
    """The ceiling is a sum across rows, and no per-row constraint can hold it."""

    def test_the_split_sums_to_the_allowance(self):
        held = self.budget.ensure_allowance(
            PROVIDER, ALLOWANCE, key_fingerprint=KEY_A,
            split={DISCOVER: 0.7, REFRESH: 0.3},
        )

        self.assertEqual(held, {DISCOVER: 35, REFRESH: 15})
        with self.connect() as conn:
            total = conn.execute(
                "SELECT sum(limit_total) FROM search_budget"
                " WHERE provider = %s AND key_fingerprint = %s",
                (PROVIDER, KEY_A),
            ).fetchone()[0]
        self.assertEqual(total, ALLOWANCE)

    def test_the_default_split_funds_discovery_only(self):
        held = self.budget.ensure_allowance(PROVIDER, ALLOWANCE, key_fingerprint=KEY_A)

        self.assertEqual(held, {DISCOVER: 50, REFRESH: 0})
        # refresh exists and is exhausted, which is not the same as unconfigured: a worker
        # gets a clean stop rather than a configuration error.
        with self.assertRaises(BudgetExhausted):
            self.budget.spend(PROVIDER, purpose=REFRESH, key_fingerprint=KEY_A)

    def test_seeding_each_purpose_at_the_full_allowance_is_rejected(self):
        # The doubling bug, attempted through the API. ensure_allowance has no argument
        # that could express it, so the attempt has to come through ensure().
        self.budget.ensure_allowance(PROVIDER, ALLOWANCE, key_fingerprint=KEY_A)

        with self.assertRaises(AllowanceAlreadySplit):
            self.budget.ensure(PROVIDER, ALLOWANCE, purpose=REFRESH, key_fingerprint=KEY_A)

        with self.connect() as conn:
            total = conn.execute(
                "SELECT sum(limit_total) FROM search_budget"
                " WHERE provider = %s AND key_fingerprint = %s",
                (PROVIDER, KEY_A),
            ).fetchone()[0]
        self.assertEqual(total, ALLOWANCE)

    def test_the_schema_alone_would_have_allowed_it(self):
        # The honest record of what the database cannot enforce, so that nobody later
        # claims it does. Both rows below satisfy every CHECK while the key's ceilings sum
        # to twice its allowance. The guarantee lives in ensure_allowance(), not here.
        with self.connect() as conn:
            for purpose in (DISCOVER, REFRESH):
                conn.execute(
                    "INSERT INTO search_budget"
                    " (provider, purpose, key_fingerprint, limit_total)"
                    " VALUES (%s, %s, %s, %s)",
                    (PROVIDER, purpose, KEY_A, ALLOWANCE),
                )
            conn.commit()
            total = conn.execute(
                "SELECT sum(limit_total) FROM search_budget"
                " WHERE provider = %s AND key_fingerprint = %s",
                (PROVIDER, KEY_A),
            ).fetchone()[0]

        self.assertEqual(total, 2 * ALLOWANCE)

    def test_a_second_call_for_a_known_key_changes_nothing(self):
        # The restart case. ON CONFLICT DO NOTHING per row, so neither `used` nor a split
        # already in force can be re-cut.
        self.budget.ensure_allowance(
            PROVIDER, ALLOWANCE, key_fingerprint=KEY_A, split={DISCOVER: 0.7, REFRESH: 0.3}
        )
        self.budget.spend(PROVIDER, 10, key_fingerprint=KEY_A)

        again = self.budget.ensure_allowance(
            PROVIDER, ALLOWANCE, key_fingerprint=KEY_A, split={DISCOVER: 0.7, REFRESH: 0.3}
        )

        self.assertEqual(again, {DISCOVER: 35, REFRESH: 15})
        self.assertEqual(self.used(key_fingerprint=KEY_A), 10)
        self.assertEqual(self.budget.remaining(PROVIDER, key_fingerprint=KEY_A), 25)

    def test_a_second_call_cannot_re_cut_a_split_or_raise_a_ceiling(self):
        self.budget.ensure_allowance(PROVIDER, ALLOWANCE, key_fingerprint=KEY_A)

        self.budget.ensure_allowance(
            PROVIDER, 5000, key_fingerprint=KEY_A, split={DISCOVER: 0.5, REFRESH: 0.5}
        )

        self.assertEqual(
            self.rows(),
            {(DISCOVER, KEY_A): (50, 0), (REFRESH, KEY_A): (0, 0)},
        )


@NEEDS_DB
class PurposesAndKeysAreIndependent(PostgresSchema, unittest.TestCase):
    """Three columns, three dimensions that must not bleed into each other."""

    def test_exhausting_discovery_does_not_block_refresh(self):
        # The reason the split exists: refresh grows with every business already stored,
        # and must not be able to eat -- or be eaten by -- discovery's share.
        self.budget.ensure_allowance(
            PROVIDER, 10, key_fingerprint=KEY_A, split={DISCOVER: 0.7, REFRESH: 0.3}
        )
        self.budget.spend(PROVIDER, 7, key_fingerprint=KEY_A)

        with self.assertRaises(BudgetExhausted):
            self.budget.spend(PROVIDER, key_fingerprint=KEY_A)

        # Refresh is untouched and still spendable.
        self.assertEqual(
            self.budget.spend(PROVIDER, purpose=REFRESH, key_fingerprint=KEY_A), 2
        )
        self.assertEqual(self.budget.remaining(PROVIDER, key_fingerprint=KEY_A), 0)

    def test_a_new_key_gets_a_fresh_allowance_and_the_old_row_keeps_its_history(self):
        # Rotation. No reset command: an unseen fingerprint simply creates its own rows.
        # The retired key's row stays, because what its allowance bought is the only record
        # of what fifty searches actually yielded.
        self.budget.ensure_allowance(PROVIDER, ALLOWANCE, key_fingerprint=KEY_A)
        self.budget.spend(PROVIDER, ALLOWANCE, key_fingerprint=KEY_A)
        with self.assertRaises(BudgetExhausted):
            self.budget.spend(PROVIDER, key_fingerprint=KEY_A)

        self.budget.ensure_allowance(PROVIDER, ALLOWANCE, key_fingerprint=KEY_B)

        # Both rows, asserted together -- the point is that one is fresh and one is spent.
        self.assertEqual(self.budget.remaining(PROVIDER, key_fingerprint=KEY_B), ALLOWANCE)
        self.assertEqual(self.used(key_fingerprint=KEY_A), ALLOWANCE)
        self.assertEqual(self.used(key_fingerprint=KEY_B), 0)
        self.assertEqual(self.budget.remaining(PROVIDER, key_fingerprint=KEY_A), 0)

    def test_remaining_total_sums_purposes_for_one_key(self):
        # The question a free-text key could not answer.
        self.budget.ensure_allowance(
            PROVIDER, ALLOWANCE, key_fingerprint=KEY_A,
            split={DISCOVER: 0.7, REFRESH: 0.3},
        )
        self.budget.spend(PROVIDER, 5, key_fingerprint=KEY_A)
        self.budget.spend(PROVIDER, 5, purpose=REFRESH, key_fingerprint=KEY_A)

        self.assertEqual(self.budget.remaining_total(PROVIDER, key_fingerprint=KEY_A), 40)
        self.assertEqual(self.budget.remaining(PROVIDER, key_fingerprint=KEY_A), 30)
        self.assertEqual(
            self.budget.remaining(PROVIDER, purpose=REFRESH, key_fingerprint=KEY_A), 10
        )

    def test_remaining_total_does_not_sum_across_keys(self):
        # Credits stranded on a retired key are not credits anyone can spend, and adding
        # them to a total would overstate what the run can afford.
        self.budget.ensure_allowance(PROVIDER, ALLOWANCE, key_fingerprint=KEY_A)
        self.budget.ensure_allowance(PROVIDER, ALLOWANCE, key_fingerprint=KEY_B)
        self.budget.spend(PROVIDER, 50, key_fingerprint=KEY_A)

        self.assertEqual(self.budget.remaining_total(PROVIDER, key_fingerprint=KEY_A), 0)
        self.assertEqual(self.budget.remaining_total(PROVIDER, key_fingerprint=KEY_B), 50)

    def test_remaining_total_is_loud_for_a_key_that_was_never_configured(self):
        # sum() over no rows is NULL. Reporting that as 0 would turn a configuration bug
        # into a clean stop -- a run that exits zero having done nothing.
        self.budget.ensure_allowance(PROVIDER, ALLOWANCE, key_fingerprint=KEY_A)
        with self.assertRaises(BudgetNotConfigured):
            self.budget.remaining_total(PROVIDER, key_fingerprint=KEY_B)

    def test_spending_one_purpose_does_not_touch_the_other(self):
        self.budget.ensure_allowance(
            PROVIDER, 20, key_fingerprint=KEY_A, split={DISCOVER: 0.5, REFRESH: 0.5}
        )
        self.budget.spend(PROVIDER, 4, key_fingerprint=KEY_A)

        self.assertEqual(self.used(DISCOVER, KEY_A), 4)
        self.assertEqual(self.used(REFRESH, KEY_A), 0)

    def test_spending_one_key_does_not_touch_the_other(self):
        self.budget.ensure_allowance(PROVIDER, ALLOWANCE, key_fingerprint=KEY_A)
        self.budget.ensure_allowance(PROVIDER, ALLOWANCE, key_fingerprint=KEY_B)
        self.budget.spend(PROVIDER, 20, key_fingerprint=KEY_A)

        self.assertEqual(self.used(key_fingerprint=KEY_A), 20)
        self.assertEqual(self.used(key_fingerprint=KEY_B), 0)


@NEEDS_DB
class BudgetAgainstPostgres(PostgresSchema, unittest.TestCase):
    """Everything that needs a real ledger row but only one thread."""

    # --- ensure --------------------------------------------------------------------------

    def test_ensure_creates_once_and_is_idempotent(self):
        self.assertTrue(self.budget.ensure(PROVIDER, ALLOWANCE))
        self.assertFalse(self.budget.ensure(PROVIDER, ALLOWANCE))
        self.assertEqual(self.budget.remaining(PROVIDER), ALLOWANCE)

    def test_ensure_does_not_reset_a_partly_spent_ledger(self):
        # The restart case, and the whole reason this is a table rather than a counter.
        self.budget.ensure(PROVIDER, ALLOWANCE)
        self.budget.spend(PROVIDER, 40)

        self.assertFalse(self.budget.ensure(PROVIDER, ALLOWANCE))

        self.assertEqual(self.budget.remaining(PROVIDER), 10)
        self.assertEqual(self.used(), 40)

    def test_ensure_does_not_quietly_raise_the_ceiling(self):
        self.budget.ensure(PROVIDER, ALLOWANCE)
        self.budget.ensure(PROVIDER, 5000)
        self.assertEqual(self.budget.snapshot(PROVIDER).limit_total, ALLOWANCE)

    def test_ensure_still_works_for_a_provider_with_one_purpose(self):
        # Not every provider has a split. The refusal must only fire when one exists.
        self.assertTrue(self.budget.ensure("geoapify", 5, key_fingerprint=KEY_A))
        self.assertFalse(self.budget.ensure("geoapify", 5, key_fingerprint=KEY_A))

    def test_set_limit_moves_the_ceiling_deliberately(self):
        self.budget.ensure(PROVIDER, ALLOWANCE)
        self.budget.spend(PROVIDER, 10)

        state = self.budget.set_limit(PROVIDER, 200)

        self.assertEqual(state.limit_total, 200)
        self.assertEqual(state.used, 10)
        self.assertEqual(self.budget.remaining(PROVIDER), 190)

    def test_set_limit_cannot_drop_below_what_is_already_spent(self):
        self.budget.ensure(PROVIDER, ALLOWANCE)
        self.budget.spend(PROVIDER, 40)
        with self.assertRaises(psycopg.errors.CheckViolation):
            self.budget.set_limit(PROVIDER, 10)

    # --- spend and remaining -------------------------------------------------------------

    def test_spend_decrements_what_remains(self):
        self.budget.ensure(PROVIDER, ALLOWANCE)

        self.assertEqual(self.budget.spend(PROVIDER), 49)
        self.assertEqual(self.budget.spend(PROVIDER, 9), 40)
        self.assertEqual(self.budget.remaining(PROVIDER), 40)
        self.assertEqual(self.used(), 10)

    def test_remaining_is_exact_across_a_sequence_of_spends(self):
        self.budget.ensure(PROVIDER, 20)
        for expected in range(19, -1, -1):
            self.assertEqual(self.budget.spend(PROVIDER), expected)
        self.assertEqual(self.budget.remaining(PROVIDER), 0)

    def test_spending_the_last_credit_is_allowed(self):
        # Off-by-one in the safe direction is still a bug: it strands a paid-for search.
        self.budget.ensure(PROVIDER, 3)
        self.assertEqual(self.budget.spend(PROVIDER, 3), 0)
        self.assertEqual(self.used(), 3)

    def test_exhaustion_raises(self):
        self.budget.ensure(PROVIDER, 2)
        self.budget.spend(PROVIDER, 2)

        with self.assertRaises(BudgetExhausted) as caught:
            self.budget.spend(PROVIDER)

        self.assertEqual(caught.exception.provider, PROVIDER)
        self.assertEqual(caught.exception.used, 2)
        self.assertEqual(caught.exception.limit_total, 2)
        self.assertEqual(caught.exception.remaining, 0)

    def test_an_oversized_spend_charges_nothing(self):
        # All-or-nothing. Charging 2 of a requested 3 and returning success means the
        # caller makes three billed calls having paid for two.
        self.budget.ensure(PROVIDER, 10)
        self.budget.spend(PROVIDER, 8)

        with self.assertRaises(BudgetExhausted) as caught:
            self.budget.spend(PROVIDER, 3)

        self.assertEqual(caught.exception.requested, 3)
        self.assertEqual(self.used(), 8)
        self.assertEqual(self.budget.remaining(PROVIDER), 2)

    def test_the_ledger_survives_a_new_budget_object(self):
        # There is no instance state. A restarted worker sees the same numbers.
        self.budget.ensure(PROVIDER, ALLOWANCE)
        self.budget.spend(PROVIDER, 30)

        self.assertEqual(SearchBudget(self.connect).remaining(PROVIDER), 20)

    def test_providers_do_not_share_a_ledger(self):
        self.budget.ensure(PROVIDER, ALLOWANCE)
        self.budget.ensure("geoapify", 5)
        self.budget.spend(PROVIDER, 25)

        self.assertEqual(self.budget.remaining(PROVIDER), 25)
        self.assertEqual(self.budget.remaining("geoapify"), 5)

    def test_a_zero_limit_provider_can_never_spend(self):
        self.budget.ensure("disabled", 0)
        self.assertEqual(self.budget.remaining("disabled"), 0)
        with self.assertRaises(BudgetExhausted):
            self.budget.spend("disabled")

    # --- the unconfigured case -----------------------------------------------------------

    def test_an_unknown_provider_is_loud_not_a_clean_stop(self):
        for call in (
            lambda: self.budget.spend("never-configured"),
            lambda: self.budget.remaining("never-configured"),
            lambda: self.budget.snapshot("never-configured"),
            lambda: self.budget.set_limit("never-configured", 10),
            lambda: self.budget.remaining_total("never-configured"),
        ):
            with self.subTest(call=call):
                with self.assertRaises(BudgetNotConfigured):
                    call()

    def test_an_unknown_key_is_loud_too(self):
        # A rotated key before `ensure_allowance()` runs. Loud, because a worker that took
        # it for exhaustion would exit zero on every run after a rotation.
        self.budget.ensure_allowance(PROVIDER, ALLOWANCE, key_fingerprint=KEY_A)
        with self.assertRaises(BudgetNotConfigured):
            self.budget.spend(PROVIDER, key_fingerprint=KEY_B)

    def test_a_missing_row_does_not_look_like_exhaustion(self):
        # Written as the worker would write it. If BudgetNotConfigured were caught here the
        # run would exit zero having discovered nothing, and nobody investigates a green run.
        with self.assertRaises(BudgetNotConfigured):
            try:
                self.budget.spend("never-configured")
            except BudgetExhausted:  # pragma: no cover - the bug this test exists to catch
                self.fail("a missing ledger row was reported as a clean stop")


@NEEDS_DB
class SpendIsAtomic(PostgresSchema, unittest.TestCase):
    """Concurrent workers cannot overspend a limit. The reason this module exists.

    Real threads, real connections, one database row. These tests have been checked against
    a deliberately broken `spend()` -- SELECT the row, decide in Python, UPDATE with the new
    total -- and all of them go red. Under that mutation the 50-search allowance below was
    spent 1148 times.

    That number is worth staring at, because it explains what to assert. `used` in the
    database never exceeded 50 even while it happened: lost updates mean each worker
    overwrites the last, so the STORED value stays legal and the CHECK constraint never
    fires. Only the count of calls that returned successfully -- each one of which is a
    billed HTTP request the worker then makes -- reveals the overspend. So the assertions
    count outcomes, not just the column.

    That is not the whole story about the constraint, though: see
    `WhatTheConstraintCatches`, which pins the second mutation, where the arithmetic stays
    in SQL and the CHECK does fire.

    The barrier matters too. Without it the pool starts threads over several milliseconds
    and they queue up politely, the race never happens, and the test stays green while
    proving nothing. With it every worker is released into the same instant.
    """

    WORKERS = 24

    def spend_from_thread(self, barrier, n=1, purpose=DISCOVER, key_fingerprint=""):
        barrier.wait(timeout=30)
        try:
            self.budget.spend(PROVIDER, n, purpose=purpose, key_fingerprint=key_fingerprint)
            return "spent"
        except BudgetExhausted:
            return "exhausted"

    def race(self, workers, n=1):
        barrier = Barrier(workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self.spend_from_thread, barrier, n) for _ in range(workers)]
            return [future.result(timeout=60) for future in futures]

    def test_concurrent_spends_never_exceed_the_limit(self):
        # More workers than credits, all released at once. Exactly ten may win.
        self.budget.ensure(PROVIDER, 10)

        outcomes = self.race(self.WORKERS)

        self.assertEqual(outcomes.count("spent"), 10)
        self.assertEqual(outcomes.count("exhausted"), self.WORKERS - 10)
        self.assertEqual(self.used(), 10)
        self.assertEqual(self.budget.remaining(PROVIDER), 0)

    def test_no_credit_is_lost_when_every_worker_fits(self):
        # The opposite error: a spend that raises spuriously under contention strands paid
        # credits. Every worker must win here.
        self.budget.ensure(PROVIDER, self.WORKERS)

        outcomes = self.race(self.WORKERS)

        self.assertEqual(outcomes.count("spent"), self.WORKERS)
        self.assertEqual(self.used(), self.WORKERS)

    def test_multi_credit_spends_do_not_straddle_the_limit(self):
        # 8 workers x 3 credits against a limit of 10. Three can fit (9); the fourth must
        # not partially apply. Total used must be a multiple of 3 and never above 10.
        self.budget.ensure(PROVIDER, 10)

        outcomes = self.race(8, n=3)

        self.assertEqual(outcomes.count("spent"), 3)
        self.assertEqual(self.used(), 9)
        self.assertEqual(self.used() % 3, 0)

    def test_the_full_allowance_is_not_overspent(self):
        # The real numbers: a 50-search lifetime allowance, and a sweep that wants 195.
        self.budget.ensure(PROVIDER, ALLOWANCE)
        barrier = Barrier(self.WORKERS)

        def sweep():
            barrier.wait(timeout=30)
            spent = 0
            while True:
                try:
                    self.budget.spend(PROVIDER)
                except BudgetExhausted:
                    return spent
                spent += 1

        with ThreadPoolExecutor(max_workers=self.WORKERS) as pool:
            per_worker = [f.result(timeout=120) for f in
                          [pool.submit(sweep) for _ in range(self.WORKERS)]]

        # The load-bearing assertion. Every successful spend is a billed request the worker
        # goes on to make, so this sum is the real bill. Against a read-then-write spend it
        # came to 1148 while `used` below still read a perfectly legal 50.
        self.assertEqual(sum(per_worker), ALLOWANCE)
        self.assertEqual(self.used(), ALLOWANCE)
        self.assertEqual(self.budget.remaining(PROVIDER), 0)

    def test_concurrent_spends_on_two_purposes_do_not_interfere(self):
        # Both purposes are rows in one table under one provider and key. If the spend
        # statement were keyed on provider alone, these two sets of workers would race for
        # the same row and each would see the other's credits vanish.
        self.budget.ensure_allowance(
            PROVIDER, 20, key_fingerprint=KEY_A, split={DISCOVER: 0.5, REFRESH: 0.5}
        )
        barrier = Barrier(self.WORKERS)

        def worker(index):
            purpose = DISCOVER if index % 2 == 0 else REFRESH
            return purpose, self.spend_from_thread(
                barrier, purpose=purpose, key_fingerprint=KEY_A
            )

        with ThreadPoolExecutor(max_workers=self.WORKERS) as pool:
            results = [f.result(timeout=60) for f in
                       [pool.submit(worker, i) for i in range(self.WORKERS)]]

        # 12 workers per purpose against 10 credits each: exactly 10 win on each side.
        for purpose in (DISCOVER, REFRESH):
            spent = [r for p, r in results if p == purpose and r == "spent"]
            with self.subTest(purpose=purpose):
                self.assertEqual(len(spent), 10)
                self.assertEqual(self.used(purpose, KEY_A), 10)

    def test_concurrent_spends_on_two_keys_do_not_interfere(self):
        # The rotation window: the old key is still draining while the new one is in use.
        for key in (KEY_A, KEY_B):
            self.budget.ensure_allowance(PROVIDER, 10, key_fingerprint=key)
        barrier = Barrier(self.WORKERS)

        def worker(index):
            key = KEY_A if index % 2 == 0 else KEY_B
            return key, self.spend_from_thread(barrier, key_fingerprint=key)

        with ThreadPoolExecutor(max_workers=self.WORKERS) as pool:
            results = [f.result(timeout=60) for f in
                       [pool.submit(worker, i) for i in range(self.WORKERS)]]

        for key in (KEY_A, KEY_B):
            spent = [r for k, r in results if k == key and r == "spent"]
            with self.subTest(key=key):
                self.assertEqual(len(spent), 10)
                self.assertEqual(self.used(key_fingerprint=key), 10)


if __name__ == "__main__":
    unittest.main()
