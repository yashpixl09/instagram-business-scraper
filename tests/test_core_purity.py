"""The pure core must stay pure.

`lead_engine/` is the part of this system that has no network, no database, and no file
format. That is what makes it fast to test exhaustively -- the niche suite runs thousands
of subtests because nothing it touches can block. Purity erodes one convenient import at a
time, so this test reads the source rather than trusting a convention.

The check parses each module and never imports it. Importing would prove the modules load
in an environment that happens to have `httpx` installed, which is the opposite of the
question being asked; parsing catches the forbidden import even when the dependency is
absent, and catches it inside functions and `TYPE_CHECKING` blocks too.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent / "lead_engine"

FORBIDDEN = {
    "psycopg",
    "asyncpg",
    "sqlalchemy",
    "fastapi",
    "pydantic",
    "httpx",
    "requests",
    "openpyxl",
    "socket",
    "urllib",
}

# `urllib` is forbidden as a root but `urllib.parse` is pure string manipulation -- it
# parses a URL, it does not fetch one -- and both `scoring.py` and `dedupe.py` rely on it.
# Allowing the exact submodule keeps `urllib.request` (a network client) forbidden, which
# is the thing this entry of FORBIDDEN exists to catch.
ALLOWED_SUBMODULES = {"urllib.parse"}

PURE = (
    "models.py",
    "dedupe.py",
    "scoring.py",
    "ranking.py",
    "copy.py",
    "niches.py",
    # The automation catalogue. Pure for the same reason the niche registry is: what fires an
    # opportunity is a rule, not a judgement, and a rule that cannot reach the network cannot
    # quietly start guessing.
    "automations.py",
)

#: Top-level modules that are deliberately NOT pure.
#:
#: `cli.py` and `config.py` sit at the top level because they are entry points -- what a
#: person types and what a process reads on boot -- and both necessarily import the outside
#: world: argparse and the service layer, pydantic-settings and the filesystem.
#:
#: Listed separately rather than added to `PURE`, which would be the tempting fix and the
#: wrong one: they would then fail the import check instead, and silencing that would mean
#: widening `FORBIDDEN` and disarming the guard for the six modules it exists to protect.
#:
#: Adding a name here is a deliberate act. A new top-level module belongs in `PURE` unless
#: someone decides otherwise, which is what the accounting test below enforces.
SURFACES = ("cli.py", "config.py")


def is_allowed(dotted: str) -> bool:
    return any(
        dotted == allowed or dotted.startswith(allowed + ".") for allowed in ALLOWED_SUBMODULES
    )


def imported_modules(tree: ast.AST) -> list[str]:
    """Every module path the source imports, as written.

    Relative imports are skipped: they are internal to `lead_engine` by construction and
    can never name a third-party package.
    """
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            module = node.module or ""
            # `from urllib import parse` imports the submodule, not the package, so judge
            # it by the submodule when that is what the statement actually pulls in. Under
            # `from urllib import request` no candidate is allowed and the package name is
            # judged instead, which is what makes that form fail.
            candidates = [f"{module}.{alias.name}" for alias in node.names]
            modules.extend(candidates if any(is_allowed(c) for c in candidates) else [module])
    return modules


def violations(source: str) -> list[str]:
    return [
        module
        for module in imported_modules(ast.parse(source))
        if module.split(".")[0] in FORBIDDEN and not is_allowed(module)
    ]


class CorePurityTests(unittest.TestCase):
    def test_pure_modules_import_no_io(self):
        for filename in PURE:
            path = CORE / filename
            with self.subTest(module=filename):
                self.assertTrue(path.is_file(), f"{path} is missing")
                self.assertEqual(violations(path.read_text(encoding="utf-8")), [])

    def test_every_top_level_module_is_accounted_for(self):
        # A new module in the core is pure until someone decides otherwise. Classifying it
        # has to be a deliberate act, so an unlisted name fails here rather than silently
        # going unchecked -- in either direction. A new pure module that nobody added to
        # PURE would never have its imports examined, and a new impure one dropped into
        # SURFACES without thought is how the boundary erodes.
        on_disk = {
            path.name
            for path in CORE.glob("*.py")
            if path.name != "__init__.py" and not path.name.startswith("_")
        }

        unclassified = on_disk - set(PURE) - set(SURFACES)
        self.assertEqual(
            unclassified,
            set(),
            f"{sorted(unclassified)} is neither in PURE nor SURFACES. Decide which it is: "
            f"PURE means its imports are policed, SURFACES means it is an entry point that "
            f"may reach the outside world.",
        )

    def test_the_two_categories_do_not_overlap(self):
        # A name in both would be checked and exempted at once, and the exemption would win.
        self.assertEqual(set(PURE) & set(SURFACES), set())

    def test_the_surfaces_really_do_depend_on_the_impure_layer(self):
        # The counterweight. Without it SURFACES becomes a place to file a module that could
        # have stayed pure, and the exemption list grows until it means nothing.
        #
        # The test is on TRANSITIVE dependence, not direct imports. `cli.py` imports only
        # argparse and first-party names; its impurity arrives through `.api.deps`, which
        # owns a connection pool. Checking direct imports would call it pure and demand it be
        # policed -- at which point the policing would fail, correctly, one level down.
        impure_packages = {"api", "db", "providers", "discovery", "export", "geo"}
        for filename in SURFACES:
            path = CORE / filename
            with self.subTest(module=filename):
                self.assertTrue(path.is_file(), f"{path} is listed but missing")
                tree = ast.parse(path.read_text(encoding="utf-8"))
                reached = {
                    (node.module or "").split(".")[0]
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.level
                }
                self.assertTrue(
                    reached & impure_packages or violations(path.read_text(encoding="utf-8")),
                    f"{filename} reaches nothing impure, directly or through a subpackage, "
                    f"so it does not need an exemption. Move it to PURE.",
                )

    def test_the_check_still_rejects_network_urllib(self):
        # Guarding the guard: the urllib.parse allowance is a hole in FORBIDDEN, and this
        # pins its exact size. If someone widens it to the whole package to silence a
        # failure, these cases fail instead.
        self.assertEqual(violations("from urllib.parse import urlparse"), [])
        self.assertEqual(violations("import urllib.parse"), [])
        self.assertEqual(violations("from urllib import parse"), [])
        self.assertEqual(violations("from urllib.request import urlopen"), ["urllib.request"])
        self.assertEqual(violations("import urllib.request"), ["urllib.request"])
        self.assertEqual(violations("from urllib import request"), ["urllib"])
        self.assertEqual(violations("import urllib"), ["urllib"])

    def test_the_check_finds_forbidden_imports_anywhere_in_a_module(self):
        deferred = """
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import psycopg


def load(dsn):
    import httpx

    from openpyxl import Workbook
    return httpx, Workbook
"""

        self.assertEqual(sorted(violations(deferred)), ["httpx", "openpyxl", "psycopg"])
        self.assertEqual(violations("import json, socket"), ["socket"])
        self.assertEqual(violations("from .models import Lead"), [])


if __name__ == "__main__":
    unittest.main()
