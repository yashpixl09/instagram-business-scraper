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

PURE = ("models.py", "dedupe.py", "scoring.py", "ranking.py", "copy.py", "niches.py")


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

    def test_every_pure_module_is_accounted_for(self):
        # A new module in the core is pure until someone decides otherwise. Listing it in
        # PURE has to be a deliberate act, so a missing name fails here rather than
        # silently going unchecked.
        on_disk = {
            path.name
            for path in CORE.glob("*.py")
            if path.name != "__init__.py" and not path.name.startswith("_")
        }

        self.assertEqual(on_disk - set(PURE), set())

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
