from __future__ import annotations

import ast
from pathlib import Path

DOMAIN_ROOT = Path(__file__).parents[2] / "src" / "lithops" / "domain"
FORBIDDEN_IMPORT_PREFIXES = (
    "fastapi",
    "httpx",
    "supabase",
    "google",
    "google_adk",
    "ceo_bench",
    "lithops.api",
    "lithops.application",
    "lithops.benchmark",
    "lithops.infrastructure",
)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_domain_has_no_framework_or_adapter_imports() -> None:
    violations: list[str] = []
    for path in DOMAIN_ROOT.rglob("*.py"):
        for module in _imports(path):
            if module.startswith(FORBIDDEN_IMPORT_PREFIXES):
                violations.append(f"{path.relative_to(DOMAIN_ROOT)} -> {module}")

    assert violations == []
