"""AST gate preventing execution dependency on older experiment families."""

from __future__ import annotations

import ast
from pathlib import Path


FORBIDDEN_PREFIXES = (
    "project.run_scripts.ode_",
    "project.run_scripts.barrier_",
    "project.run_scripts.alphaedit_strength_neutral_barrier",
)


def scan(root: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                if name.startswith(FORBIDDEN_PREFIXES):
                    violations.append(f"{path}:{node.lineno}:{name}")
    return violations
