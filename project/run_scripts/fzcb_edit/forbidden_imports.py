"""Fail-close AST boundary against superseded scientific controllers."""

from __future__ import annotations

import ast
from pathlib import Path


FORBIDDEN = (
    "project.run_scripts.barrier_usefulness",
    "project.run_scripts.fzcb_completion_value",
    "project.run_scripts.ode_",
    "project.run_scripts.barrier_guided_ode",
)


def scan(root: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "tests" in path.parts or path.name == "forbidden_imports.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN):
                    violations.append(f"{path}:{node.lineno}:{name}")
    return violations
