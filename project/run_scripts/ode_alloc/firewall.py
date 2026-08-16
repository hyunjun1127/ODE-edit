"""Static and schema firewalls for coefficient optimization inputs."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping

from .contracts import ODEAllocContractError


_ROW_KEYS = frozenset({"case_id", "requested_rewrite"})
_REWRITE_KEYS = frozenset(
    {"prompt", "relation_id", "subject", "target_new", "target_true"}
)
_FORBIDDEN_AST_TOKENS = (
    "session03",
    "paraphrase_prompts",
    "neighborhood_prompts",
    "generation_prompts",
    "attribute_prompts",
    "locality",
)
_FORBIDDEN_RUNTIME_IMPORTS = (
    "transformers",
    "datasets",
    "easyeditor",
)
_HELDOUT_ROW_KEYS = frozenset(
    {
        "paraphrase_prompts",
        "neighborhood_prompts",
        "generation_prompts",
        "attribute_prompts",
    }
)


def assert_inner_payload_schema(payload: Mapping[str, Any]) -> None:
    if set(payload) != _ROW_KEYS:
        raise ODEAllocContractError("inner payload exposes non-request row fields")
    rewrite = payload.get("requested_rewrite")
    if not isinstance(rewrite, Mapping) or set(rewrite) != _REWRITE_KEYS:
        raise ODEAllocContractError("inner rewrite payload schema differs")
    for target_name in ("target_new", "target_true"):
        target = rewrite.get(target_name)
        if not isinstance(target, Mapping) or "str" not in target:
            raise ODEAllocContractError("inner rewrite target schema differs")


def assert_dry_launcher_ast(path: str | Path) -> None:
    """Fail closed if a prep launcher can import runtime/evaluation surfaces."""

    source_path = Path(path).resolve(strict=True)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.casefold() for alias in node.names]
            if any(
                token in name
                for token in (*_FORBIDDEN_RUNTIME_IMPORTS, *_FORBIDDEN_AST_TOKENS)
                for name in names
            ):
                violations.append(f"runtime-import:{node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").casefold()
            imported = [alias.name.casefold() for alias in node.names]
            if any(
                token in value
                for token in (*_FORBIDDEN_RUNTIME_IMPORTS, *_FORBIDDEN_AST_TOKENS)
                for value in (module, *imported)
            ):
                violations.append(f"runtime-import:{node.lineno}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.casefold()
            if any(token in lowered for token in _FORBIDDEN_AST_TOKENS):
                violations.append(f"forbidden-string:{node.lineno}")
        elif isinstance(node, ast.Attribute):
            lowered = node.attr.casefold()
            if any(token in lowered for token in _FORBIDDEN_AST_TOKENS):
                violations.append(f"forbidden-attribute:{node.lineno}")
        elif isinstance(node, ast.Name):
            lowered = node.id.casefold()
            if any(token in lowered for token in _FORBIDDEN_AST_TOKENS):
                violations.append(f"forbidden-name:{node.lineno}")
    if violations:
        raise ODEAllocContractError(
            "dry launcher crosses the held-out/runtime firewall: " + ",".join(violations)
        )


def assert_p0_runtime_firewall_ast(paths: list[str | Path]) -> None:
    """P0 may import model runtime, but never evaluation row keys or foreign sessions."""

    violations: list[str] = []
    for value in paths:
        source_path = Path(value).resolve(strict=True)
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lowered = node.value.casefold()
                if lowered in _HELDOUT_ROW_KEYS or "session03" in lowered or "session_03" in lowered:
                    violations.append(f"forbidden-runtime-string:{source_path.name}:{node.lineno}")
            elif isinstance(node, ast.Import):
                if any(alias.name.casefold() == "datasets" for alias in node.names):
                    violations.append(f"dataset-import:{source_path.name}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").casefold().split(".")[0] == "datasets":
                    violations.append(f"dataset-import:{source_path.name}:{node.lineno}")
    if violations:
        raise ODEAllocContractError(
            "P0 runtime crosses the held-out/session firewall: " + ",".join(violations)
        )
