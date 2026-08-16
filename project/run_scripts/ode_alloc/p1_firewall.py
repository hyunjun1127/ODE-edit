"""AST firewalls separating P1 optimization from terminal-only evaluation."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

from .contracts import ODEAllocContractError


_INNER_FORBIDDEN = (
    "session03",
    "session_03",
    "paraphrase_prompts",
    "neighborhood_prompts",
    "generation_prompts",
    "attribute_prompts",
)


def _tree(path: str | Path) -> tuple[Path, ast.AST]:
    source = Path(path).resolve(strict=True)
    return source, ast.parse(source.read_text(encoding="utf-8"), filename=str(source))


def assert_p1_inner_firewall(paths: Iterable[str | Path]) -> None:
    violations: list[str] = []
    for path in paths:
        source, tree = _tree(path)
        for node in ast.walk(tree):
            values: tuple[str, ...] = ()
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                values = (node.value.casefold(),)
            elif isinstance(node, ast.Name):
                values = (node.id.casefold(),)
            elif isinstance(node, ast.Attribute):
                values = (node.attr.casefold(),)
            elif isinstance(node, ast.Import):
                values = tuple(alias.name.casefold() for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                values = ((node.module or "").casefold(),) + tuple(
                    alias.name.casefold() for alias in node.names
                )
            if any(token in value for token in _INNER_FORBIDDEN for value in values):
                violations.append(f"{source.name}:{getattr(node, 'lineno', 0)}")
    if violations:
        raise ODEAllocContractError(
            "P1 inner runtime crosses the held-out/session firewall: "
            + ",".join(violations)
        )


def assert_p1_evaluator_firewall(path: str | Path) -> None:
    source, tree = _tree(path)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.casefold() == "generate":
            violations.append(f"model-generate:{node.lineno}")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names = tuple(alias.name.casefold() for alias in node.names)
            else:
                names = ((node.module or "").casefold(),)
            if any("easyeditor" in value or "session03" in value or "session_03" in value for value in names):
                violations.append(f"foreign-evaluator:{node.lineno}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.casefold()
            if "session03" in lowered or "session_03" in lowered:
                violations.append(f"foreign-string:{node.lineno}")
    if violations:
        raise ODEAllocContractError(
            "P1 evaluator crosses the independent terminal firewall: "
            + ",".join(violations)
        )


def assert_no_alias_specific_scientific_branch(path: str | Path) -> None:
    source, tree = _tree(path)
    aliases = {"llama3-8b-inst", "qwen2.5-7b-inst"}
    violations: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.If, ast.IfExp, ast.Match)):
            continue
        constants = {
            child.value
            for child in ast.walk(node)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        }
        if constants.intersection(aliases):
            violations.append(getattr(node, "lineno", 0))
    if violations:
        raise ODEAllocContractError(
            "P1 runtime contains an alias-specific scientific branch: "
            + ",".join(str(value) for value in violations)
        )


def assert_projector_is_only_adaptive_arm_branch(path: str | Path) -> None:
    """Allow Generic-vs-ODE branching only in the projector factory."""

    source, tree = _tree(path)
    violations: list[str] = []
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        for node in ast.walk(function):
            if not isinstance(node, (ast.If, ast.IfExp, ast.Match)):
                continue
            adaptive_names = {
                child.attr
                for child in ast.walk(node)
                if isinstance(child, ast.Attribute)
                and child.attr in {"GENERIC_ADAPTIVE", "ODE_ALLOC"}
            }
            if adaptive_names and function.name != "_projector_for_arm":
                violations.append(f"{function.name}:{getattr(node, 'lineno', 0)}")
    if violations:
        raise ODEAllocContractError(
            "P1 adaptive arms branch outside the projector hook: "
            + ",".join(violations)
        )
