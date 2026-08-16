"""Schema and source firewalls for controller/evaluator separation."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import BATCH_SIZE, ODEBFContractError, canonical_hash


CONTROLLER_ALLOWED_FIELDS = {
    "case_id",
    "request_sha256",
    "prompt",
    "subject",
    "relation_id",
    "target_new",
    "target_true",
    "authorized_rewrite_prefixes",
}
FORBIDDEN_FIELD_FRAGMENTS = (
    "paraphrase",
    "neighborhood",
    "locality",
    "generation",
    "heldout",
    "held_out",
    "session03",
)
FORBIDDEN_IMPORT_FRAGMENTS = (
    "session03",
    "ode_alloc.p1_evaluator",
    "ode_alloc.p1_runtime",
    "knowledge-revision",
    "knowledge_revision",
)


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, Mapping):
            for key, child in item.items():
                keys.append(str(key).lower())
                stack.append(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            stack.extend(item)
    return keys


def validate_controller_batch(requests: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    if len(requests) != BATCH_SIZE:
        raise ODEBFContractError("controller requires one joint B10 request batch")
    identities: list[str] = []
    case_ids: list[int] = []
    for request in requests:
        keys = _walk_keys(request)
        if any(fragment in key for key in keys for fragment in FORBIDDEN_FIELD_FRAGMENTS):
            raise ODEBFContractError("held-out/evaluation field entered the controller")
        if not set(request).issubset(CONTROLLER_ALLOWED_FIELDS):
            raise ODEBFContractError("controller request contains an unapproved field")
        required = {
            "case_id",
            "request_sha256",
            "prompt",
            "subject",
            "target_new",
            "target_true",
            "authorized_rewrite_prefixes",
        }
        if not required.issubset(request):
            raise ODEBFContractError("controller request schema is incomplete")
        identity = str(request["request_sha256"])
        if len(identity) != 64:
            raise ODEBFContractError("controller request identity is not SHA-256")
        identities.append(identity)
        case_ids.append(int(request["case_id"]))
    if len(set(identities)) != BATCH_SIZE or len(set(case_ids)) != BATCH_SIZE:
        raise ODEBFContractError("controller batch contains duplicate requests")
    return tuple(identities)


def evaluator_request_hash(request: Mapping[str, Any]) -> str:
    allowed = {
        "case_id",
        "prompt",
        "relation_id",
        "subject",
        "target_new",
        "target_true",
        "request_sha256",
    }
    if set(request) != allowed:
        raise ODEBFContractError("canonical evaluator request schema differs")
    return canonical_hash(
        {
            "case_id": request["case_id"],
            "prompt": request["prompt"],
            "relation_id": request["relation_id"],
            "subject": request["subject"],
            "target_new": request["target_new"],
            "target_old": request["target_true"],
        }
    )


def assert_ast_firewall(paths: Sequence[Path]) -> None:
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.lower() for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").lower()]
            else:
                names = []
            if any(fragment in name for name in names for fragment in FORBIDDEN_IMPORT_FRAGMENTS):
                raise ODEBFContractError("forbidden foreign/session import found")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "generate":
                    raise ODEBFContractError("model.generate is forbidden in ODE-BF")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lowered = node.value.lower()
                if (
                    ("/" in lowered or "\\" in lowered or "." in lowered)
                    and ("session03" in lowered or "knowledge-revision" in lowered)
                ):
                    raise ODEBFContractError("forbidden foreign/session string found")


def assert_no_alias_specific_controller_branch(paths: Sequence[Path]) -> None:
    aliases = ("llama3-8b-inst", "qwen2.5-7b-inst")
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                source = ast.unparse(node.test).lower()
                if any(alias in source for alias in aliases):
                    raise ODEBFContractError("controller contains an alias-specific branch")
