"""Sealed model-token topology cases for BGODE-R2 Stage C."""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from project.run_scripts.ode_edit_motivation.contracts import canonical_json, sha256_bytes

from .errors import PrefixEventBoundary


TOPOLOGY_MANIFEST = Path(__file__).with_name("locks") / "bgode-r2-stage-c-topology-manifest.json"
TOPOLOGY_NAMES = (
    "both-multi-unequal-nonprefix",
    "source-prefix-target",
    "target-prefix-source",
)


@dataclass(frozen=True, slots=True)
class StageCTopologyCase:
    model_alias: str
    topology: str
    target_token_ids: tuple[int, ...]
    source_token_ids: tuple[int, ...]
    case_identity: str

    def receipt(self) -> dict[str, Any]:
        return {
            "model_alias": self.model_alias,
            "topology": self.topology,
            "target_token_ids": list(self.target_token_ids),
            "source_token_ids": list(self.source_token_ids),
            "case_identity": self.case_identity,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relation(target: tuple[int, ...], source: tuple[int, ...]) -> str:
    if len(source) < len(target) and target[: len(source)] == source:
        return "source-prefix-target"
    if len(target) < len(source) and source[: len(target)] == target:
        return "target-prefix-source"
    return "both-multi-unequal-nonprefix" if len(target) > 1 and len(source) > 1 and len(target) != len(source) else "other"


def load_stage_c_manifest(path: Path = TOPOLOGY_MANIFEST) -> tuple[Mapping[str, Any], str]:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise PrefixEventBoundary("Stage-C topology manifest must be regular mode0600")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "ode-edit-bgode-r2-stage-c-topology-manifest/v1":
        raise PrefixEventBoundary("Stage-C topology manifest schema differs")
    body = {key: value for key, value in payload.items() if key != "identity"}
    if sha256_bytes(canonical_json(body).encode("utf-8")) != payload.get("identity"):
        raise PrefixEventBoundary("Stage-C topology manifest identity differs")
    return MappingProxyType(payload), _sha256(path)


def topology_case(model_alias: str, topology: str) -> StageCTopologyCase:
    if topology not in TOPOLOGY_NAMES:
        raise PrefixEventBoundary("unknown Stage-C topology")
    manifest, _ = load_stage_c_manifest()
    rows = manifest.get("cases")
    if not isinstance(rows, list):
        raise PrefixEventBoundary("Stage-C topology cases are absent")
    matches = [row for row in rows if row.get("model_alias") == model_alias and row.get("topology") == topology]
    if len(matches) != 1:
        raise PrefixEventBoundary("Stage-C topology case cardinality differs")
    row = matches[0]
    target = tuple(int(value) for value in row["target_token_ids"])
    source = tuple(int(value) for value in row["source_token_ids"])
    if not target or not source or target == source or _relation(target, source) != topology:
        raise PrefixEventBoundary("Stage-C topology relation differs")
    body = {key: value for key, value in row.items() if key != "case_identity"}
    if sha256_bytes(canonical_json(body).encode("utf-8")) != row.get("case_identity"):
        raise PrefixEventBoundary("Stage-C case identity differs")
    return StageCTopologyCase(model_alias, topology, target, source, row["case_identity"])


def cell_case(cell: int) -> StageCTopologyCase:
    if isinstance(cell, bool) or cell < 0 or cell >= 6:
        raise PrefixEventBoundary("Stage-C cell is outside 0..5")
    alias = "llama3-8b-inst" if cell < 3 else "qwen2.5-7b-inst"
    return topology_case(alias, TOPOLOGY_NAMES[cell % 3])


__all__ = ["StageCTopologyCase", "TOPOLOGY_MANIFEST", "TOPOLOGY_NAMES", "cell_case", "load_stage_c_manifest", "topology_case"]
