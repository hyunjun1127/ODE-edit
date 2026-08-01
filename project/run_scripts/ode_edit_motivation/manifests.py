"""Pinned MV-0 inputs and case-id-only CounterFact selection.

This module deliberately contains the complete byte identities instead of
discovering artifacts at runtime.  A missing or changed input is an error; no
caller in this module downloads, repairs, or recomputes it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    ContractError,
    EditRequest,
    ExpectedFileIdentity,
    ProvenanceManifest,
    canonical_json,
    preflight_pinned_files,
    sha256_bytes,
)


COUNTERFACT_RELATIVE_PATH = "data/counterfact/counterfact.json"
COUNTERFACT_ROW_COUNT = 21_919
DEFAULT_SELECTION_SEED = "ode-edit-motivation-counterfact-v1"
DEFAULT_SPLIT_COUNTS = (20, 60, 20)

EASYEDIT_SOURCE_PATHS: tuple[str, ...] = (
    "easyeditor/models/memit/memit_main.py",
    "easyeditor/models/memit/compute_z.py",
    "easyeditor/models/memit/compute_ks.py",
    "easyeditor/models/memit/memit_hparams.py",
    "easyeditor/models/rome/layer_stats.py",
    "easyeditor/models/rome/repr_tools.py",
    "easyeditor/util/nethook.py",
    "easyeditor/util/generate.py",
    "easyeditor/util/logit_lens.py",
    "easyeditor/util/globals.py",
    "easyeditor/util/runningstats.py",
    "easyeditor/util/hparams.py",
    "easyeditor/models/rome/tok_dataset.py",
)

MEMIT_HPARAM_PATHS: tuple[str, ...] = (
    "hparams/MEMIT/llama3-8b.yaml",
    "hparams/MEMIT/qwen2.5-7b.yaml",
)

# AlphaEdit is consumed as a pinned, read-only mathematical reference by the
# ODE-side isolated-first-edit adapter.  These files are deliberately not part
# of ``EASYEDIT_SOURCE_PATHS``: the verified MEMIT import bridge must keep its
# executable import closure unchanged.
ALPHAEDIT_REFERENCE_SOURCE_PATHS: tuple[str, ...] = (
    "easyeditor/models/alphaedit/AlphaEdit_main.py",
    "easyeditor/models/alphaedit/compute_ks.py",
    "easyeditor/models/alphaedit/compute_z.py",
    "easyeditor/models/alphaedit/AlphaEdit_hparams.py",
)

ALPHAEDIT_HPARAM_BY_MODEL: Mapping[str, str] = MappingProxyType(
    {
        "llama3-8b-inst": "hparams/AlphaEdit/llama3-8b.yaml",
        "qwen2.5-7b-inst": "hparams/AlphaEdit/qwen2.5-7b.yaml",
    }
)

EASYEDIT_RUNTIME_PATHS: tuple[str, ...] = (
    "pyproject.toml",
    "uv.lock",
    ".python-version",
)


@dataclass(frozen=True, slots=True)
class FixedModelSpec:
    """Exact model and local MEMIT artifact identity."""

    alias: str
    repository_id: str
    revision: str
    hparams_path: str
    covariance_paths: tuple[str, ...]
    projector_path: str
    layers: tuple[int, ...] = (4, 5, 6, 7, 8)

    @property
    def snapshot_name(self) -> str:
        return f"{self.repository_id}@{self.revision}"

    def covariance_path_for_layer(self, layer: int) -> str:
        try:
            index = self.layers.index(int(layer))
        except ValueError as exc:
            raise ContractError(
                f"layer {layer} is outside the fixed {self.alias} MEMIT layer set"
            ) from exc
        return self.covariance_paths[index]


_LLAMA_STATS = tuple(
    "examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats/"
    f"model.layers.{layer}.mlp.down_proj_float32_mom2_100000.npz"
    for layer in (4, 5, 6, 7, 8)
)
_QWEN_STATS = tuple(
    "examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/"
    f"model.layers.{layer}.mlp.down_proj_float32_mom2_100000.npz"
    for layer in (4, 5, 6, 7, 8)
)

MODEL_SPECS: Mapping[str, FixedModelSpec] = MappingProxyType(
    {
        "llama3-8b-inst": FixedModelSpec(
            alias="llama3-8b-inst",
            repository_id="meta-llama/Meta-Llama-3-8B-Instruct",
            revision="8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
            hparams_path="hparams/MEMIT/llama3-8b.yaml",
            covariance_paths=_LLAMA_STATS,
            projector_path=(
                "examples/null_space_project_Meta-Llama-3-8B-Instruct.pt"
            ),
        ),
        "qwen2.5-7b-inst": FixedModelSpec(
            alias="qwen2.5-7b-inst",
            repository_id="Qwen/Qwen2.5-7B-Instruct",
            revision="a09a35458c702b33eeacc393d103063234e8bc28",
            hparams_path="hparams/MEMIT/qwen2.5-7b.yaml",
            covariance_paths=_QWEN_STATS,
            projector_path="examples/null_space_project_Qwen2.5-7B-Instruct.pt",
        ),
    }
)


def _identity(sha256: str, size: int) -> ExpectedFileIdentity:
    return ExpectedFileIdentity(sha256=sha256, size=size)


# These are the fixed local bytes audited on 2026-07-30.  Projector entries
# are preflight-only: MV-0 never deserializes them.
FIXED_FILE_IDENTITIES: Mapping[str, ExpectedFileIdentity] = MappingProxyType(
    {
        "easyeditor/models/memit/memit_main.py": _identity(
            "32f27516d30d196ceb2f99bc02359b66a0e886246953bdedc232f64fe7ee6d0a",
            10_600,
        ),
        "easyeditor/models/memit/compute_z.py": _identity(
            "6e43c1f03bc03c87dcff70111dff1a93f4e00ea31b4bfeb5e8eab4a516ea7ae4",
            12_192,
        ),
        "easyeditor/models/memit/compute_ks.py": _identity(
            "0b039be47c548f046b71cad4efce63414f53edbaa657bde868a20cdd35a2f5c7",
            1_540,
        ),
        "easyeditor/models/memit/memit_hparams.py": _identity(
            "223dc4e663a049214c0528a02eff8c066413fbee9bce98735a2ce8d1c6585f41",
            1_578,
        ),
        "easyeditor/models/alphaedit/AlphaEdit_main.py": _identity(
            "a3a459abc4e05f4b3ccaf424a8950245c491686e943fdcaa52b296343e7458b1",
            14_586,
        ),
        "easyeditor/models/alphaedit/compute_ks.py": _identity(
            "6c5b53d4a1fae02d7b303fe67acc724b31261e2a82b915de4c3862661023bdba",
            1_552,
        ),
        "easyeditor/models/alphaedit/compute_z.py": _identity(
            "e12140c66b467759f3fb0061ec844f79147221e0649f0d0a5f9657225ff12c95",
            10_210,
        ),
        "easyeditor/models/alphaedit/AlphaEdit_hparams.py": _identity(
            "703776fdfd095d0834483705311b7e103b898c9cf18c66388d5b4b06e8aeeac2",
            1_674,
        ),
        "easyeditor/models/rome/layer_stats.py": _identity(
            "10c5f6204697a090671d3810f7fd60e781c44b7537e47a9625c07e542ff87cc4",
            7_157,
        ),
        "easyeditor/models/rome/repr_tools.py": _identity(
            "a83de9ef7a7d65403ccde83a4abf6bb9407d29da19d96cc3508da362c03b23a1",
            7_011,
        ),
        "easyeditor/util/nethook.py": _identity(
            "2026b36c3a17cd6da0c6e80c79d32d5d80f835e7eb7904d298c0abaa098ecf66",
            16_739,
        ),
        "easyeditor/util/generate.py": _identity(
            "13d28184bd215ea7d51bef091fcd3624271d93f4c3df32fdbf2e97cc15581b45",
            7_296,
        ),
        "easyeditor/util/logit_lens.py": _identity(
            "7167211cb97f00544e14dc3cfd3f212f6486818dd8e6931e136226d710a3acc2",
            2_890,
        ),
        "easyeditor/util/globals.py": _identity(
            "fdb1cbed3ff82eb87b26bced5bb4f2f43e8d8cb76c81b1cd5d83a3b3cd830ed1",
            1_291,
        ),
        "easyeditor/util/runningstats.py": _identity(
            "cae826b6471b6b5a4dd1f5da3adbd9cf697b01ae80585d59e727ca64f4ec47ae",
            65_306,
        ),
        "easyeditor/util/hparams.py": _identity(
            "9ee7a423fd70a9eaacb4c2ef1aaf2268ba87e28656fd7e3c562a7c9505d25b72",
            1_319,
        ),
        "easyeditor/models/rome/tok_dataset.py": _identity(
            "3dd4815dd6d15e0727939f38e8a8f8ddfaea56f10b700249334982c56e4883b4",
            3_103,
        ),
        "pyproject.toml": _identity(
            "375451c522866ac335043f33e9f6d169ab1e17648d56a5ed7812d6a26fbf956e",
            3_080,
        ),
        "uv.lock": _identity(
            "628fa22b1effa8f4c8135e06e75b8a956fba2f684edd11ee0b553d61a54f8dff",
            538_466,
        ),
        ".python-version": _identity(
            "7b55f8e67b5623c4bef3fa691288da9437d79d3aba156de48d481db32ac7d16d",
            5,
        ),
        "hparams/MEMIT/llama3-8b.yaml": _identity(
            "2b81838b49b1a5e0e4ef41f229216fda473200093fcbed6bd040f07a70d43818",
            631,
        ),
        "hparams/MEMIT/qwen2.5-7b.yaml": _identity(
            "fccad05cf749c710ba0ce58ae24203f0bb90d9bf488966a5184dacd489a0311b",
            639,
        ),
        "hparams/AlphaEdit/llama3-8b.yaml": _identity(
            "d403e1875e62096b089be5343d33896510e454b0cdd2b256618ab53de6609ef8",
            1_134,
        ),
        "hparams/AlphaEdit/qwen2.5-7b.yaml": _identity(
            "82d04976c4ab65e67c537ac3bd1b04d42c8f7527e2a749bdefcce63e43b995c3",
            695,
        ),
        COUNTERFACT_RELATIVE_PATH: _identity(
            "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f",
            45_108_470,
        ),
        _LLAMA_STATS[0]: _identity(
            "7f5fc9b194d86ce289d607a23d02de5e7d7eb5a0833aaf6ef1698d814353585e",
            822_084_814,
        ),
        _LLAMA_STATS[1]: _identity(
            "a99a36207d3b27f129bb5323c11139853a7851c9788c58931621aa08c81fb435",
            822_084_814,
        ),
        _LLAMA_STATS[2]: _identity(
            "3a6a0c682f09a8b725e624fec1265adb46a2f7328586acc0b77c08e587e1c495",
            822_084_814,
        ),
        _LLAMA_STATS[3]: _identity(
            "a29e2d2ffb408eeba4a507d5ab9523129b2520bafc2874847f8d396796ebc830",
            822_084_814,
        ),
        _LLAMA_STATS[4]: _identity(
            "f6bbc2f240343d5244730c6422743344193dc16ee6afe64dc6cd3a97497d9050",
            822_084_814,
        ),
        _QWEN_STATS[0]: _identity(
            "935af1e9a7c5fe690471c668af225b882b3ae49039435f7143cd3c136c1f049b",
            1_435_501_774,
        ),
        _QWEN_STATS[1]: _identity(
            "7e4730511077be9b7370f29e94036d3ca08e7e0f0023395341e7496e16340519",
            1_435_501_774,
        ),
        _QWEN_STATS[2]: _identity(
            "d47f4bb2454555740c6bc46ab7f894e170cf6f5fe4bc6c26a3fc93179e18dc2a",
            1_435_501_774,
        ),
        _QWEN_STATS[3]: _identity(
            "61526068f1e1283d2e8554b514f9c7ef726957a8582b62c181f393e5589c0b0d",
            1_435_501_774,
        ),
        _QWEN_STATS[4]: _identity(
            "f5b00a555c9d860af1732ae3c48b1a04ebe8e3427873081fb0cb4cc8b5b30d46",
            1_435_501_774,
        ),
        "examples/null_space_project_Meta-Llama-3-8B-Instruct.pt": _identity(
            "6d356468c6408dca694c1907ffde31cddb99f7e67fa2e74502910d5afbede5ec",
            4_110_419_877,
        ),
        "examples/null_space_project_Qwen2.5-7B-Instruct.pt": _identity(
            "d3a9687d196f7ee06739253813917a632542717ce1166c0433aa7eec70c5c8ff",
            7_177_504_758,
        ),
    }
)


def fixed_model_spec(alias: str) -> FixedModelSpec:
    try:
        return MODEL_SPECS[alias]
    except KeyError as exc:
        raise ContractError(
            f"unknown model alias {alias!r}; expected one of {sorted(MODEL_SPECS)}"
        ) from exc


def fixed_paths_for_model(alias: str) -> tuple[str, ...]:
    """Return every input that one model's MV-0 process must verify."""

    spec = fixed_model_spec(alias)
    paths = (
        *EASYEDIT_SOURCE_PATHS,
        *EASYEDIT_RUNTIME_PATHS,
        *MEMIT_HPARAM_PATHS,
        COUNTERFACT_RELATIVE_PATH,
        *spec.covariance_paths,
        spec.projector_path,
    )
    return tuple(dict.fromkeys(paths))


def fixed_pins(
    *,
    model_alias: str | None = None,
) -> dict[str, ExpectedFileIdentity]:
    paths: Iterable[str]
    if model_alias is None:
        paths = FIXED_FILE_IDENTITIES
    else:
        paths = fixed_paths_for_model(model_alias)
    return {path: FIXED_FILE_IDENTITIES[path] for path in paths}


def preflight_fixed_artifacts(
    easyedit_root: str | Path,
    *,
    model_alias: str | None = None,
) -> ProvenanceManifest:
    """Hash the fixed files and abort on the first aggregate mismatch."""

    scope = "all" if model_alias is None else fixed_model_spec(model_alias).alias
    return preflight_pinned_files(
        fixed_pins(model_alias=model_alias),
        label=f"ODE-Edit MV-0 fixed inputs ({scope})",
        base_dir=easyedit_root,
    )


_JSON_CASE_ID = (
    rb'"(?:\\["\\/bfnrt]|\\u[0-9a-fA-F]{4}|[^"\\])*"'
    rb"|-?(?:0|[1-9][0-9]*)"
)
_CASE_ID_PATTERN = re.compile(rb'"case_id"\s*:\s*(' + _JSON_CASE_ID + rb")")


def _decode_case_id(raw_value: bytes) -> str:
    try:
        value = json.loads(raw_value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("CounterFact contains an invalid case_id token") from exc
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ContractError("CounterFact case_id must be a string or integer")
    return str(value)


def scan_counterfact_case_ids(path: str | Path) -> tuple[str, ...]:
    """Extract only lexical ``case_id`` values, without decoding row payloads."""

    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file() or source.stat().st_size == 0:
        raise ContractError(f"CounterFact source is not a non-empty file: {source}")
    with source.open("rb") as handle:
        with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as mapped:
            case_ids = tuple(
                _decode_case_id(match.group(1))
                for match in _CASE_ID_PATTERN.finditer(mapped)
            )
    if not case_ids:
        raise ContractError("CounterFact source contains no case_id fields")
    if len(set(case_ids)) != len(case_ids):
        raise ContractError("CounterFact case_id fields are not unique")
    return case_ids


@dataclass(frozen=True, slots=True)
class CounterFactSelectionManifest:
    """A selection artifact containing IDs and hashes, never row payloads."""

    seed: str
    source_sha256: str
    source_size: int
    source_row_count: int
    calibration: tuple[str, ...]
    confirmatory: tuple[str, ...]
    untouched: tuple[str, ...]

    @property
    def ordered_case_ids(self) -> tuple[str, ...]:
        return self.calibration + self.confirmatory + self.untouched

    @property
    def order_hash(self) -> str:
        return sha256_bytes(
            canonical_json(list(self.ordered_case_ids)).encode("utf-8")
        )

    @property
    def split_hash(self) -> str:
        return sha256_bytes(
            canonical_json(
                {
                    "calibration": list(self.calibration),
                    "confirmatory": list(self.confirmatory),
                    "untouched": list(self.untouched),
                }
            ).encode("utf-8")
        )

    @property
    def manifest_id(self) -> str:
        return sha256_bytes(
            canonical_json(self.to_dict(include_id=False)).encode("utf-8")
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "ode-edit-counterfact-selection/v1",
            "seed": self.seed,
            "source": {
                "sha256": self.source_sha256,
                "size": self.source_size,
                "row_count": self.source_row_count,
            },
            "case_ids": {
                "calibration": list(self.calibration),
                "confirmatory": list(self.confirmatory),
                "untouched": list(self.untouched),
            },
            "order_hash": self.order_hash,
            "split_hash": self.split_hash,
        }
        if include_id:
            payload["manifest_id"] = self.manifest_id
        return payload


def build_counterfact_selection(
    case_ids: Sequence[str],
    *,
    source_identity: ExpectedFileIdentity,
    seed: str = DEFAULT_SELECTION_SEED,
    split_counts: tuple[int, int, int] = DEFAULT_SPLIT_COUNTS,
) -> CounterFactSelectionManifest:
    """Rank IDs by a salted SHA-256 and assign the fixed 20/60/20 slices."""

    if not isinstance(seed, str) or not seed:
        raise ContractError("selection seed must be a non-empty string")
    if (
        len(split_counts) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in split_counts)
    ):
        raise ContractError("split_counts must contain three non-negative integers")
    normalized = tuple(str(case_id) for case_id in case_ids)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ContractError("selection requires non-empty unique case IDs")
    total = sum(split_counts)
    if total <= 0 or total > len(normalized):
        raise ContractError(
            f"requested {total} selected cases from only {len(normalized)} IDs"
        )

    def rank_key(case_id: str) -> tuple[bytes, str]:
        digest = hashlib.sha256(
            seed.encode("utf-8") + b"\0" + case_id.encode("utf-8")
        ).digest()
        return digest, case_id

    selected = tuple(sorted(normalized, key=rank_key)[:total])
    calibration_count, confirmatory_count, _ = split_counts
    calibration_end = calibration_count
    confirmatory_end = calibration_end + confirmatory_count
    return CounterFactSelectionManifest(
        seed=seed,
        source_sha256=source_identity.sha256,
        source_size=source_identity.size,
        source_row_count=len(normalized),
        calibration=selected[:calibration_end],
        confirmatory=selected[calibration_end:confirmatory_end],
        untouched=selected[confirmatory_end:],
    )


def generate_counterfact_selection(
    easyedit_root: str | Path,
    *,
    seed: str = DEFAULT_SELECTION_SEED,
    expected_row_count: int = COUNTERFACT_ROW_COUNT,
) -> CounterFactSelectionManifest:
    """Verify the fixed dataset, scan IDs only, then make the 20/60/20 split."""

    root = Path(easyedit_root).expanduser().resolve(strict=True)
    identity = FIXED_FILE_IDENTITIES[COUNTERFACT_RELATIVE_PATH]
    preflight_pinned_files(
        {COUNTERFACT_RELATIVE_PATH: identity},
        label="CounterFact selection source",
        base_dir=root,
    )
    case_ids = scan_counterfact_case_ids(root / COUNTERFACT_RELATIVE_PATH)
    if len(case_ids) != expected_row_count:
        raise ContractError(
            f"CounterFact row count mismatch: expected {expected_row_count}, "
            f"found {len(case_ids)}"
        )
    return build_counterfact_selection(
        case_ids,
        source_identity=identity,
        seed=seed,
    )


def _iter_top_level_json_objects(path: Path) -> Iterable[bytes]:
    """Yield object bytes from a top-level JSON array without decoding rows."""

    started = False
    finished = False
    depth = 0
    in_string = False
    escaped = False
    current: bytearray | None = None
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            for byte in chunk:
                if not started:
                    if chr(byte).isspace():
                        continue
                    if byte != ord("["):
                        raise ContractError("CounterFact source must be a JSON array")
                    started = True
                    continue
                if finished:
                    if not chr(byte).isspace():
                        raise ContractError("CounterFact has bytes after the top-level array")
                    continue
                if current is None:
                    if chr(byte).isspace() or byte == ord(","):
                        continue
                    if byte == ord("]"):
                        finished = True
                        continue
                    if byte != ord("{"):
                        raise ContractError("CounterFact array entries must be objects")
                    current = bytearray((byte,))
                    depth = 1
                    in_string = False
                    escaped = False
                    continue

                current.append(byte)
                if in_string:
                    if escaped:
                        escaped = False
                    elif byte == ord("\\"):
                        escaped = True
                    elif byte == ord('"'):
                        in_string = False
                    continue
                if byte == ord('"'):
                    in_string = True
                elif byte == ord("{"):
                    depth += 1
                elif byte == ord("}"):
                    depth -= 1
                    if depth == 0:
                        yield bytes(current)
                        current = None
    if not started or not finished or current is not None or in_string:
        raise ContractError("CounterFact source ended before its JSON array was complete")


def load_counterfact_requests(
    easyedit_root: str | Path,
    case_ids: Sequence[str],
) -> tuple[EditRequest, ...]:
    """Load only selected rows and return the four-field editor contract."""

    requested_order = tuple(str(case_id) for case_id in case_ids)
    if not requested_order or len(set(requested_order)) != len(requested_order):
        raise ContractError("requested CounterFact case IDs must be non-empty and unique")
    requested = set(requested_order)
    source = (
        Path(easyedit_root).expanduser().resolve(strict=True)
        / COUNTERFACT_RELATIVE_PATH
    )
    found: dict[str, EditRequest] = {}
    for blob in _iter_top_level_json_objects(source):
        matches = _CASE_ID_PATTERN.findall(blob)
        if len(matches) != 1:
            raise ContractError("each CounterFact row must contain exactly one case_id")
        case_id = _decode_case_id(matches[0])
        if case_id not in requested:
            continue
        try:
            row = json.loads(blob)
            rewrite = row["requested_rewrite"]
            target = rewrite["target_new"]["str"]
            sanitized = {
                "case_id": case_id,
                "prompt": rewrite["prompt"],
                "subject": rewrite["subject"],
                "target_new": target,
            }
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError(
                f"CounterFact case {case_id} lacks the fixed rewrite schema"
            ) from exc
        found[case_id] = EditRequest.from_mapping(sanitized)
        if len(found) == len(requested):
            break
    missing = requested - set(found)
    if missing:
        raise ContractError(f"CounterFact is missing requested case IDs: {sorted(missing)}")
    return tuple(found[case_id] for case_id in requested_order)


def write_selection_manifest(
    path: str | Path,
    manifest: CounterFactSelectionManifest,
) -> Path:
    """Exclusively create a compact selection JSON file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            manifest.to_dict(),
            handle,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        handle.write("\n")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-counterfact-manifest",
        description="Verify fixed artifacts or generate the case-id-only 20/60/20 split",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--easyedit-root", required=True)
    preflight.add_argument("--model", choices=sorted(MODEL_SPECS))
    selection = commands.add_parser("selection")
    selection.add_argument("--easyedit-root", required=True)
    selection.add_argument("--output", required=True)
    selection.add_argument("--seed", default=DEFAULT_SELECTION_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        manifest = preflight_fixed_artifacts(
            args.easyedit_root,
            model_alias=args.model,
        )
        print(
            json.dumps(
                {
                    "manifest_id": manifest.manifest_id,
                    "file_count": len(manifest.files),
                    "model": args.model,
                    "status": "verified",
                },
                sort_keys=True,
            )
        )
        return 0
    manifest = generate_counterfact_selection(
        args.easyedit_root,
        seed=args.seed,
    )
    write_selection_manifest(args.output, manifest)
    print(
        json.dumps(
            {
                "manifest_id": manifest.manifest_id,
                "output": str(Path(args.output).resolve()),
                "status": "written",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
