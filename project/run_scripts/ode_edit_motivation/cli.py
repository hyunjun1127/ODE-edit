"""Command-line skeleton for provenance-first motivation diagnostics.

The built-in commands perform validation and preflight only.  Model loading and
heavy solves are delegated to an explicit ``module:callable`` runtime driver.
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import (
    ContextManifest,
    ContractError,
    EditRequest,
    ProvenanceManifest,
    sanitize_edit_requests,
)
from .easyedit_bridge import EasyEditBridge


@dataclass(frozen=True, slots=True)
class DiagnosticRunSpec:
    config_path: Path
    config: Mapping[str, Any]
    requests: tuple[EditRequest, ...]
    contexts: ContextManifest
    easyedit_provenance: ProvenanceManifest


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_stdout(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))


def _pins_from_json(path: str | Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    value = _read_json(path)
    if isinstance(value, Mapping) and set(value) == {"files"}:
        value = value["files"]
    if not isinstance(value, Mapping):
        raise ContractError("pins JSON must be a mapping or {'files': mapping}")
    return value


def _cmd_preflight(args: argparse.Namespace) -> int:
    bridge = EasyEditBridge(
        args.easyedit_root,
        expected_files=_pins_from_json(args.pins),
    )
    _write_stdout(bridge.preflight().to_dict())
    return 0


def _request_payload(value: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        value = [value]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContractError("request JSON must contain an object or array of objects")
    return value


def _cmd_validate_request(args: argparse.Namespace) -> int:
    requests = sanitize_edit_requests(_request_payload(_read_json(args.input)))
    _write_stdout(
        {
            "requests": [
                {**request.to_dict(), "request_id": request.request_id}
                for request in requests
            ]
        }
    )
    return 0


def _build_run_spec(config_path: str | Path) -> DiagnosticRunSpec:
    path = Path(config_path).expanduser().resolve(strict=True)
    config = _read_json(path)
    if not isinstance(config, Mapping):
        raise ContractError("run config must be a JSON object")
    required = {"easyedit_root", "requests", "contexts"}
    missing = required - set(config)
    if missing:
        raise ContractError(f"run config is missing fields: {sorted(missing)}")
    requests = sanitize_edit_requests(_request_payload(config["requests"]))
    context_config = config["contexts"]
    if not isinstance(context_config, Mapping) or set(context_config) != {"source", "templates"}:
        raise ContractError("contexts config requires exactly 'source' and 'templates'")
    contexts = ContextManifest.freeze(
        context_config["templates"],
        source=context_config["source"],
    )
    expected = config.get("easyedit_expected_files")
    bridge = EasyEditBridge(config["easyedit_root"], expected_files=expected)
    provenance = bridge.preflight()
    return DiagnosticRunSpec(
        config_path=path,
        config=config,
        requests=requests,
        contexts=contexts,
        easyedit_provenance=provenance,
    )


def _load_driver(entrypoint: str) -> Callable[[DiagnosticRunSpec], Any]:
    if ":" not in entrypoint:
        raise ContractError("runtime driver must use 'module:callable' syntax")
    module_name, attribute = entrypoint.rsplit(":", 1)
    if not module_name or not attribute:
        raise ContractError("runtime driver must use 'module:callable' syntax")
    callback = getattr(importlib.import_module(module_name), attribute)
    if not callable(callback):
        raise ContractError(f"runtime driver is not callable: {entrypoint}")
    return callback


def _cmd_plan(args: argparse.Namespace) -> int:
    spec = _build_run_spec(args.config)
    _write_stdout(
        {
            "config_path": str(spec.config_path),
            "request_ids": [request.request_id for request in spec.requests],
            "context_id": spec.contexts.manifest_id,
            "easyedit_provenance_id": spec.easyedit_provenance.manifest_id,
            "solver": spec.config.get("solver", {"kind": "native-memit"}),
            "status": "validated; no model loaded and no solve executed",
        }
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    spec = _build_run_spec(args.config)
    result = _load_driver(args.driver)(spec)
    if result is not None:
        _write_stdout(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-motivation",
        description="Provenance-first MEMIT motivation diagnostic scaffold",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight",
        help="hash approved EasyEdit files; optionally require pinned size/hash identities",
    )
    preflight.add_argument("--easyedit-root", required=True)
    preflight.add_argument(
        "--pins",
        help="JSON mapping of every approved relative path to {sha256, size}",
    )
    preflight.set_defaults(handler=_cmd_preflight)

    validate = subparsers.add_parser(
        "validate-request",
        help="sanitize edit request JSON and print canonical request IDs",
    )
    validate.add_argument("--input", required=True)
    validate.set_defaults(handler=_cmd_validate_request)

    plan = subparsers.add_parser(
        "plan",
        help="validate a frozen-context run config without loading a model",
    )
    plan.add_argument("--config", required=True)
    plan.set_defaults(handler=_cmd_plan)

    run = subparsers.add_parser(
        "run",
        help="hand a validated run spec to an explicitly supplied integration driver",
    )
    run.add_argument("--config", required=True)
    run.add_argument("--driver", required=True, help="runtime entrypoint as module:callable")
    run.set_defaults(handler=_cmd_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
