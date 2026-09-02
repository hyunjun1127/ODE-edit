#!/usr/bin/env python3
"""Validate Slurm host-memory requests against the tracked server policy."""

from __future__ import annotations

import argparse
import dataclasses
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = REPO_ROOT / "servers" / "slurm-memory-policy.tsv"
MEMORY_RE = re.compile(r"^(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<unit>[KMGTP]?)$", re.IGNORECASE)
SERVER_MARKER_RE = re.compile(r"^#\s*ODEEDIT_SLURM_SERVER=(\S+)\s*$", re.MULTILINE)
MEM_DIRECTIVE_RE = re.compile(r"^#SBATCH\s+--mem(?:=|\s+)(\S+)\s*$", re.MULTILINE)
NODE_DIRECTIVE_RE = re.compile(
    r"^#SBATCH\s+(?:--nodelist(?:=|\s+)|-w\s+)(\S+)\s*$", re.MULTILINE
)
GPU_PATTERNS = (
    re.compile(r"^#SBATCH\s+--gres(?:=|\s+)gpu(?::[^:\s]+)?:(\d+)\s*$", re.MULTILINE),
    re.compile(r"^#SBATCH\s+--gpus(?:=|\s+)(?:[^:\s]+:)?(\d+)\s*$", re.MULTILINE),
    re.compile(
        r"^#SBATCH\s+--gpus-per-node(?:=|\s+)(?:[^:\s]+:)?(\d+)\s*$",
        re.MULTILINE,
    ),
)


class PolicyError(ValueError):
    """A typed policy or Slurm script validation error."""


@dataclasses.dataclass(frozen=True)
class ServerPolicy:
    server: str
    slurm_node: str
    scheduler_max_gib_per_gpu: int
    repo_request_max_gib_per_gpu: int

    @property
    def request_max_mib_per_gpu(self) -> int:
        return self.repo_request_max_gib_per_gpu * 1024


def parse_memory_mib(value: str) -> int:
    """Parse a Slurm memory token into MiB without rounding upward."""

    match = MEMORY_RE.fullmatch(value.strip())
    if match is None:
        raise PolicyError(f"invalid Slurm memory value: {value!r}")
    amount = float(match.group("value"))
    unit = match.group("unit").upper()
    factors = {"": 1.0, "K": 1.0 / 1024.0, "M": 1.0, "G": 1024.0, "T": 1024.0**2, "P": 1024.0**3}
    memory_mib = int(amount * factors[unit])
    if memory_mib < 1:
        raise PolicyError(f"Slurm memory request must be positive: {value!r}")
    return memory_mib


def load_policy(path: Path = DEFAULT_POLICY) -> dict[str, ServerPolicy]:
    policies: dict[str, ServerPolicy] = {}
    nodes: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PolicyError(f"cannot read policy file {path}: {exc}") from exc
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) != 4:
            raise PolicyError(f"{path}:{line_number}: expected four tab-separated fields")
        server, node, scheduler_text, request_text = fields
        try:
            scheduler_gib = int(scheduler_text)
            request_gib = int(request_text)
        except ValueError as exc:
            raise PolicyError(f"{path}:{line_number}: memory limits must be integers") from exc
        if not server or not node or scheduler_gib < 1 or request_gib < 1:
            raise PolicyError(f"{path}:{line_number}: invalid policy row")
        if request_gib >= scheduler_gib:
            raise PolicyError(
                f"{path}:{line_number}: repo ceiling must leave at least 1 GiB below scheduler ceiling"
            )
        if server in policies or node in nodes:
            raise PolicyError(f"{path}:{line_number}: duplicate server or node")
        policies[server] = ServerPolicy(server, node, scheduler_gib, request_gib)
        nodes.add(node)
    required = {"server1", "server2", "server3", "server4"}
    if set(policies) != required:
        raise PolicyError(f"policy must define exactly {sorted(required)}; got {sorted(policies)}")
    return policies


def check_request(
    policy: ServerPolicy,
    requested_gpus: int,
    requested_memory: str,
    local_limit_mib_per_gpu: int | None = None,
) -> tuple[int, int]:
    if requested_gpus < 1:
        raise PolicyError("requested GPU count must be a positive integer")
    requested_mib = parse_memory_mib(requested_memory)
    limit_per_gpu = policy.request_max_mib_per_gpu
    if local_limit_mib_per_gpu is not None:
        if local_limit_mib_per_gpu < 1:
            raise PolicyError("local memory limit must be a positive integer")
        limit_per_gpu = min(limit_per_gpu, local_limit_mib_per_gpu)
    maximum_mib = requested_gpus * limit_per_gpu
    return requested_mib, maximum_mib


def _server_for_script(text: str, policies: dict[str, ServerPolicy]) -> str:
    markers = SERVER_MARKER_RE.findall(text)
    nodes = NODE_DIRECTIVE_RE.findall(text)
    if len(markers) > 1 or len(nodes) > 1:
        raise PolicyError("multiple server markers or node directives")
    marker = markers[0] if markers else ""
    node = nodes[0] if nodes else ""
    by_node = {policy.slurm_node: server for server, policy in policies.items()}
    node_server = by_node.get(node, "")
    if node and not node_server:
        raise PolicyError(f"unrecognized Slurm node: {node}")
    if marker == "portable":
        if node:
            raise PolicyError("portable marker cannot be combined with a fixed node")
        return marker
    if marker and marker not in policies:
        raise PolicyError(f"unrecognized ODEEDIT_SLURM_SERVER marker: {marker}")
    if marker and node_server and marker != node_server:
        raise PolicyError(f"server marker {marker} conflicts with node {node}")
    server = marker or node_server
    if not server:
        raise PolicyError("missing fixed node or # ODEEDIT_SLURM_SERVER=<server|portable>")
    return server


def _gpu_count(text: str) -> int:
    counts: list[int] = []
    for pattern in GPU_PATTERNS:
        counts.extend(int(value) for value in pattern.findall(text))
    if len(counts) > 1:
        raise PolicyError("multiple GPU-count directives")
    return counts[0] if counts else 0


def audit_script(path: Path, policies: dict[str, ServerPolicy]) -> tuple[str, int, str, int, int]:
    if not path.is_file() or path.is_symlink():
        raise PolicyError("Slurm script must be a regular non-symlink file")
    text = path.read_text(encoding="utf-8")
    memory_values = MEM_DIRECTIVE_RE.findall(text)
    if len(memory_values) != 1:
        raise PolicyError(f"expected exactly one explicit #SBATCH --mem directive; found {len(memory_values)}")
    if re.search(r"^#SBATCH\s+--mem-per-cpu(?:=|\s+)", text, re.MULTILINE):
        raise PolicyError("--mem-per-cpu is not a substitute for the mandatory --mem directive")
    server = _server_for_script(text, policies)
    gpu_count = _gpu_count(text)
    accounting_gpus = max(gpu_count, 1)
    if server == "portable":
        request_limit_mib = min(item.request_max_mib_per_gpu for item in policies.values())
    else:
        request_limit_mib = policies[server].request_max_mib_per_gpu
    requested_mib = parse_memory_mib(memory_values[0])
    maximum_mib = accounting_gpus * request_limit_mib
    if requested_mib > maximum_mib:
        raise PolicyError(
            f"requested {requested_mib} MiB exceeds {maximum_mib} MiB for server={server}, "
            f"requested_gpus={gpu_count}"
        )
    return server, gpu_count, memory_values[0], requested_mib, maximum_mib


def _tracked_sbatch_files() -> list[Path]:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "ls-files", "*.sbatch"], text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PolicyError(f"cannot enumerate tracked Slurm scripts: {exc}") from exc
    return [REPO_ROOT / relative for relative in output.splitlines() if relative]


def command_request(args: argparse.Namespace) -> int:
    policies = load_policy(Path(args.policy))
    if args.server not in policies:
        raise PolicyError(f"unknown server: {args.server}")
    requested_mib, maximum_mib = check_request(
        policies[args.server], args.gpus, args.mem, args.local_limit_mib_per_gpu
    )
    if requested_mib > maximum_mib:
        print(
            f"DENY_MEMORY server={args.server} requested_mem_mib={requested_mib} "
            f"max_mem_mib={maximum_mib} requested_gpus={args.gpus} "
            "decision=pending_resource_cap"
        )
        return 4
    print(
        f"ALLOW_MEMORY_POLICY server={args.server} requested_mem_mib={requested_mib} "
        f"max_mem_mib={maximum_mib} requested_gpus={args.gpus} decision=submit_now"
    )
    return 0


def command_audit(args: argparse.Namespace) -> int:
    policies = load_policy(Path(args.policy))
    paths = [Path(item).resolve() for item in args.paths] if args.paths else _tracked_sbatch_files()
    failures: list[str] = []
    for path in paths:
        try:
            audit_script(path, policies)
        except (OSError, UnicodeError, PolicyError) as exc:
            try:
                label = str(path.relative_to(REPO_ROOT))
            except ValueError:
                label = str(path)
            failures.append(f"{label}: {exc}")
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        print(f"SLURM_MEMORY_POLICY_AUDIT_FAIL checked={len(paths)} failures={len(failures)}", file=sys.stderr)
        return 4
    print(f"SLURM_MEMORY_POLICY_AUDIT_PASS checked={len(paths)} failures=0")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(policy=str(DEFAULT_POLICY))
    subparsers = parser.add_subparsers(dest="command", required=True)

    request = subparsers.add_parser("request", help="validate one job-total memory request")
    request.add_argument("--server", required=True)
    request.add_argument("--gpus", required=True, type=int)
    request.add_argument("--mem", required=True, help="Slurm memory token, e.g. 59G or 60416M")
    request.add_argument("--local-limit-mib-per-gpu", type=int)
    request.add_argument("--policy", default=str(DEFAULT_POLICY))
    request.set_defaults(handler=command_request)

    audit = subparsers.add_parser("audit", help="audit tracked or named .sbatch files")
    audit.add_argument("paths", nargs="*")
    audit.add_argument("--policy", default=str(DEFAULT_POLICY))
    audit.set_defaults(handler=command_audit)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except PolicyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
