"""Read-only preflight plus create-once compact receipts. Never submits jobs.

Storage arithmetic is an estimate, not a reservation, a measured EOS distribution,
or permission to reduce R512/positions/vocabulary. The initial upper bound is
replaced by measured T_i only after complete correctly bound generation.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from .config import Scope, INPUT_SHA, runtime_policy, validate_runtime_policy

ROOT = Path('/data/janghj/ODE-edit/local/en-execution-reuse/20260919-v1')
DEPS = Path('/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/deps-transformers-4.44.2')
GIB = 2**30


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8*2**20), b''):
            h.update(data)
    return h.hexdigest()


def member(path):
    p = Path(path)
    st = p.stat()
    return dict(path=str(p), realpath=str(p.resolve()), bytes=st.st_size,
                sha256=sha(p), inode=st.st_ino, device=st.st_dev)


def create_bytes(path, content):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    return member(p)


def create_json(path, value):
    return create_bytes(path, (json.dumps(value, ensure_ascii=False, sort_keys=True,
                                        indent=2, allow_nan=False)+'\n').encode())


def storage_plan(lengths=None, current_valid_token_upper=0):
    if lengths is None:
        lengths = [256]*640
        length_status = 'MAXIMUM_NOT_MEASURED'
    else:
        if (len(lengths) != 640 or any(type(t) is not int or not 1 <= t <= 256 for t in lengths)):
            raise ValueError('COMPLETE_R512_DEV128_ACTUAL_LENGTHS_REQUIRED')
        length_status = 'CALLER_SUPPLIED_REQUIRES_SEALED_GENERATION_EVIDENCE'
    if type(current_valid_token_upper) is not int or current_valid_token_upper < 0:
        raise ValueError('CURRENT_TOKEN_BOUND')
    weight = 4096*14336*4
    memory = 14336*14336*4
    # No CP copy of full pretrained model. Shared native + both schedules.
    parts = dict(teacher_FP32=sum(lengths)*128256*4,
                 reference_key_residual_FP32=sum(128+t for t in lengths)*(14336+4096)*4,
                 three_W4_M4_checkpoint_payload=3*(weight+memory),
                 largest_atomic_checkpoint_temporary=weight+memory,
                 two_saved_gradient_FP64=2*4096*14336*8,
                 two_geometry_factor_upper_FP64=2*14336*14336*8,
                 current_key_residual_upper_FP32=current_valid_token_upper*(14336+4096)*4,
                 largest_document_atomic_teacher=256*128256*4)
    return dict(length_status=length_status, documents=640, reference=512, dev=128,
                actual_positions=sum(lengths), components_bytes=parts,
                payload_teacher_key_bytes=parts['teacher_FP32']+parts['reference_key_residual_FP32'],
                estimated_bytes_without_unrecorded_overhead=sum(parts.values()),
                other_overhead='NOT_YET_MEASURED: current keys if bound zero, serialization headers, compact raw, logs, filesystem overhead',
                current_valid_token_upper=current_valid_token_upper,
                reservation=False, storage_waiver_inherited=False)


def resource_snapshot(path, plan):
    path = Path(path)
    usage = shutil.disk_usage(path)
    stat = os.statvfs(path)
    needed = plan['estimated_bytes_without_unrecorded_overhead']
    return dict(time_utc=datetime.now(timezone.utc).isoformat(), path=str(path.resolve()),
                available_bytes=usage.free, available_GiB=usage.free/GIB,
                available_inodes=stat.f_favail, estimate_bytes=needed, estimate_GiB=needed/GIB,
                minimum_shortfall_bytes=max(0, needed-usage.free),
                status='RESOURCE_BLOCKED_STORAGE' if usage.free < needed else 'ESTIMATE_FITS_NOT_EXCLUSIVE_RESERVATION',
                shared_filesystem=True, user_cleanup_observed=False, deleted_or_moved_files=0)


def seal_inputs(repo, destination):
    """Copies only exact explicitly enumerated task documents, never raw assets."""
    repo, destination = Path(repo), Path(destination)
    manifest_path = repo/'audits/global/2026-09-19-sh4-en-reuse-g256-b1-dispatch/source-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    docs = manifest['members'] + [dict(path=p) for p in (
        'messages/head/2026-09-19-sh4-en-execution-reuse-r512-g256-b1.md',
        'tasks/pending/en-execution-reuse-r512-g256-b1-sh4-20260919-v1.json',
        'plans/global/2026-09-18-single-layer-edit-preserving-correction-contract-v1.json',
        'PROTOCOL.md', 'control/gpu-concurrency-policy.tsv', 'servers/active/server4.md')]
    result = []
    for item in docs:
        source = repo/item['path']
        data = source.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if item.get('sha256', actual) != actual:
            raise ValueError('AUTHORITATIVE_BYTES_MISMATCH:'+item['path'])
        out = destination/item['path']
        create_bytes(out, data)
        result.append(dict(path=item['path'], sha256=actual, bytes=len(data), copy=str(out)))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seal-authoritative', action='store_true')
    args = p.parse_args()
    # Receipt operations are confined to this task's ignored namespace.
    output = args.output.resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError('TASK_LOCAL_OUTPUT_ONLY')
    scope = Scope().validate()
    validate_runtime_policy(runtime_policy())
    lock = Path('/data/janghj/ODE-edit/local/bpcw512/20260918-v2/B1/attempt-r1/execution.lock.json')
    prior = json.loads(lock.read_text())
    inputs = Path(prior['reference_inputs']['path'])
    if sha(inputs) != INPUT_SHA:
        raise ValueError('EXACT_R512_INPUT_SHA')
    rows = json.loads(inputs.read_text())
    if (len(rows) != 640 or [r['role'] for r in rows] != ['R512']*512+['Dev128']*128 or
        any(len(r['input_ids']) != 129 for r in rows) or
        len({r['source_row_id'] for r in rows}) != 640):
        raise ValueError('REFERENCE_INPUT_CARDINALITY_ROLE_LENGTH_ID')
    import torch, transformers, numpy, scipy
    versions = dict(torch=str(torch.__version__), transformers=transformers.__version__,
                    numpy=numpy.__version__, scipy=scipy.__version__,
                    transformers_path=transformers.__file__, cuda_initialized=torch.cuda.is_initialized())
    if versions['transformers'] != '4.44.2' or versions['cuda_initialized']:
        raise ValueError('PINNED_IMPORT_OR_CPU_PREFLIGHT')
    plan = storage_plan()
    members = seal_inputs(args.repo, ROOT/'authoritative') if args.seal_authoritative else []
    report = dict(time_utc=datetime.now(timezone.utc).isoformat(),
                  analysis_source_base=subprocess.check_output(['git','rev-parse','HEAD'], cwd=args.repo, text=True).strip(),
                  scope=asdict(scope), runtime_policy=runtime_policy(), versions=versions,
                  inputs=member(inputs), authoritative=members, storage=plan,
                  resource=resource_snapshot(ROOT, plan),
                  prior_asset_locator=member(lock), model_loads=0, model_forwards=0,
                  scheduler_queries=0, new_jobs=0, native_fit_reuse='CANDIDATE_NOT_YET_AUDITED',
                  new_generated256_teacher='NOT_BUILT', actual_Llama_validation='NOT_RUN',
                  runner_ready=False, inherited_storage_waiver=False,
                  implementation='PARTIAL_CPU_COMPONENTS; NO_EXECUTABLE_SCIENCE_RUNNER_YET')
    create_json(output, report)
    print(json.dumps({k:report[k] for k in ('resource','versions','new_jobs','actual_Llama_validation','runner_ready')}, indent=2))


if __name__ == '__main__':
    main()
