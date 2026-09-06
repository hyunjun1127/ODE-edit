"""Create-once records, outcome-blind sample seal, source/asset preflight."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from project.run_scripts.ordered_response_barrier_ode import preflight as historical

NAMESPACE = 'ODEEDIT-V31-PARALLEL-PILOT-20260906'
ARM_ORDER = ['O_NATIVE', 'ORBFH_HIST', 'JV_NATIVE', 'ORB_RAY_N']
SCIENCE = dict(schema='native-response-v31.science.v1', primary_lambda=.1,
    T=2., N=4, h=.5, normalization='SOURCE_EXACT_N0', first_hit_controls_dynamics=False,
    native_rhs_divisor=1, qN_ref_capture_count=1, fixed_z_recompute_count=0,
    shadow_lambda=[.01,.1,1.], refinement_N=[2,4,8], arm_order=ARM_ORDER,
    historical_ORBFH=dict(T=1,N=4,h=.25,ordered_layer_visits=5),
    rescue=False, adaptive_gain=False, scientific_promotion=False)


def save(path, obj):
    path = Path(path).absolute()
    for part in [path, *path.parents]:
        if part.is_symlink():
            raise RuntimeError(f'SYMLINK_PUBLICATION_BOUNDARY: {part}')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False, default=str)+'\n').encode()
    fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def select_samples(repo, dataset):
    payload = Path(dataset).read_bytes()
    if len(payload) != historical.DATASET_BYTES or hashlib.sha256(payload).hexdigest() != historical.DATASET_SHA256:
        raise RuntimeError('DATASET_IDENTITY_BOUNDARY')
    rows = json.loads(payload)
    reserved = json.loads((repo / historical.STREAM_SEAL_RELATIVE).read_text())
    excluded = {int(r['case_id']) for r in reserved['requests']}
    if len(excluded) != 1000:
        raise RuntimeError('RESERVED_STREAM_BOUNDARY')
    ranked = sorted((hashlib.sha256((NAMESPACE+'|'+str(int(r['case_id']))).encode()).hexdigest(),
                     int(r['case_id']), r) for r in rows if int(r['case_id']) not in excluded)
    selected = ranked[:38]
    if len({r[1] for r in selected}) != 38:
        raise RuntimeError('SAMPLE_DENOMINATOR_BOUNDARY')
    labels = ['G0A','G0B','D1','D2','D3','H1','H2','H3'] + ['D10A']*10 + ['D10B']*10 + ['H10']*10
    records = [dict(ordinal=i, fixture=labels[i], case_id=case, rank_sha256=rank,
                    raw_record_sha256=historical.canonical_hash(raw),
                    request_sha256=historical.canonical_hash(raw['requested_rewrite']))
               for i, (rank,case,raw) in enumerate(selected)]
    return dict(schema='native-response-v31.samples.v1', namespace=NAMESPACE,
                selection='SHA256(namespace|canonical_case_id), case_id tie break; exclude reserved1000',
                dataset_sha256=historical.DATASET_SHA256, reserved_order_root=historical.ORDER_ROOT,
                reserved_stream_root=historical.STREAM_ROOT, reserved_overlap_count=0,
                outcome_selection_count=0, records=records,
                ordered_root=historical.canonical_hash(records))


def prepare(repo, output):
    repo, output = Path(repo).absolute(), Path(output).absolute()
    if output.exists():
        raise RuntimeError('CREATE_ONCE_RUN_EXISTS')
    output.mkdir(parents=True, mode=0o700)
    save(output/'science.lock.json', SCIENCE)
    save(output/'sample.lock.json', select_samples(repo, historical.EASYEDIT_ARTIFACT_ROOT/historical.DATASET_RELATIVE))
    source = dict(head=git(repo,'rev-parse','HEAD'), tree=git(repo,'rev-parse','HEAD^{tree}'),
                  tracked_dirty=git(repo,'status','--porcelain','--untracked-files=no'),
                  historical_report_ref='f2dcfd4ab6fcb2917ae0a29cbc384cf95a82eb3c',
                  historical_execution_heads=['9b167455c1562be076aa2437fa63565187804aa7','f47d035dc1ac35b1adcd8c036336ea69c5f32a4e'])
    if source['tracked_dirty']:
        raise RuntimeError('DIRTY_EXECUTION_SOURCE')
    source['easyedit'] = historical.validate_easyedit_source(historical.OFFICIAL_EASYEDIT_ROOT)
    save(output/'source.lock.json', source)
    result = historical.validate_model_artifacts(repo, historical.EASYEDIT_ARTIFACT_ROOT,
                                                 historical.HF_HUB_CACHE_ROOT, deep_hash=True)
    save(output/'assets.lock.json', result)
    save(output/'resource.lock.json', dict(server='server1', node='devbox',gpu_per_task=1,
        max_parallel_gpu_jobs=1,max_extra_gpu_hours_total=8,max_extra_gpu_hours_per_cell=2,
        mem_mib=182272,cpus_per_task=12,array='0-3%1',export='NONE',
        retry_budget_shared=True, unrelated_mutation=0))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(prepare(args.repo,args.output))
