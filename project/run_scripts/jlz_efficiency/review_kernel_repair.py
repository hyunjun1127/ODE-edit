"""Offline scalar review; no model, torch, scheduler, or remote access."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    inputs = {}

    def read(name):
        p = a.run / name
        inputs[name] = dict(bytes=p.stat().st_size, sha256=sha(p))
        return json.loads(p.read_text())

    def write(name, value):
        (a.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

    def table(name, rows):
        with (a.output / name).open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    lock = read('execution.lock.json')
    runtime = read('output/runtime.json')
    terminal = read('output/terminal.json')
    reuse = read('output/reuse-receipt.json')
    kernels = read('output/kernels.json')
    ref = read('output/B100-oracle-0.json')
    cand = read('output/B100-oracle-1.json')
    entry = read('output/B100-entry.json')
    probe = read('output/small-probe.json')
    receipt = read('collected/receipt.json')
    summary = read('output/B100-summary.json')
    for item in receipt['artifact_manifest']:
        p = a.run / 'output' / item['name']
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256']
    assert not receipt['independent_reducer_errors']
    assert terminal['budget']['small_total'] == 12
    assert terminal['budget']['separate']['B100'] == 2 == summary['completed']
    assert reuse['native_new_requests'] == 0 and reuse['reference_return_trace_exact']
    assert reuse['reused_fixed']['E123_MB8'] == 'UNQUALIFIED'
    assert reuse['reused_native']['native-batched.json'] == 'UNQUALIFIED'
    assert len(kernels['cases']) == 12 and len(kernels['comparisons']) == 6
    assert kernels['schema'] == 2
    comparison = cand['comparison']
    failed = [k for k, v in comparison['errors'].items() if v > comparison['limits'][k]]
    assert failed and comparison['status'] == 'UNQUALIFIED'
    assert comparison['same_materialization']
    assert terminal['B100_status'] == 'CANDIDATE_EXCLUDED_PARITY'
    assert not (a.run / 'output/REPAIR_INITIAL_VALID.json').exists()
    assert terminal['state_restored'] and not terminal['checkpoint_saved']
    for p in probe['probes']:
        assert p['entry'] == p['post_restore'] and p['restored']
        assert p['appends'] == 5 and p['evaluated_weight_bitwise'] and p['L4_key_bitwise']
    table('b100-parity.csv', [dict(metric=k, error=v, limit=comparison['limits'][k],
          status='FAIL' if k in failed else 'PASS') for k, v in comparison['errors'].items()])
    table('warmup-only.csv', [dict(route=d['route'], warmup=d['warmup'],
          seconds=d['timing']['seconds'], peak_allocated_bytes=d['timing']['peak_allocated'],
          peak_reserved_bytes=d['timing']['peak_reserved'], forward_chunks=d['timing']['forward_chunks'],
          backward_chunks=d['timing']['backward_chunks'], interpretation='WARMUP_NOT_SPEED_QUALIFICATION') for d in (ref, cand)])
    table('kernel-comparisons.csv', [dict(B=d['B'], reference_kernel=d['reference_kernel'],
          candidate_kernel=d['candidate_kernel'], reference_median=d['reference']['median'],
          candidate_median=d['candidate']['median'], reference_min=d['reference']['min'],
          candidate_max=d['candidate']['max'], status=d['status'], scope='SYNTHETIC_KERNEL_ONLY')
          for d in kernels['comparisons']])
    metrics = []
    paired = []
    observations = []
    for route in ('REF_MB2', 'E123_MB4'):
        d = read('output/small-observer-' + route + '.json')
        rows = d['rows']
        assert len({r['identity'] for r in rows}) == len(rows) == 52
        observations.append({r['identity']: r for r in rows})
        for kind, n in [('R', 4), ('P', 8), ('N', 40)]:
            part = [r for r in rows if r['kind'] == kind]
            assert len(part) == n
            desired = 'true' if kind == 'N' else 'new'
            assert all(math.isfinite(r[k]) for r in part for k in ('true_nll', 'new_nll'))
            metrics.append(dict(route=route, kind=kind, denominator=n,
                successes=sum(r['true_nll'] < r['new_nll'] if kind == 'N' else r['new_nll'] < r['true_nll'] for r in part),
                strict=sum(r[desired + '_strict'] for r in part),
                mean_true_nll=sum(r['true_nll'] for r in part)/n,
                mean_new_nll=sum(r['new_nll'] for r in part)/n))
    assert observations[0].keys() == observations[1].keys()
    for kind in ('R', 'P', 'N'):
        rows = [(r, observations[1][key]) for key, r in observations[0].items() if r['kind'] == kind]
        def good(r):
            return r['true_nll'] < r['new_nll'] if kind == 'N' else r['new_nll'] < r['true_nll']
        paired.append(dict(kind=kind, denominator=len(rows), lost=sum(good(x) and not good(y) for x,y in rows),
            gained=sum(not good(x) and good(y) for x,y in rows),
            max_nll_abs=max(abs(x[k]-y[k]) for x,y in rows for k in ('true_nll','new_nll'))))
    table('small-observer.csv', metrics)
    table('small-paired.csv', paired)
    table('coverage.csv', [dict(stage=k, status=v) for k,v in [
        ('old fixed32/short84/native', 'IDENTITY_LOCKED_REUSE; UNQUALIFIED_PRESERVED'),
        ('new reference R reconstruction', '12_CALLS_EXACT_TRACE; NOT_CRASH_RESUME'),
        ('kernel', '12_CASES_6_COMPARISONS_SAVED'), ('small RAM probe/observer', 'PASS_52_PROMPTS'),
        ('B100 entry/key', str(entry['comparison'].get('status', entry['comparison']))),
        ('B100 warmup pair', 'UNQUALIFIED_CANDIDATE_EXCLUDED'),
        ('B100 measured3pairs', 'NOT_RUN_GATE_EXCLUSION'),
        ('B100 observer1300', 'NOT_RUN_GATE_EXCLUSION'),
        ('new native fit', '0_REUSED_PRIOR'), ('new scientific chain', '0_NOT_AUTHORIZED')]])
    evidence = dict(inputs=inputs, execution_source=runtime['source'], failed_metrics=failed,
        GPU=runtime['GPU'], GPU_UUID=runtime['GPU_UUID'], program_seconds=terminal['seconds'],
        allocated_GPU_seconds=393, old_failed_GPU_seconds=401, total_lineage_GPU_seconds=794,
        allocation_evidence='owner bounded sacct 56758/56759; collector parent sacct',
        kernel_bug='REPAIRED_ACTUAL_SUMMARY_SAVED', numerical_qualification='B100_CANDIDATE_EXCLUDED',
        repair_initial_gate='NOT_PASSED_B100_PARITY; TERMINAL_ALREADY_OBSERVED',
        checkpoint_saved=False, exact_resume='NOT_AVAILABLE', old56684_mutation=False,
        independent_review='separate post-review receipt', scientific_promotion=False,
        broadcast='NO_BROADCAST_NOT_REQUIRED')
    write('evidence.json', evidence)
    write('rooted-receipt.json', dict(inputs=inputs, analysis_source_sha256=sha(Path(__file__)),
        outputs={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(a.output.iterdir())},
        scope='CPU_SCALAR_REVIEW_NOT_GPU_REEXECUTION', status='REVIEWED_WITH_NUMERICAL_EXCLUSION'))
    print(json.dumps(dict(failed_metrics=failed, output=str(a.output), artifacts=len(list(a.output.iterdir())))))


if __name__ == '__main__':
    main()
