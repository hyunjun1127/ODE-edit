"""CPU-only stored-result verification and raw-free create-once publication.

No model import, scheduler call, source mutation, or raw/checkpoint broadcast.
The existing collector's independent reducer is reused with a verified hash.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from . import reduce

SOURCE = '9cf106713e587fa08ddcfed89628b2770a55b74b'
LOCK = 'bd93594b34db719098f811294a7a8e04966356bd9981525b2b5df1145d57ceb5'
NONCE = 'ODEEDIT-USER-GH-SH1-JLZ-V10-A-B1-DIAGNOSTICS-20261003-R1'
CAPTURE = (0, 9, 13, 17, 21, 25)
COPY = ('metrics.csv', 'paired.csv', 'candidate.csv', 'layers.csv',
        'interaction-summary.csv', 'compute.csv', 'D1.png')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def member(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def verify(row):
    path = Path(row['path'])
    require(path.is_file() and path.stat().st_size == row['bytes']
            and sha(path) == row['sha256'], 'ARTIFACT_IDENTITY:' + str(path))


def read_rows(path):
    return [row for p in sorted(path.glob('chunk-*.json')) for row in load(p)['rows']]


def publish(attempt, out):
    root, collection = attempt / 'output', attempt / 'collection'
    lock = load(attempt / 'execution.lock.json')
    manifest = load(collection / 'manifest.json')
    terminal = load(root / 'terminal.json')
    collected = load(collection / 'terminal.json')
    summary = load(collection / 'summary.json')
    require(sha(attempt / 'execution.lock.json') == LOCK and lock['source'] == SOURCE, 'LOCK')
    require(lock['instruction'] == NONCE and manifest['source'] == SOURCE, 'SOURCE')
    require(terminal['source'] == SOURCE and terminal['job_id'] == '57780', 'RUN_IDENTITY')
    for obj in (terminal, collected, summary):
        require(obj['status'] == 'D1_D2_COMPLETE', 'TERMINAL_NOT_COMPLETE')
    require(not summary['missing'] and terminal['restored'], 'MISSING_OR_RESTORE')
    require(not terminal['checkpoint_saved'] and not terminal['automatic_retry'], 'NO_CP_RETRY')
    evidence = manifest['inputs'] + manifest['products'] + lock['members']
    evidence += [manifest['execution_lock'], manifest['analysis_source'], manifest['reducer'],
                 collected['manifest'], collected['report']]
    for row in evidence:
        verify(row)
    require(sha(Path(reduce.__file__)) == manifest['reducer']['sha256'], 'REDUCER_CHANGED')
    lookup = load(attempt / 'lookup-local.json')
    require(not lookup['model_results_used'] and lookup['identifiable'] == 874, 'LOOKUP')
    raw = {c: read_rows(root / 'D1' / f'c{c:02d}') for c in CAPTURE}
    for c, rows in raw.items():
        reduce.validate(rows, raw[0])
        require(reduce.metrics(rows) == summary['D1'][str(c)], 'D1_REDUCTION')
        require({k: v['denominator'] for k, v in summary['D1'][str(c)].items()}
                == dict(R=100, P=200, N=1000), 'D1_DENOMINATOR')
    masked = {mode: read_rows(root / 'D2' / mode)
              for mode in ('NONE', 'SUBJECT_ONLY', 'NONSUBJECT_ONLY', 'ALL')}
    for mode, rows in masked.items():
        require([r['identity'] for r in rows] == lookup['common_four_mask_subset'], 'D2_COVERAGE')
        require(reduce.metrics(rows) == summary['D2'][mode], 'D2_REDUCTION')
    pair_lookup = {'W0': raw[0], **{str(c): r for c, r in raw.items()},
                   **{'D2_' + mode: r for mode, r in masked.items()}}
    pair_rows = list(csv.DictReader((collection / 'paired.csv').open()))
    for row in pair_rows:
        actual = reduce.paired(pair_lookup[row['before']], pair_lookup[row['after']])[row['family']]
        require(all(int(row[k]) == v for k, v in actual.items()), 'PAIRED_REDUCTION')
    interactions, distribution = reduce.interaction(masked)
    require(interactions == load(collection / 'interaction-local.json')['rows'], 'INTERACTION')
    recorded = next(csv.DictReader((collection / 'interaction-summary.csv').open()))
    require(int(recorded['denominator']) == 874 and all(float(recorded[k]) == v
            for k, v in distribution.items()), 'INTERACTION_SUMMARY')
    fit = [load(p) for p in sorted((root / 'fit').glob('candidate-*.json'))]
    require([r['candidate'] for r in fit] == list(range(1, 26)), 'FIT_CANDIDATES')
    require(all(r['Adam_updates_after'] == min(r['candidate'], 24) for r in fit), 'FIT_UPDATES')
    commit = load(root / 'ephemeral-commit.json')
    require(commit['history_appends'] == 5 and commit['accepted_weight_copy_exact'], 'COMMIT')
    qualification = load(root / 'qualification/instrumentation-parity.json')
    require(qualification['passed'] and qualification['materialization_exact'], 'QUALIFICATION')
    mask_checks = {}
    for mode, c in (('NONE', 0), ('ALL', 25)):
        receipt = load(root / 'qualification' / ('mask-' + mode + '.json'))
        require(receipt['passed'] and all(r['passed'] for r in receipt['rows']), 'MASK_PARITY')
        subset = [r for r in raw[c] if r['identity'] in set(lookup['common_four_mask_subset'])]
        reduce.validate(masked[mode], subset)
        require(all(a[key] == b[key] for a, b in zip(masked[mode], subset)
                    for key in ('new_nll', 'true_nll', 'new_strict', 'true_strict')), 'EXACT_MASK_RECHECK')
        mask_checks[mode] = {k: v for k, v in receipt.items() if k != 'rows'}
    snapshots = [load(root / 'instrument' / f'ram-c{c:02d}.json') for c in CAPTURE[1:]]
    require([s['candidate'] for s in snapshots] == list(CAPTURE[1:]), 'SNAPSHOTS')
    require(all(not s['checkpoint_saved'] and s['from_evaluated_materialized_weights'] for s in snapshots), 'RAM_ONLY')
    accounting = load(collection / 'accounting.json')
    require(accounting['returncode'] == 0 and len(accounting['rows']) == 1, 'ACCOUNTING')
    fields = accounting['rows'][0].split('|')
    require(fields[:5] == ['57780', 'odeedit_jlz_v10_a_b1_diag_s1_gpu', 'janghj', 'COMPLETED', '0:0'], 'JOB_EXIT')
    seconds = int(fields[5])
    checks = dict(status='VERIFIED_COMPLETE', instruction=NONCE, execution_source=SOURCE,
                  execution_lock_sha256=LOCK, D1_candidates=list(CAPTURE), D1_rows_each=1300,
                  D2_masks=4, D2_rows_each=874, fits=1, candidates=25, Adam_updates=24,
                  history_appends=5, snapshot_count=5, restored=terminal['restored'],
                  manifest_member_checks=len(evidence), paired_table_rows=len(pair_rows),
                  metric_reduction='REUSED_INDEPENDENT_COLLECTOR_REDUCER_REEXECUTED_ON_STORED_RAW',
                  independent_reviewer_agent=False, GPU_forward=0, new_fit=0,
                  allocated_GPU_seconds=seconds, allocated_GPU_hours=seconds/3600,
                  checkpoint_saved=False, exact_resume='NOT_AVAILABLE',
                  raw_local_only=True, broadcast='NO_BROADCAST_NOT_REQUIRED',
                  scientific_promotion=False, mask_parity=mask_checks)
    products = {name: (collection / name).read_bytes() for name in COPY}
    products['collector-report-ko.md'] = (collection / 'report-ko.md').read_bytes()
    supplement = f'''\n## 완료 검산·게시 기록\n\nGPU57780과 CPU57781은 각각 COMPLETED/0:0이다. GPU 할당 {seconds:,}초
({seconds/3600:.4f} GPU시간), 수집기 할당 2초다. 이는 program timer와 구분한다.
이번 CPU 검산은 저장 raw의 counts/NLL/TF/paired/interaction을 다시 계산했고,
collector manifest의 입력/산출물과 frozen source bytes를 재검증했다. 새 GPU평가0.
원 수집기 보고 bytes는 [collector-report-ko.md](collector-report-ko.md)에 보존한다.
\n- [지표·분포](metrics.csv), [paired 손실·회복](paired.csv), [후보별 비용·loss](candidate.csv)
- [층별 수치](layers.csv), [D2 interaction](interaction-summary.csv), [비용](compute.csv)
- [검산·출처](publication-manifest.json), [기술 검증](qualification-summary.json)
\n![D1 고정후보별 관측](D1.png)
\n실행 source `{SOURCE}`와 이번 publication source는 구분한다.
원본 raw·큰 per-row 표·로그는 `{attempt}`에 보존하며 Git에는 포함하지 않았다.
미분리 시간·전체 모델 byte 복원·교차 플랫폼 동등성의 제한은 원 보고와 동일하다.
원래의 공개 artifact를 게시하는 것이며 새로운 방법 선택/가설 판정은 하지 않는다.
\n재현(CPU only, 새 빈 출력경로 지정):
```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.jlz_realized_subject_diagnostics.publish_completed --attempt {attempt} --out /mnt/raid5/janghj/ODE-edit/local/jlz-v10-a-b1-diagnostics/20261003-v1/publication-recheck-new
```
'''
    products['report-ko.md'] = products['collector-report-ko.md'] + supplement.encode()
    products['qualification-summary.json'] = (json.dumps(dict(instrumentation=qualification,
        masks=mask_checks, commit=dict(history_appends=5, accepted_weight_copy_exact=True),
        final_restore_recorded=True), ensure_ascii=False, indent=2)+'\n').encode()
    products['snapshot-ram-manifest.json'] = (json.dumps(snapshots, indent=2)+'\n').encode()
    input_manifest = dict(checks=checks, execution_lock=member(attempt/'execution.lock.json'),
        runner_terminal=member(root/'terminal.json'), collector_manifest=member(collection/'manifest.json'),
        collector_terminal=member(collection/'terminal.json'), source=member(Path(__file__)),
        reducer=member(Path(reduce.__file__)), local_only_inputs=manifest['inputs'],
        local_only_collection_products=manifest['products'],
        products=[dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name,data in products.items()])
    products['publication-manifest.json'] = (json.dumps(input_manifest,ensure_ascii=False,indent=2)+'\n').encode()
    out.mkdir(parents=True, exist_ok=True)
    require(all(not (out / name).exists() for name in products), 'CREATE_ONCE_OUTPUT')
    for name, data in products.items():
        with (out / name).open('xb') as f:
            f.write(data)
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    publish(args.attempt, args.out)
