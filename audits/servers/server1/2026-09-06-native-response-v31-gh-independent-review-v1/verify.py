"""GH independent, CPU-only, read-only verification; prints raw-free JSON."""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

WT = Path('/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s06-native-response-v31-analysis-v1')
PKG = WT / 'experiment-reports/servers/server1/native-response-v31-b10-warm-pilot-2026-09-06-v1'
ARM = ('O_NATIVE', 'ORBFH_HIST', 'JV_NATIVE', 'ORB_RAY_N')


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def canonical(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(WT), *args])


def table(p):
    with Path(p).open(newline='') as f:
        return list(csv.DictReader(f))


def near(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12), (a, b)


def metrics(evaluation):
    out = {}
    for label, kind, reverse in [('RS', 'rewrite', False), ('PS', 'rephrase', False), ('NS', 'locality', True)]:
        new = evaluation[kind + '_target_new']
        old = evaluation[kind + '_target_true']
        hits = []
        for a, b in zip(new, old, strict=True):
            assert (a['case_id'], a['prompt_index'], a['prompt']) == (b['case_id'], b['prompt_index'], b['prompt'])
            assert math.isfinite(a['nll']) and math.isfinite(b['nll'])
            hits.append(a['nll'] > b['nll'] if reverse else a['nll'] < b['nll'])
        assert len({(a['case_id'], a['prompt_index']) for a in new}) == len(new)
        out[label + '_n'] = sum(hits)
        out[label + '_d'] = len(hits)
        out[kind + '_new_nll_mean'] = sum(a['nll'] for a in new) / len(new)
        out[kind + '_strict_n'] = sum(a['all_tokens_correct'] for a in (old if reverse else new))
    return out


started = time.monotonic()
result = {'kind': 'GH_INDEPENDENT_READ_ONLY_AUDIT', 'scientific_promotion': False}
result['reviewed_head'] = git('rev-parse', 'HEAD').decode().strip()
assert result['reviewed_head'] == '347a892a910317b606443cf53d9155ae6a460831'
assert not git('status', '--porcelain', '--untracked-files=no')
review = read(PKG / 'REVIEW_READY.json')
member_counts = {}
for rel in ('', 'primary', 'diagnostics'):
    p = PKG / rel
    manifest = read(p / 'manifest.json')
    assert canonical(manifest['members']) == manifest['members_root']
    for m in manifest['members']:
        f = p / m['path']
        assert f.is_file() and not f.is_symlink()
        assert f.stat().st_size == m['bytes'] and sha(f) == m['sha256'], str(f)
        if rel == '':
            blob = git('show', 'HEAD:' + str(f.relative_to(WT)))
            assert hashlib.sha256(blob).hexdigest() == m['sha256']
    receipt = read(p / 'rooted-receipt.json')
    ident = receipt.pop('identity')
    assert canonical(receipt) == ident
    assert sha(p / 'manifest.json') == receipt['manifest_sha256']
    assert manifest['members_root'] == receipt['members_root']
    member_counts[rel or 'final'] = len(manifest['members'])
result['package_members_rehashed'] = member_counts
inputs = {}
for rel in ('primary', 'diagnostics'):
    inv = read(PKG / rel / 'external-inputs.json')
    assert canonical(inv) == read(PKG / rel / 'manifest.json')['input_root']
    for m in inv:
        if m['path'] in inputs:
            assert inputs[m['path']]['sha256'] == m['sha256']
        inputs[m['path']] = m
for name, m in inputs.items():
    p = Path(name)
    assert p.is_file() and not p.is_symlink()
    assert p.stat().st_size == m['bytes'] and sha(p) == m['sha256'], name
result['external_inputs'] = {'unique_files': len(inputs), 'bytes': sum(m['bytes'] for m in inputs.values()), 'mismatches': 0}
for m in read(PKG / 'primary/source-manifest.json')['members']:
    blob = git('show', m['head'] + ':' + m['path'])
    assert len(blob) == m['bytes'] and hashlib.sha256(blob).hexdigest() == m['sha256']
for label, lineage in review['lineages'].items():
    assert git('rev-parse', lineage['head'] + '^{tree}').decode().strip() == lineage['tree']
result['source_lineages'] = {'checked': len(review['lineages']), 'mismatches': 0}
primary = Path(review['primary_raw_root'])
result['endpoint_reductions'] = {}
summary = []
for fixture, roots, csvfile in [('D10B', {i: primary for i in range(4)}, 'primary/pilot_main_table.csv'),
                                ('H10', {int(i): Path(p) for i, p in review['audit_raw_roots'].items()}, 'diagnostics/audit_main_table.csv')]:
    lookup = {(int(r['cell']), r['arm']): r for r in table(PKG / csvfile)}
    counts = [0, 0, 0]
    for cell in range(4):
        for arm in ARM:
            raw = read(roots[cell] / f'cell-{cell}/{fixture}-{arm}.json')
            if fixture == 'D10B':
                ep = raw.get('endpoint') or raw['historical']['endpoint']
            else:
                v = raw['result']
                ep = v if 'evaluation' in v else v['endpoint']
            assert raw['w0_restore'] is True
            facts = metrics(ep['evaluation'])
            row = lookup[(cell, arm)]
            for key, value in facts.items():
                near(value, row[key])
            for j, key in enumerate(('RS_d', 'PS_d', 'NS_d')):
                counts[j] += facts[key]
            before = raw['old_before']; after = raw['old_after']
            bn = {v['case_id']: v['nll'] for v in before['rewrite_target_new']}
            bt = {v['case_id']: v['nll'] for v in before['rewrite_target_true']}
            an = {v['case_id']: v['nll'] for v in after['rewrite_target_new']}
            at = {v['case_id']: v['nll'] for v in after['rewrite_target_true']}
            assert bn.keys() == bt.keys() == an.keys() == at.keys()
            failures = sum(bn[i] < bt[i] and an[i] >= at[i] for i in bn)
            near(failures, row['new_failure_n'])
            near(sum(bn[i] < bt[i] for i in bn), row['entry_success_d'])
            summary.append(dict(fixture=fixture, cell=cell, arm=arm, **facts, old_new_failures=failures))
    assert counts == [160, 320, 1600]
    result['endpoint_reductions'][fixture] = dict(arms=16, denominators=counts, mismatches=0)
result['endpoint_summary'] = summary
turns = []
dissipation = []
for cell in range(4):
    for arm in ('JV_NATIVE', 'ORB_RAY_N'):
        raw = read(primary / f'cell-{cell}/D10B-{arm}.json')
        refs = {n['qN_ref'] for n in raw['nodes']}
        assert len(refs) == 1
        for n in raw['nodes']:
            s = {r['comparator']: r for r in raw['same_state_fields'] if r['node'] == n['node'] and r['lambda_value'] == 0.1}
            j, r = s['JV_NATIVE']['c'], s['ORB_RAY_N']['c']
            dot = sum(a*b for a,b in zip(j,r,strict=True))
            qj, qr = sum(a*a for a in j), sum(a*a for a in r)
            beta = dot/qr
            turn = sum((a-beta*b)**2 for a,b in zip(j,r,strict=True))/qj
            near(turn, s['ORB_RAY_N']['Rturn'])
            assert s['JV_NATIVE']['gain'] >= s['ORB_RAY_N']['gain'] - 1e-12
            turns.append(turn)
            if arm == 'JV_NATIVE':
                err = n['gain'] - n['response_sq'] - .1*n['qN']
                near(err, n['dissipation_residual'])
                assert abs(err) < 1e-10 and n['qN'] <= n['V']/.2 + 1e-10
                dissipation.append(abs(err))
result['joint_same_state'] = dict(checked=len(turns), positive=sum(v>0 for v in turns), min_Rturn=min(turns), max_dissipation_error=max(dissipation))
fidelity = read(PKG / 'primary/gpu_fidelity_checks.json')
p_res = []
for cell in fidelity:
    for f in cell['fixtures']:
        assert f['status'] == 'PASS' and f['entry_pointer_bytes_restore']
        assert f['activation_parity']['pass_check'] and f['logit_parity']['pass_check']
        for layer in f['layers']:
            assert layer['adjacent_pass'] and layer['residual_scaling_half_exact']
            if 'stored_projector_operator_action_relative' in layer:
                p_res.append(layer['stored_projector_operator_action_relative'])
result['gpu_receipt_audit'] = {'fixtures': 8, 'layer_FD_receipts': 40, 'projector_operator_relative_max': max(p_res), 'model_rerun': 0}
ledger = read(PKG / 'gpu-hour-ledger.json')
observed = subprocess.check_output(['sacct','-n','-P','-j','37675,37677,37681,37688,37693,37698','--format=JobID,JobIDRaw,State,ElapsedRaw,AllocTRES','-S','2026-09-06']).decode()
rows = [r.split('|') for r in observed.splitlines() if r]
totals = [0]*4
for a in ledger['attempts']:
    members = [r for r in rows if r[0] == a['job'] or r[0].startswith(a['job']+'.')]
    assert members and max(int(r[3]) for r in members) == a['charged_seconds']
    totals[a['cell']] += a['charged_seconds']
assert totals == ledger['cumulative_seconds_by_cell'] and max(totals) <= 7200 and sum(totals) <= 28800
result['budget'] = dict(cell_seconds=totals, gpu_hours=sum(totals)/3600, independent_scheduler_match=True)
result['elapsed_seconds'] = time.monotonic()-started
result['status'] = 'VERIFIED_RECORDED_DATA_NOT_FULL_MEASUREMENT_CONTRACT_SIGNOFF'
print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
