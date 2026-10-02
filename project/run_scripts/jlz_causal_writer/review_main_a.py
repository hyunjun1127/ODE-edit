"""CPU-only, read-only W5 Arm A review; no model/runtime/collector import.

Writes only aggregate review tables, checks and inventories to --output.
Historical baseline aggregates are reused, not advertised as new raw audits.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def success(row):
    return (row['true_nll'] < row['new_nll'] if row['kind'] == 'N'
            else row['new_nll'] < row['true_nll'])


def reduce(rows):
    check(len({r['identity'] for r in rows}) == len(rows), 'duplicate identity')
    for r in rows:
        check(r['kind'] in ('R', 'P', 'N'), 'unknown category')
        for label in ('true', 'new'):
            check(math.isfinite(r[label + '_nll']), 'nonfinite NLL')
            n, c = r[label + '_token_count'], r[label + '_token_correct']
            check(type(n) is int and type(c) is int and 0 <= c <= n and n > 0,
                  'invalid token denominator')
            check(type(r[label + '_strict']) is bool and r[label + '_strict'] == (c == n),
                  'strict mismatch')
    out = {}
    for kind in 'RPN':
        group = [r for r in rows if r['kind'] == kind]
        if not group:
            continue
        desired = 'true' if kind == 'N' else 'new'
        n = len(group)
        tc = sum(r[desired + '_token_correct'] for r in group)
        tn = sum(r[desired + '_token_count'] for r in group)
        out[kind] = dict(denominator=n, numerator=sum(map(success, group)),
            rate=sum(map(success, group)) / n,
            true_nll_mean=sum(r['true_nll'] for r in group) / n,
            new_nll_mean=sum(r['new_nll'] for r in group) / n,
            desired_token_count=tn, desired_token_correct=tc, token_micro=tc / tn,
            prompt_macro=sum(r[desired + '_token_correct'] / r[desired + '_token_count']
                             for r in group) / n,
            strict_numerator=sum(r[desired + '_strict'] for r in group),
            strict_denominator=n, new_strict_numerator=sum(r['new_strict'] for r in group))
    return out


def paired(before, after):
    index = {r['identity']: r for r in before}
    check(len(index) == len(before), 'duplicate paired input')
    result = []
    for kind in 'RPN':
        group = [r for r in after if r['kind'] == kind]
        check(all(r['identity'] in index for r in group), 'missing pair')
        old = [index[r['identity']] for r in group]
        for a, b in zip(old, group):
            check((a['case_id'], a['kind'], a['prompt_index']) ==
                  (b['case_id'], b['kind'], b['prompt_index']), 'paired case/order')
            for label in ('true', 'new'):
                check(a[label + '_token_count'] == b[label + '_token_count'], 'paired tokens')
                if label + '_token_identity' in a:
                    check(a[label + '_token_identity'] == b[label + '_token_identity'],
                          'paired token identity')
        n = len(group)
        result.append(dict(kind=kind, denominator=n, before_success=sum(map(success, old)),
            after_success=sum(map(success, group)),
            lost=sum(success(a) and not success(b) for a, b in zip(old, group)),
            gained=sum(not success(a) and success(b) for a, b in zip(old, group)),
            true_nll_change=sum(b['true_nll'] - a['true_nll'] for a, b in zip(old, group)) / n,
            new_nll_change=sum(b['new_nll'] - a['new_nll'] for a, b in zip(old, group)) / n))
    return result


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        check(path.read_text() == text, 'output collision: ' + str(path))
    else:
        path.write_text(text)


def table(path, rows):
    import io
    fields = list(dict.fromkeys(k for r in rows for k in r))
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    if path.exists():
        check(path.read_text() == buf.getvalue(), 'output collision: ' + str(path))
    else:
        path.write_text(buf.getvalue())


def review(attempt, repo, output):
    output.mkdir(parents=True, exist_ok=True)
    root = attempt / 'main-A'
    used = set()
    def load(p):
        used.add(p); return read(p)
    def csvrows(p):
        used.add(p); return list(csv.DictReader(p.open()))
    config = load(attempt / 'config.json'); lock = load(attempt / 'execution.lock.json')
    check(sha(attempt / 'config.json') == lock['config_sha256'], 'config SHA')
    check(sha(Path(lock['archive']['path'])) == lock['archive']['sha256'], 'runtime archive SHA')
    used.add(Path(lock['archive']['path']))
    for m in lock['source_members']:
        p = Path(m['path']); check(sha(p) == m['sha256'], 'runtime source SHA'); used.add(p)
    term = load(root / 'terminal.json')
    check(term['status'] == 'COMPLETED' and term['job_id'] == '57513'
          and term['arm'] == 'A' and term['main_commits'] == 5, 'not complete A500')
    check(term['source'] == lock['source_commit'] and term['config_sha256'] == lock['config_sha256'],
          'terminal binding')
    check(term['no_B6'] and not term['checkpoint_saved'], 'terminal noB6/noCP')
    check(not (root / 'batch-06').exists(), 'B6 exists')
    schedule = config['schedule_binding']['case_schedule']
    sp = Path(schedule['path']); check(sha(sp) == schedule['sha256'], 'schedule SHA')
    all_schedule = csvrows(sp)
    check(len(all_schedule) == 2000, 'schedule denominator')
    ids = [int(r['case_id']) for r in all_schedule[:500]]
    check(digest(ids) == config['experiment']['main']['case_ids_sha256'], '500 order')
    previous = load(root / 'initial-state.json')
    bridge = load(Path(config['w0_reuse']['receipt']['path']))
    check(sha(Path(config['w0_reuse']['receipt']['path'])) == config['w0_reuse']['receipt']['sha256'],
          'reuse bridge SHA')
    check(previous == bridge['state'], 'cold W0 identity')
    w0path = Path(bridge['observations']['path'])
    check(sha(w0path) == bridge['observations']['sha256'], 'W0 raw SHA')
    w0 = [r for r in load(w0path)['rows'] if r['case_id'] in set(ids)]
    metrics, costs, checks, layers, birth = [], [], [], [], []
    max_residual = max_key_error = 0.0
    for b in range(1, 6):
        batchroot = root / f'batch-{b:02d}'
        c = load(batchroot / 'commit.json'); entry = load(batchroot / 'entry.json')
        inp = load(batchroot / 'input.json')
        current = ids[(b-1)*100:b*100]
        check(c['current_ids'] == current == inp['ids'], 'batch input order')
        check(c['before'] == previous == entry['state'], 'W/H chain')
        check(entry['source'] == c['source'] == lock['source_commit'], 'commit source')
        check(c['config'] == digest(config), 'canonical config digest')
        check(c['batch'] == b and c['actual_B'] == 100 and c['candidate_count'] == 25
              and c['Adam_updates'] == c['backward_count'] == 24, 'fit counts')
        check(c['history_appends'] == 5 and c['no_checkpoint'] and not c['terminal_gradient_measured'],
              'commit history/terminal/noCP')
        check(c['exact_commit_sha'] == c['after']['W'] and c['history']['accepted_weight_copy_exact'],
              'exact commit receipt')
        check(c['memory_after']['events'] == b*100 and c['memory_after']['commits'] == b
              and c['memory_after']['resident_count'] == min(b*100,128), 'memory counters')
        previous = c['after']
        candidates = sorted((batchroot / 'fit').glob('candidate-*.json'))
        check(len(candidates) == 25, 'candidate count')
        for k, path in enumerate(candidates, 1):
            q = load(path)
            check(q['candidate'] == q['K_candidate'] == q['P_candidate'] == k, 'candidate identity')
            check(q['lr'] == .1 and q['warmup'] == 0, 'LR policy')
            check(q['gradient_measured'] == (k < 25) and q['Adam_updates_after'] == min(k,24), 'Adam ledger')
            check((q['gradient_norm'] is None) == (k == 25), 'terminal gradient')
            check(q['builder_RW_rows'] == 600 and q['all_current_columns']
                  and not q['builder_delta_hook'] and q['key_source'] == 'actual_lower_writer', 'builder flags')
            check((q['auxiliary'] is not None) == (k in (5,10,15,20)), 'pulse timing')
            for layer, v in q['solve'].items():
                check(v['relative_residual'] <= v['residual_tolerance'], 'solve residual')
                max_residual = max(max_residual, v['relative_residual'])
            if k == 25:
                for layer, v in q['layer'].items():
                    layers.append(dict(batch=b, layer=layer, **{n:v[n] for n in
                        ('delta_norm','actual_write_norm','g','e','key_entry_drift','P_zero_candidate_drift')}))
        parts = load(batchroot / 'fit/partitions.json')
        check(sorted(x for p in parts['current'] for x in p) == list(range(100)), 'current pulse coverage')
        check(sorted(x for p in parts['past'] for x in p) == list(range(c['reference_count'])), 'past pulse coverage')
        keycheck = load(batchroot / 'fit/terminal-key-comparison.json')
        for v in keycheck.values():
            check(v['excess'] <= 0, 'terminal key tolerance')
            max_key_error = max(max_key_error, v['max_absolute'])
        rows = []
        for path in sorted((root / f'observe-W{b:02d}').glob('chunk-*.json')):
            raw = load(path)
            check(raw['state'] == c['after'] and raw['optimizer_feedback'] is False, 'observer state')
            rows.extend(raw['rows'])
        expected = ids if b == 5 else current
        specs = [(cid,kind,i) for cid in expected for kind,n in [('R',1),('P',2),('N',10)] for i in range(n)]
        check([(r['case_id'],r['kind'],r['prompt_index']) for r in rows] == specs, 'observer case/order coverage')
        reduced = reduce(rows)
        declared = load(root / f'observe-W{b:02d}/summary.json')
        check(reduced == declared['summary'], 'independent reducer mismatch')
        check(declared['no_mutation'] and declared['memory_no_mutation'], 'observer mutation receipt')
        check(declared['row_count'] == len(rows) and digest([r['identity'] for r in rows]) == declared['row_order'],
              'observer row order digest')
        current_rows = [r for r in rows if r['case_id'] in set(current)]
        check(reduce(current_rows) == declared['current'], 'current subset')
        birth.extend(current_rows)
        if b == 5:
            final = rows; final_summary = reduced
        for kind,v in reduced.items():
            metrics.append(dict(batch=b, scope='ALL500' if b==5 else 'CURRENT100',kind=kind,**v))
        costs.append(dict(batch=b, commit_inclusive_seconds=c['seconds'],
                          observer_seconds=declared['seconds'], references=c['reference_count']))
        checks.append(dict(batch=b,requests=100,candidates=25,Adam=24,history_layer_appends=5,
                           state_chain=True,exact_commit_receipt=True,observer_restored=True))
    transitions = [dict(comparison=label,**r) for label,before in [('AT_WRITE_TO_W5',birth),('W0_TO_W5',w0)]
                   for r in paired(before,final)]
    cohort = [dict(cohort=b,comparison='AT_WRITE_TO_W5',**r) for b in range(1,6)
              for r in paired([x for x in birth if x['case_id'] in set(ids[(b-1)*100:b*100])],
                              [x for x in final if x['case_id'] in set(ids[(b-1)*100:b*100])])]
    comparison = []
    for kind,v in final_summary.items():
        comparison.append(dict(method='JLZ v7 A',job='57513',layers='L4-L8',metric=kind+'S',
          numerator=v['numerator'],denominator=v['denominator'],rate=v['rate'],
          tf_token_micro=v['token_micro'],tf_prompt_macro=v['prompt_macro'],
          tf_strict=v['strict_numerator']/v['denominator'],true_nll=v['true_nll_mean'],new_nll=v['new_nll_mean'],
          evidence='NEW_INDEPENDENT_RAW_REDUCTION'))
    base = repo / 'experiment-reports/servers/server4/cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v2'
    baseline_csv = csvrows(base/'baseline-cumulative.csv')
    for label,arm,job,ls in [('AlphaEdit','BASE_ALPHAEDIT_NATIVE','42657','L4-L8'),
                             ('AlphaEdit-BLUE','AlphaEdit_BLUE(L4+L8)','39283_1','L4+L8')]:
        for r in baseline_csv:
            if r['arm'] != arm or r['batch'] != '5': continue
            desired = 'true' if r['metric']=='NS' else 'new'
            comparison.append(dict(method=label,job=job,layers=ls,metric=r['metric'],
                numerator=int(r['numerator']),denominator=int(r['denominator']),rate=float(r['rate']),
                tf_token_micro=int(r[desired+'_token_correct'])/int(r[desired+'_token_den']),
                tf_prompt_macro='NOT_RECORDED',tf_strict=int(r[desired+'_strict_num'])/int(r[desired+'_strict_den']),
                true_nll=float(r['true_nll_prompt_mean']),new_nll=float(r['new_nll_prompt_mean']),
                evidence='REUSED_PUBLISHED_W5_AGGREGATE'))
    for r in csvrows(base/'seen-prefix.csv'):
        if r['batch']!='5' or r['population']!='ACTUAL_FULL_SEEN': continue
        comparison.append(dict(method='CAKE',job='48101',layers='L4-L8',metric=r['metric'],
            numerator=int(r['numerator']),denominator=int(r['denominator']),rate=int(r['numerator'])/int(r['denominator']),
            tf_token_micro=int(r['desired_token_correct'])/int(r['desired_token_d']),tf_prompt_macro='NOT_RECORDED',
            tf_strict=int(r['desired_strict_n'])/int(r['desired_strict_d']),
            true_nll=float(r['true_nll_mean']),new_nll=float(r['new_nll_mean']),evidence='REUSED_PUBLISHED_W5_AGGREGATE'))
    memit = repo/'experiment-reports/servers/server3/memit-history-fixed10k-20260928-v1/completion-review-r1'
    for r in csvrows(memit/'all-seen-metrics.csv'):
        if r['batch']!='5': continue
        comparison.append(dict(method='MEMIT-H',job='54007',layers='L4-L8',metric=r['metric'],
            numerator=int(r['numerator']),denominator=int(r['denominator']),rate=float(r['rate']),
            tf_token_micro=float(r['tf_token_micro']),tf_prompt_macro=float(r['tf_prompt_macro']),
            tf_strict=float(r['tf_strict']),true_nll=float(r['true_nll']),new_nll=float(r['new_nll']),
            evidence='REUSED_PUBLISHED_W5_AGGREGATE'))
    check(len(comparison)==15, 'baseline coverage')
    for r in comparison:
        check(r['denominator']=={'RS':500,'PS':1000,'NS':5000}[r['metric']], 'baseline denominator')
        check(abs(r['rate']-r['numerator']/r['denominator']) < 1e-14, 'baseline arithmetic')
        r['v7_minus_method_pp'] = 100*(final_summary[r['metric'][0]]['rate']-r['rate'])
    used.update([base/'baseline-compatibility.csv',base/'compatibility.json',base/'diagnostic-report-ko.md',
                 memit/'report-ko.md',memit/'four-method-comparison-ko.md'])
    inventory = sorted(root.rglob('*'))
    check(all(p.suffix not in ('.pt','.pth','.safetensors','.npy','.npz','.bin') for p in inventory if p.is_file()),
          'unexpected tensor output')
    used.update(p for p in inventory if p.is_file())
    summary = dict(scope='main-A only; B/collector NOT_REVIEWED',job='57513',source_commit=lock['source_commit'],
        source_tree=lock['source_tree'],config_file_sha256=lock['config_sha256'],config_canonical_digest=digest(config),
        lock_sha256=sha(attempt/'execution.lock.json'),terminal=term,checks=checks,
        final_metrics=final_summary,paired=transitions,
        active_cases_at_W5=len({r['case_id'] for r in final if r['active_at_endpoint']}),
        max_stored_SPD_relative_residual=max_residual,max_terminal_builder_key_abs_error=max_key_error,
        source_members_sha_verified=len(lock['source_members']),baseline_raw_reaudit=False,
        independent_reviewer='NOT_USED; owner independent CPU reducer only',new_GPU=0,
        monitoring_active=False,automatic_resume=False,
        limitation='Stored hashes/receipts only for W/H; no saved weights or new GPU parity. Historical runtime differs.')
    dump(output/'summary.json',summary)
    table(output/'metrics.csv',metrics);table(output/'baselines-W5.csv',comparison)
    table(output/'paired.csv',transitions);table(output/'cohort-retention.csv',cohort)
    table(output/'batch-costs.csv',costs);table(output/'terminal-layer-stats.csv',layers)
    dump(output/'artifact-index.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(used)])
    print(json.dumps(dict(final=final_summary,paired=transitions,checks=len(checks),
        SPD=max_residual,key_error=max_key_error,input_files=len(used)),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempt',type=Path,required=True)
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();review(args.attempt,args.repo,args.output)
