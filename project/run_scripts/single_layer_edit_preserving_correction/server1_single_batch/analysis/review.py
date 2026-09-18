"""CPU-only, immutable ENFC b001 evidence reducer. No runtime/model imports.

Raw prompts and token arrays are read locally and never published. Pairing uses
case, prompt index, exact prompt, both targets and token inventories, not row order.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ARMS = ['N4', 'SCALE', 'CA', 'KL-P', 'EN-S', 'EN-F', 'EN-COV', 'EN-F4']
METRICS = {'RS': 'rewrite', 'PS': 'rephrase', 'NS': 'locality'}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n')


def csvout(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def pair_rows(raw, metric):
    prefix = METRICS[metric]
    def index(rows):
        out = {}
        for r in rows:
            key = (r['case_id'], r['prompt_index'], r['prompt'])
            assert key not in out, 'duplicate prompt identity'
            assert math.isfinite(r['nll'])
            assert len(r['token_correct']) == len(r['target_token_ids']) > 0
            assert bool(r['all_tokens_correct']) == all(r['token_correct'])
            out[key] = r
        return out
    new = index(raw[prefix + '_target_new']); true = index(raw[prefix + '_target_true'])
    assert new.keys() == true.keys(), 'unpaired target rows'
    result = {}
    for k, n in new.items():
        t = true[k]
        key = k + (n['target'], t['target'], tuple(n['target_token_ids']), tuple(t['target_token_ids']))
        margin = t['nll'] - n['nll']
        desired = -margin if metric == 'NS' else margin
        result[key] = dict(case_id=k[0], prompt_index=k[1], new_nll=n['nll'], true_nll=t['nll'],
            margin=margin, desired_margin=desired, success=desired > 0, tie=desired == 0,
            strict=n['all_tokens_correct'], true_strict=t['all_tokens_correct'],
            token_correct=sum(n['token_correct']), token_count=len(n['token_correct']))
    return result


def quantiles(values):
    import numpy as np
    x = np.asarray(list(values), dtype=float)
    assert x.size and np.isfinite(x).all()
    return dict(count=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p90=float(np.quantile(x, .9)), p95=float(np.quantile(x, .95)),
                p99=float(np.quantile(x, .99)), min=float(x.min()), max=float(x.max()))


def arm_ledger(out, arm):
    d = out / 'arms' / arm
    if (d / 'reuse.json').exists():
        reuse = read(d / 'reuse.json')
        return read(reuse['source_ledger']['path']), Path(reuse['source']['path']), 'REUSE'
    return read(d / 'selection-ledger.json'), d / 'final-L4.pt', 'RUN_MISSING'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--first-only', action='store_true')
    args = p.parse_args(); out, dest = args.input, args.output
    dest.mkdir(parents=True, exist_ok=True)
    docs = {a: read(out / 'observers' / (a + '.json')) for a in ARMS + ['W0', 'RAND+', 'RAND-']}
    tables = []; ledgers = {}
    for a in ARMS:
        d = docs[a]; ledger, endpoint, reuse = arm_ledger(out, a); ledgers[a] = ledger
        row = dict(arm=a, scope='b001_unique100', integrity='PRELIMINARY_STORED_AGGREGATES', optimization=reuse)
        for m in METRICS:
            row[m + '_n'] = d['metrics'][m]['numerator']; row[m + '_d'] = d['metrics'][m]['denominator']
            row[m + '_delta_pp_vs_N4'] = 100 * (d['metrics'][m]['rate'] - docs['N4']['metrics'][m]['rate'])
        row.update({k: v for k, v in d['strict'].items() if k != 'rows'})
        row.update(S64_KL=read(out / 'observers' / (a + '-S64-output-KL.json'))['loss'],
            Dev128_KL=read(out / 'observers' / (a + '-Dev128.json'))['loss'],
            correction_norm=ledger.get('actual_delta_norm', 0 if a == 'N4' else 'NOT_RECORDED'),
            fallback=ledger.get('native_fallback', False if a == 'N4' else 'NOT_RECORDED'), stop=ledger['stop_reason'])
        tables.append(row)
    csvout(dest / 'first-eight-arm-table.csv', tables)
    if args.first_only:
        print(json.dumps(tables, indent=2)); return
    pairs = {a: {m: pair_rows(d['raw'], m) for m in METRICS} for a, d in docs.items()}
    counts, tails, changes, secondary, generation, changed_ids = [], [], [], [], [], []
    checks = []
    for a, d in docs.items():
        for m, rows in pairs[a].items():
            base = pairs['N4'][m]; assert rows.keys() == base.keys(), (a, m, 'identity mismatch')
            expected = {'RS': 100, 'PS': 200, 'NS': 1000}[m]
            n = sum(r['success'] for r in rows.values()); assert len(rows) == expected
            stored = d['metrics'][m]; assert n == stored['numerator'] and len(rows) == stored['denominator']
            indexed = {(r['case_id'], r['prompt_index']): r for r in stored['rows']}
            assert len(indexed) == len(rows)
            for r in rows.values():
                s = indexed[(r['case_id'], r['prompt_index'])]
                for field in ['new_nll', 'true_nll', 'margin', 'success']:
                    assert s[field] == r[field], (a, m, field)
            counts.append(dict(arm=a, metric=m, n=n, d=len(rows), rate=n/len(rows), ties=sum(r['tie'] for r in rows.values())))
            for field in ['new_nll', 'true_nll', 'desired_margin']:
                tails.append(dict(arm=a, metric=m, quantity=field, **quantiles(r[field] for r in rows.values())))
                delta = [r[field]-base[k][field] for k, r in rows.items()]
                tails.append(dict(arm=a, metric=m, quantity=field+'_delta_vs_N4', **quantiles(delta)))
            lost = sum(base[k]['success'] and not r['success'] for k, r in rows.items())
            gained = sum(not base[k]['success'] and r['success'] for k, r in rows.items())
            changes.append(dict(arm=a, metric=m, reference='N4', denominator=len(rows), lost=lost, gained=gained,
                                unchanged=len(rows)-lost-gained, delta_pp=100*(gained-lost)/len(rows)))
            if a != 'W0':
                for key,r in rows.items():
                    if r['success'] != base[key]['success']:
                        changed_ids.append(dict(arm=a,metric=m,case_id=r['case_id'],prompt_index=r['prompt_index'],
                            identity_sha256=hashlib.sha256(json.dumps(key,ensure_ascii=False).encode()).hexdigest(),
                            transition='gained' if r['success'] else 'lost',native_margin=base[key]['desired_margin'],
                            selected_margin=r['desired_margin']))
            secondary.append(dict(arm=a, metric=m, new_TF_strict_n=sum(r['strict'] for r in rows.values()),
                prompt_d=len(rows), new_token_correct=sum(r['token_correct'] for r in rows.values()),
                target_token_d=sum(r['token_count'] for r in rows.values())))
        bycase = {}
        for m in ['RS', 'PS']:
            for r in pairs[a][m].values(): bycase.setdefault(r['case_id'], []).append(r)
        assert len(bycase) == 100 and all(len(r) == 3 for r in bycase.values())
        strict = sum(all(r['strict'] for r in rs) for rs in bycase.values())
        joint = sum(all(r['success'] for r in rs) for rs in bycase.values())
        assert strict == d['strict']['R_two_P_strict'] and joint == d['strict']['R_two_P_NLL_joint']
        w0 = pairs['W0']['NS']; ns = pairs[a]['NS']
        retain = sum(w0[k]['success'] and r['success'] for k,r in ns.items()); den = sum(r['success'] for r in w0.values())
        if 'numerator' in d.get('W0_correct_NS', {}):
            assert d['W0_correct_NS']['numerator'] == retain and d['W0_correct_NS']['denominator'] == den
        secondary.append(dict(arm=a, metric='JOINT_AND_W0', R_two_P_strict=strict, R_two_P_NLL_joint=joint,
            W0_correct_N_retained=retain, W0_correct_N_d=den, W0_correct_N_lost=den-retain,
            W0_incorrect_N_gained=sum(not w0[k]['success'] and r['success'] for k,r in ns.items())))
        g = d.get('generation', [])
        if g:
            assert len(g)==100
            for r in g:
                assert len(r['generated_token_ids'])==r['generated_length']
                assert len(r['target_token_ids'])==r['target_length']
                if not r['target_over_32_censored']:
                    assert r['target_prefix_match']==(r['generated_token_ids'][:r['target_length']]==r['target_token_ids'])
            generation.append(dict(arm=a, d=len(g), target_prefix_match=sum(bool(r['target_prefix_match']) for r in g),
                censored=sum(r['target_over_32_censored'] for r in g), EOS=sum(r['stopped_on_original_eos'] for r in g),
                at_limit=sum(r['reached_max_new_tokens'] for r in g), **quantiles(r['generated_length'] for r in g)))
        checks.append(dict(arm=a, raw_pair_identity='PASS', raw_nll_reduction='PASS', joint='PASS', W0_N='PASS'))
        if a != 'W0':
            for suffix,den in [('S64-output-KL',64),('Dev128',128)]:
                obs=read(out/'observers'/(a+'-'+suffix+'.json'))
                assert len(obs['rows'])==den
                assert abs(sum(r['loss'] for r in obs['rows'])/den-obs['loss'])<1e-14
    for r in tables: r['integrity']='CPU_RAW_REDUCTION_PASS_NOT_T_PASS'
    for name, rows in [('final-eight-arm-table',tables),('endpoint-counts',counts),('request-tail',tails),
                       ('paired-transitions',changes),('changed-identities',changed_ids),('strict-and-retention',secondary),('generation',generation)]:
        csvout(dest/(name+'.csv'), rows)
    trials, mechanics, costs = [], [], []
    for a, l in ledgers.items():
        mechanics.append({**{k:v for k,v in l.items() if isinstance(v,(str,int,float,bool)) or v is None}, 'arm': a})
        for t in l.get('trials',[]):
            trials.append(dict(arm=a, **{k:v for k,v in t.items() if isinstance(v,(str,int,float,bool)) or v is None}))
        for k,v in l.get('counters',{}).items(): costs.append(dict(scope=a, component=k, value=v, provenance='S4_REUSED' if a in ARMS[:5] else 'S1_NEW', additive='NO_OVERLAP_UNRESOLVED'))
    term=read(out/'terminal.json')
    def flat(v, path):
        for k,x in v.items():
            if isinstance(x,dict): yield from flat(x,path+'.'+k)
            elif isinstance(x,(int,float)): yield dict(scope=path,component=k,value=x,provenance='S1_NEW',additive='NO_OVERLAP_UNRESOLVED')
    costs.extend(flat(term,'terminal'))
    costs.append(dict(scope='Slurm50098_parent_only',component='allocated_GPU_seconds',value=9147,provenance='ONE_SACCT_READ',additive='TOTAL_DO_NOT_ADD_COMPONENTS'))
    for name,rows in [('mechanism',mechanics),('trials',trials),('compute',costs)]: csvout(dest/(name+'.csv'),rows)
    dump(dest/'reducer-checks.json',dict(checks=checks, unique_requests=100, primary_arm_requests=800,
        statistical_independent_batches=1, full_numerical_validation='NOT_ESTABLISHED', new_GPU_calls=0))
    files=[p for p in out.rglob('*') if p.is_file() and p.suffix=='.json']
    dump(dest/'input-json-manifest.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)])
    print('CPU_REDUCTION_PASS',len(counts),'count rows;',len(tails),'distribution rows')


if __name__ == '__main__': main()
