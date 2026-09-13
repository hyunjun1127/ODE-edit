"""Independent, CPU-only reduction of sealed low-cost core evaluation JSON.

No runtime/evaluator imports. CounterFact success is recomputed from raw NLL
pairs; pairing uses complete case/prompt/target identity, never row index alone.
Published outputs contain aggregates and hashes only. Quantiles use linear
interpolation at (n-1)*p. No missing-value fill or model execution is supported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

ARMS = ('N4', 'S875', 'S75', 'FULL8', 'RES8', 'REFIT4')
STATES = ('W0', 'ENTRY') + ARMS
MULT = {'RS': 1, 'PS': 2, 'NS': 10}
DATA_SHA = '3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1'


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def panel_digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def quantile(values, p):
    x = sorted(values)
    if not x:
        return None
    k = (len(x) - 1) * p
    a, b = math.floor(k), math.ceil(k)
    return x[a] + (x[b] - x[a]) * (k - a)


def distribution(values):
    x = list(values)
    require(all(math.isfinite(v) for v in x), 'NONFINITE_DISTRIBUTION')
    return dict(n=len(x), mean=fmean(x) if x else None,
                median=quantile(x, .5), q25=quantile(x, .25), q75=quantile(x, .75),
                iqr=(quantile(x, .75)-quantile(x, .25)) if x else None,
                p90=quantile(x, .9), p95=quantile(x, .95), p99=quantile(x, .99),
                min=min(x) if x else None, max=max(x) if x else None)


def outcome(row, metric):
    new, true = row['new_nll'], row['true_nll']
    require(math.isfinite(new) and math.isfinite(true), 'NONFINITE_PAIR')
    return new < true if metric in ('RS', 'PS') else true < new


def desired(row, metric):
    return row['true_nll'] - row['new_nll'] if metric in ('RS', 'PS') else row['new_nll'] - row['true_nll']


def key(row):
    return row['case_id'], row['prompt_index'], row['identity']


def check_rows(rows, metric, expected=None):
    keys = [key(r) for r in rows]
    require(len(set(keys)) == len(keys), 'DUPLICATE_ROW')
    if expected is not None:
        require(keys == expected, 'PROMPT_ORDER_OR_TARGET_IDENTITY')
    for r in rows:
        require(bool(r['success']) == outcome(r, metric), 'SUCCESS_PAIR_MISMATCH')
        require(r['margin'] == r['true_nll'] - r['new_nll'], 'MARGIN_MISMATCH')
        require(r['desired_margin'] == desired(r, metric), 'DESIRED_MARGIN_MISMATCH')
        for side in ('new', 'true'):
            c, n = r[side+'_token_correct'], r[side+'_token_count']
            require(type(c) is int and type(n) is int and 0 <= c <= n and n > 0, 'TOKEN_COUNT')
            require(bool(r[side+'_strict']) == (c == n), 'STRICT_ALL_TARGET_TOKEN_MISMATCH')
    return keys


def aggregate(rows, metric):
    n = len(rows)
    cases = {}
    for r in rows:
        cases.setdefault(r['case_id'], []).append(r)
    out = dict(prompt_denominator=n, request_denominator=len(cases),
               numerator=sum(outcome(r, metric) for r in rows),
               rate=sum(outcome(r, metric) for r in rows)/n if n else None,
               ties=sum(r['new_nll'] == r['true_nll'] for r in rows),
               request_all_success_numerator=sum(all(outcome(r, metric) for r in rr) for rr in cases.values()),
               order_sha256=digest([key(r) for r in rows]),
               bit_order_sha256=digest([(r['identity'], outcome(r, metric)) for r in rows]))
    for side in ('new', 'true'):
        out[side+'_strict_numerator'] = sum(bool(r[side+'_strict']) for r in rows)
        out[side+'_strict_rate'] = out[side+'_strict_numerator']/n if n else None
        out[side+'_request_all_strict_numerator'] = sum(all(bool(r[side+'_strict']) for r in rr) for rr in cases.values())
        out[side+'_request_all_strict_rate'] = out[side+'_request_all_strict_numerator']/len(cases) if cases else None
        out[side+'_token_correct'] = sum(r[side+'_token_correct'] for r in rows)
        out[side+'_token_denominator'] = sum(r[side+'_token_count'] for r in rows)
        out[side+'_token_accuracy'] = out[side+'_token_correct']/out[side+'_token_denominator'] if n else None
    return out


def mmlu_prediction(probabilities):
    require(len(probabilities)==4 and all(math.isfinite(x) and 0<=x<=1 for x in probabilities), 'MMLU_PROBABILITY')
    top=max(probabilities)
    return probabilities.index(top) if probabilities.count(top)==1 else -1


def pairs(before, after, metric):
    require([key(r) for r in before] == [key(r) for r in after], 'PAIRED_IDENTITY_OR_ORDER_MISMATCH')
    require(len({key(r) for r in before}) == len(before), 'PAIRED_DUPLICATE')
    n = len(before)
    a, b = [outcome(r, metric) for r in before], [outcome(r, metric) for r in after]
    lost = sum(x and not y for x, y in zip(a, b))
    gained = sum(not x and y for x, y in zip(a, b))
    out = dict(prompt_denominator=n, request_denominator=len({r['case_id'] for r in before}),
               before_numerator=sum(a), after_numerator=sum(b), lost=lost, gained=gained,
               retained=sum(x and y for x, y in zip(a, b)),
               failed_both=sum(not x and not y for x, y in zip(a, b)),
               delta_numerator=sum(b)-sum(a), delta_pp=100*(sum(b)-sum(a))/n if n else None,
               conditional_loss_rate=lost/sum(a) if sum(a) else None,
               conditional_gain_rate=gained/(n-sum(a)) if n-sum(a) else None,
               before_ties=sum(r['new_nll']==r['true_nll'] for r in before),
               after_ties=sum(r['new_nll']==r['true_nll'] for r in after),
               order_sha256=digest([key(r) for r in before]),
               transition_sha256=digest([(r['identity'], x, y) for r,x,y in zip(before,a,b)]))
    require(gained-lost == out['delta_numerator'], 'TRANSITION_ARITHMETIC')
    for side in ('new', 'true'):
        out[side+'_strict_delta_numerator'] = sum(bool(r[side+'_strict']) for r in after)-sum(bool(r[side+'_strict']) for r in before)
    for field in ('new_nll', 'true_nll', 'desired_margin'):
        vals = [(desired(y,metric)-desired(x,metric)) if field=='desired_margin' else y[field]-x[field] for x,y in zip(before,after)]
        out.update({field+'_delta_'+k: v for k,v in distribution(vals).items()})
    return out


def expected_rows(records, metric):
    result = []
    for r in records:
        w = r['requested_rewrite']
        pp = ([w['prompt'].format(w['subject'])] if metric == 'RS' else
              r['paraphrase_prompts' if metric == 'PS' else 'neighborhood_prompts'])
        require(len(pp) == MULT[metric], 'SOURCE_PROMPT_CARDINALITY')
        result.extend((r['case_id'], i, digest([r['case_id'], i, p, w['target_new']['str'], w['target_true']['str']])) for i,p in enumerate(pp))
    return result


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def group_rows(rows, label):
    if label == 'ALL':
        return rows
    return [r for r in rows if r.get('historical_status', {}).get('status') == label]


def reduce(root, out, dataset, panel_path, contract_path):
    require(file_sha(dataset) == DATA_SHA, 'FIXED_DATASET_SHA')
    records = json.loads(dataset.read_text())
    panel = json.loads(panel_path.read_text())
    contract = json.loads(contract_path.read_text())
    inputs_path=root.parent/'inputs.json'
    inputs=json.loads(inputs_path.read_text())
    wiki_path=Path(inputs['wiki_panel'])
    wiki_input=json.loads(wiki_path.read_text())
    wiki_expected=[(r['ordinal'],panel_digest(r['input_ids']),len(r['input_ids'])-1) for r in wiki_input['rows']]
    mmlu_path=Path(inputs['mmlu100'])
    mmlu_input=json.loads(mmlu_path.read_text())
    mmlu_expected=[mmlu_input[i] for i in panel['MMLU']['groups']['development']['indices']]
    require([panel_digest(r) for r in mmlu_expected]==panel['MMLU']['groups']['development']['row_hashes'], 'MMLU_SOURCE_ROW_BINDING')
    last={}
    for i,rec in enumerate(records[:5000]):
        w=rec['requested_rewrite']
        if w.get('relation_id') is not None:
            last[(w['subject'],w['relation_id'])]=i
    annotation={}
    for saved in panel['Historical']['rows']:
        i=saved['ordinal'];w=records[i]['requested_rewrite'];k=(w['subject'],w.get('relation_id'));latest=last.get(k)
        status='UNKNOWN_RELATION' if latest is None else ('SUPERSEDED' if records[latest]['requested_rewrite']['target_new']['str']!=w['target_new']['str'] else 'ACTIVE_TARGET')
        annotation[saved['case_id']]=dict(ordinal=i,case_id=saved['case_id'],status=status,latest_known_event_ordinal=latest,subject_relation_sha256=panel_digest(k),known_exact_key_only=True)
    panels = {}
    for p in ('current', 'historical'):
        pp = panel[p.title()]
        rr = [records[r['ordinal']] for r in pp['rows']]
        for saved, rec in zip(pp['rows'], rr):
            require(saved['case_id'] == rec['case_id'] and saved['record_sha256'] == panel_digest(rec), 'PANEL_SOURCE_RECORD')
        panels[p] = rr
    docs, summary, dist, inventory, core = {}, [], [], [], []
    sentinels, checks, outliers = [], [], []
    for state in STATES:
        path = root/state/'evaluation.json'
        d = json.loads(path.read_text())
        docs[state] = d
        inventory.append(dict(state=state, local_path=str(path), bytes=path.stat().st_size, sha256=file_sha(path)))
        require(d['evaluation_nonmutation'] is True, 'EVALUATION_NONMUTATION_RECEIPT')
        headline = dict(state=state, role='REFERENCE' if state in ('W0','ENTRY') else 'CORE_ENDPOINT')
        for p, rr in panels.items():
            require(d[p]['requests'] == len(rr), 'REQUEST_COUNT')
            require(d[p]['request_order'] == digest([r['case_id'] for r in rr]), 'REQUEST_ORDER')
            require(d[p]['evaluator_layout'] == 'HISTORICAL_MICROBATCH16_MANUAL_LEFT_PADDING_NO_POSITION_OVERRIDE', 'EVALUATOR_LAYOUT')
            for metric in MULT:
                saved = d[p]['metrics'][metric]
                rows = saved['rows']
                check_rows(rows, metric, expected_rows(rr, metric))
                if p=='historical':
                    require(all(r['historical_status']==annotation[r['case_id']] for r in rows), 'HISTORICAL_ENTRY5000_ANNOTATION')
                agg = aggregate(rows, metric)
                for sk, ak in [('numerator','numerator'), ('denominator','prompt_denominator'), ('rate','rate'), ('bit_order_sha256','bit_order_sha256')]:
                    require(saved[sk] == agg[ak], 'STORED_AGGREGATE_'+sk)
                headline.update({p+'_'+metric+'_'+k: agg[v] for k,v in [('n','numerator'),('d','prompt_denominator'),('rate','rate')]})
                groups = ('ALL','ACTIVE_TARGET','SUPERSEDED') if p=='historical' else ('ALL',)
                for group in groups:
                    sub = group_rows(rows, group)
                    base = dict(state=state, panel=p, metric=metric, population=group)
                    summary.append(dict(base, **aggregate(sub, metric)))
                    cases = {}
                    for r in sub:
                        cases.setdefault(r['case_id'], []).append(r)
                    for field in ('new_nll','true_nll','desired_margin'):
                        values = [desired(r,metric) if field=='desired_margin' else r[field] for r in sub]
                        dist.append(dict(base, field=field, aggregation_unit='PROMPT', **distribution(values)))
                        cluster = [fmean(desired(r,metric) if field=='desired_margin' else r[field] for r in gr) for gr in cases.values()]
                        dist.append(dict(base, field=field, aggregation_unit='REQUEST_CLUSTER_MEAN', **distribution(cluster)))
                        if group == 'ALL' and values:
                            for extreme, index in [('min', min(range(len(values)), key=lambda i: values[i])),('max', max(range(len(values)), key=lambda i: values[i]))]:
                                outliers.append(dict(base, field=field, extreme=extreme, identity=sub[index]['identity'], case_sha256=digest(sub[index]['case_id']), value=values[index], finite=True, identity_verified=True))
                checks.append(dict(state=state,panel=p,metric=metric,rows=len(rows),request_count=len(rr),identity_order='PASS',raw_pair_reduction='PASS',stored_count_hash='PASS',nonfinite=0,duplicate=0))
        wr = d['wiki']['rows']
        require([(r['ordinal'],r['input_sha256'],r['predicted_tokens']) for r in wr]==wiki_expected,'WIKI_INPUT_TOKENS_ORDER_AND_MASK')
        require(len(wr)==128 and len({r['input_sha256'] for r in wr})==128, 'WIKI_CARDINALITY')
        require(sum(r['predicted_tokens'] for r in wr)==panel['Wiki']['predicted_tokens']==d['wiki']['seal']['predicted_tokens'], 'WIKI_TOKENS')
        require(d['wiki']['seal']['token_rows_sha256']==panel['Wiki']['token_rows_sha256'], 'WIKI_SEAL')
        wn = [r['nll'] for r in wr]
        require(all(math.isfinite(v) for v in wn), 'WIKI_NONFINITE')
        require(math.isclose(sum(wn), d['wiki']['numerator_nll_sum'], abs_tol=1e-10), 'WIKI_SUM')
        require(math.isclose(fmean(wn), d['wiki']['mean_nll'], abs_tol=1e-12), 'WIKI_MEAN')
        headline['wiki_sequence_mean_nll'] = fmean(wn)
        headline['wiki_sequences'] = len(wr)
        headline['wiki_predicted_tokens'] = sum(r['predicted_tokens'] for r in wr)
        for k, v in distribution(wn).items():
            sentinels.append(dict(state=state,panel='wiki',metric='sequence_nll_'+k,value=v,denominator=len(wr),unit='NATS_PER_TOKEN_THEN_SEQUENCE'))
        sentinels.append(dict(state=state,panel='wiki',metric='token_weighted_nll_mean',value=sum(r['nll']*r['predicted_tokens'] for r in wr)/sum(r['predicted_tokens'] for r in wr),denominator=sum(r['predicted_tokens'] for r in wr),unit='TOKEN_WEIGHTED_SEQUENCE_MEANS'))
        mr = d['mmlu']['rows']
        require([r['row_sha256'] for r in mr]==panel['MMLU']['groups']['development']['row_hashes'], 'MMLU_ORDER')
        require(len(mr)==32 and len({r['row_sha256'] for r in mr})==32, 'MMLU_CARDINALITY')
        correct, invalid = 0, 0
        for r,source_row in zip(mr,mmlu_expected):
            require(r['gold']==source_row['answer'], 'MMLU_GOLD_SOURCE_BINDING')
            nn, pr = r['alternative_nll'], r['alternative_probability']
            require(len(nn)==len(pr)==4 and all(math.isfinite(x) for x in nn+pr), 'MMLU_ALTERNATIVE')
            # Stored probabilities preserve the executed backend's exponent rounding.
            # Recompute unique maximum from them, independently of saved prediction.
            pred=mmlu_prediction(pr)
            require(all(math.isclose(p,math.exp(-x),rel_tol=1e-6,abs_tol=1e-40) for p,x in zip(pr,nn)), 'MMLU_PROBABILITY_NLL_BINDING')
            require(pred==r['prediction'], 'MMLU_PREDICTION')
            require(bool(r['tie_or_underflow_invalid'])==(pred==-1), 'MMLU_INVALID')
            require(int(pred==r['gold'])==r['correct'], 'MMLU_CORRECT')
            correct += int(pred==r['gold']); invalid += int(pred==-1)
        require(correct==d['mmlu']['correct'] and invalid==d['mmlu']['invalid'] and d['mmlu']['denominator']==32, 'MMLU_COUNTS')
        headline.update(mmlu_correct=correct,mmlu_d=32,mmlu_wrong=32-correct-invalid,mmlu_invalid=invalid)
        core.append(headline)
    for row in core:
        baseline = next(r for r in core if r['state']=='N4')
        for p in panels:
            for metric in MULT:
                row[p+'_'+metric+'_delta_vs_N4_pp'] = 100*(row[p+'_'+metric+'_rate']-baseline[p+'_'+metric+'_rate'])
        row['wiki_delta_vs_N4_nats'] = row['wiki_sequence_mean_nll']-baseline['wiki_sequence_mean_nll']
        row['mmlu_delta_vs_N4_count'] = row['mmlu_correct']-baseline['mmlu_correct']
    write_csv(out/'first-core-table.csv', [r for r in core if r['state'] in ARMS])
    write_csv(out/'reference-and-core-table.csv', core)
    comparisons = [(base, arm) for base in ('W0','ENTRY','N4') for arm in ARMS if base!=arm] + [('S75','RES8'),('REFIT4','RES8'),('FULL8','RES8'),('W0','ENTRY')]
    transitions, delta_dist, general_transitions = [], [], []
    for before, after in comparisons:
        for p in panels:
            for metric in MULT:
                a, b = docs[before][p]['metrics'][metric]['rows'], docs[after][p]['metrics'][metric]['rows']
                groups = ('ALL','ACTIVE_TARGET','SUPERSEDED') if p=='historical' else ('ALL',)
                for group in groups:
                    aa, bb = group_rows(a,group), group_rows(b,group)
                    base = dict(before=before,after=after,panel=p,metric=metric,population=group)
                    transitions.append(dict(base, **pairs(aa,bb,metric)))
                    bycase = {}
                    for x,y in zip(aa,bb):
                        bycase.setdefault(x['case_id'], []).append((x,y))
                    for field in ('new_nll','true_nll','desired_margin'):
                        dd=[fmean(desired(y,metric)-desired(x,metric) if field=='desired_margin' else y[field]-x[field] for x,y in gr) for gr in bycase.values()]
                        delta_dist.append(dict(base,field=field,aggregation_unit='REQUEST_CLUSTER_MEAN_PAIRED_DELTA',**distribution(dd)))
        wa, wb = docs[before]['wiki']['rows'], docs[after]['wiki']['rows']
        require([(r['ordinal'],r['input_sha256'],r['predicted_tokens']) for r in wa]==[(r['ordinal'],r['input_sha256'],r['predicted_tokens']) for r in wb], 'WIKI_PAIRED_IDENTITY')
        delta=[y['nll']-x['nll'] for x,y in zip(wa,wb)]
        general_transitions.append(dict(before=before,after=after,panel='wiki',denominator=len(delta),worse=sum(v>0 for v in delta),better=sum(v<0 for v in delta),unchanged=sum(v==0 for v in delta),**distribution(delta)))
        ma, mb = docs[before]['mmlu']['rows'], docs[after]['mmlu']['rows']
        require([r['row_sha256'] for r in ma]==[r['row_sha256'] for r in mb], 'MMLU_PAIRED_IDENTITY')
        general_transitions.append(dict(before=before,after=after,panel='mmlu',denominator=32,lost=sum(x['correct'] and not y['correct'] for x,y in zip(ma,mb)),gained=sum(not x['correct'] and y['correct'] for x,y in zip(ma,mb)),retained=sum(x['correct'] and y['correct'] for x,y in zip(ma,mb)),failed_both=sum(not x['correct'] and not y['correct'] for x,y in zip(ma,mb)),prediction_changed=sum(x['prediction']!=y['prediction'] for x,y in zip(ma,mb)),delta_correct=sum(y['correct']-x['correct'] for x,y in zip(ma,mb))))
    frontier = quality(docs, contract)
    tables={'first-core-table.csv':[r for r in core if r['state'] in ARMS],'reference-and-core-table.csv':core,'endpoint-metrics.csv':summary,'nll-distributions.csv':dist,'paired-transitions.csv':transitions,'paired-request-distributions.csv':delta_dist,'sentinel-metrics.csv':sentinels,'paired-sentinels.csv':general_transitions,'quality-frontier.csv':frontier,'metric-integrity.csv':checks,'metric-raw-inputs.csv':inventory,'nll-outliers.csv':outliers}
    for name, rows in tables.items():
        write_csv(out/name,rows)
    receipt=dict(status='INDEPENDENT_RAW_PAIR_REDUCTION_PASS',evaluation_files=8,core_endpoints=6,raw_prompt_pairs=sum(r['rows'] for r in checks),duplicate=0,nonfinite=0,imputation=0,gpu=0,source_reducer_import=0,quantiles='LINEAR_(N-1)*P',missing_policy='NO_IMPUTATION_EMPTY_GROUP_N0',tables={name:dict(rows=len(rows),sha256=file_sha(out/name)) for name,rows in tables.items()},source_sha256=file_sha(Path(__file__)),inputs=inventory,
                 supporting_input_files=[dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha(p)) for p in (inputs_path,wiki_path,mmlu_path,dataset,panel_path,contract_path)],historical_annotation_scope='EXACT_SUBJECT_RELATION_AND_TARGET_AT_ENTRY5000_124_ACTIVE_4_SUPERSEDED',
                 scientific_claim='PENDING_GH_REVIEW',audit_evaluation='NOT_RUN',suffix='NOT_RUN',wiki_mask='ALL_NEXT_TOKEN_POSITIONS_24999',mmlu='ALTERNATIVE_UNIQUE_MAX_PROBABILITY_INTEGER_COUNTS_NOT_F1')
    (out/'independent-metric-receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    return core, receipt


def quality(docs, contract):
    refs=contract['quality_diagnostic_references']; result=[]
    for state in ARMS:
        def rr(s,p,m,g='ALL'):
            return group_rows(docs[s][p]['metrics'][m]['rows'],g)
        def add(name,value,unit,denominator,threshold=None,inclusive=True,metric='NOT_APPLICABLE'):
            threshold=refs[name] if threshold is None else threshold
            status='NOT_RECORDED' if value is None else ('WITHIN' if (value<=threshold if inclusive else value<threshold) else 'EXCEEDS')
            result.append(dict(state=state,reference=name,metric=metric,value=value,threshold=threshold,status=status,unit=unit,denominator=denominator,comparator='N4',enforcement='DESCRIPTIVE_ONLY_NO_REJECTION'))
        c={m:pairs(rr('N4','current',m),rr(state,'current',m),m) for m in MULT}
        add('current_R_new_failures_among_native_success_max',c['RS']['lost'],'PROMPT_COUNT',100)
        add('current_P_net_loss_max_count',-c['PS']['delta_numerator'],'PROMPT_COUNT',200)
        for m in ('RS','PS'):
            rp=[y['new_nll']-x['new_nll'] for x,y in zip(rr('N4','current',m),rr(state,'current',m))]
            add('current_RP_mean_new_NLL_increase_max_nats',fmean(rp),'PROMPT_NLL_NATS',len(rp),metric=m)
            add('current_RP_paired_NLL_harm_q95_max_nats',quantile(rp,.95),'PROMPT_PAIRED_NEW_NLL_DELTA_NATS',len(rp),metric=m)
        for m,name in [('RS','current_R_strict_net_loss_max_count'),('PS','current_P_strict_net_loss_max_count')]:
            add(name,-c[m]['new_strict_delta_numerator'],'PROMPT_ALL_TARGET_TOKEN_COUNT',c[m]['prompt_denominator'])
        h={m:pairs(rr('N4','historical',m,'ACTIVE_TARGET'),rr(state,'historical',m,'ACTIVE_TARGET'),m) for m in ('RS','PS')}
        add('active_historical_R_new_failures_max',h['RS']['lost'],'ACTIVE_PROMPT_COUNT',h['RS']['prompt_denominator'])
        add('active_historical_P_net_loss_max_pp',-h['PS']['delta_pp'] if h['PS']['delta_pp'] is not None else None,'PP',h['PS']['prompt_denominator'])
        add('active_historical_P_mean_new_NLL_increase_max_nats',h['PS']['new_nll_delta_mean'],'PROMPT_NLL_NATS',h['PS']['prompt_denominator'])
        add('wiki128_mean_NLL_increase_max_nats',fmean(r['nll'] for r in docs[state]['wiki']['rows'])-fmean(r['nll'] for r in docs['N4']['wiki']['rows']),'NATS_PER_TOKEN_THEN_SEQUENCE_MEAN',128)
        add('mmlu32_warning_net_accuracy_loss_count_at_least',docs['N4']['mmlu']['correct']-docs[state]['mmlu']['correct'],'INTEGER_CORRECT_COUNT',32,inclusive=False)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--panel-lock',type=Path,required=True);p.add_argument('--contract',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    core,receipt=reduce(a.root,a.out,a.dataset,a.panel_lock,a.contract)
    print(json.dumps({'status':receipt['status'],'core_rows':len([r for r in core if r['state'] in ARMS]),'prompt_pairs':receipt['raw_prompt_pairs'],'table_sha256':file_sha(a.out/'first-core-table.csv')},sort_keys=True))


if __name__=='__main__':
    main()
