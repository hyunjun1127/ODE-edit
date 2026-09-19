"""Terminal B1 CPU audit: raw NLL pairing, gate arithmetic, state and cost.

No model/runtime imports. Scientific raw is read-only; outputs are create-once.
This does not establish FD parity, tensor reconstruction or GPU continuation.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from .review_b1 import ARMS, reduce_raw, dump


def read(path):return json.loads(Path(path).read_text())


def member(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())


def csv_write(path,rows):
    if not rows:return
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def quantile(values,q):
    values=sorted(values)
    if not values:return None
    i=(len(values)-1)*q;j=math.floor(i);k=math.ceil(i)
    return values[j]+(values[k]-values[j])*(i-j)


def paired(base,candidate):
    if [r['pair_id'] for r in base]!=[r['pair_id'] for r in candidate]:
        raise ValueError('CROSS_ENDPOINT_PAIR_IDENTITY')
    lost=[a['pair_id'] for a,b in zip(base,candidate) if a['success'] and not b['success']]
    gained=[a['pair_id'] for a,b in zip(base,candidate) if not a['success'] and b['success']]
    change=[b['desired_margin']-a['desired_margin'] for a,b in zip(base,candidate)]
    return dict(denominator=len(base),lost=len(lost),gained=len(gained),net=len(gained)-len(lost),
        lost_ids=lost,gained_ids=gained,margin_mean_change=sum(change)/len(change),
        margin_change_p01=quantile(change,.01),margin_change_p05=quantile(change,.05),
        margin_change_p50=quantile(change,.5),margin_change_p95=quantile(change,.95),
        margin_change_p99=quantile(change,.99),margin_worsened=sum(x<0 for x in change))


def independent_gate(native,primary,selection):
    phi=selection.get('phi_reference_native')
    gain=selection.get('phi_reference_gain')
    checks=dict(nonzero_valid_correction=selection.get('accepted') is True and selection.get('actual_delta_norm',0)>0,
        current=selection.get('current_pass') is True,full512=selection.get('full512') is True,
        choice_or_margin=selection.get('repaired_choices',0)>0 or
            (isinstance(phi,(int,float)) and phi>0 and isinstance(gain,(int,float)) and
             gain>=.05*phi and gain>selection.get('tau_risk',math.inf)),
        PS_point_nonloss=primary['PS']['count']>=native['PS']['count'],
        strict_joint_point_nonloss=all(primary['strict'][k]>=native['strict'][k] for k in
             ('rewrite_strict','two_P_strict','R_two_P_strict','R_two_P_NLL_joint')))
    return dict(checks=checks,pass_=all(checks.values()),sequential_authorized=False)


def run(output,destination,accounting):
    out=Path(output);dest=Path(destination)
    terminal=read(out/'terminal.json')
    if terminal['status']!='B1_COMPLETE_USER_LIMIT' or terminal['maximum_batch']!=1:
        raise ValueError('EXACT_COMPLETED_B1_REQUIRED')
    if terminal['sequential_authorized'] is not False or terminal['checkpoint_saved'] is not False:
        raise ValueError('B1_NOCP_AUTHORITY')
    dest.mkdir(parents=True,exist_ok=False)
    root=out/'B1';obsroot=root/'observers/current'
    observations={};summaries={};rows={};inputs=[]
    for arm in (*ARMS,'ENTRY','OWN_NATIVE','W0'):
        path=obsroot/f'{arm}.json';observations[arm]=read(path)
        summaries[arm],rows[arm]=reduce_raw(observations[arm]);inputs.append(member(path))
        if observations[arm]['requests']!=100:raise ValueError('B1_EXACT_100_REQUESTS')
        for metric,multiple in (('RS',1),('PS',2),('NS',10)):
            if observations[arm]['metrics'][metric]['denominator']!=100*multiple:
                raise ValueError('REPORTED_DENOMINATOR_MISMATCH')
    final=[];selection_rows=[];commit_rows=[];trial_rows=[]
    for arm in ARMS:
        s=read(root/'arms'/arm/'selection-ledger.json');cp=read(root/'commits'/arm/'commit.json')
        endpoint=observations[arm]['selection_seal']['endpoint_weight_sha256']
        # Controller: SHA(shape|dtype|bytes); observer/commit: SHA(bytes).
        # No full endpoint survives the user noCP policy: do not equate the
        # different protocols or claim independent tensor reconstruction.
        if observations[arm]['selection_seal']['selection_ledger_sha256']!=member(root/'arms'/arm/'selection-ledger.json')['sha256']:
            raise ValueError('EVALUATION_SELECTION_LEDGER')
        if cp['history_appends']!=1 or cp['candidate_observer_appends']!=0 or cp['checkpoint'] is not None:
            raise ValueError('HISTORY_ONCE_NOCP')
        if s['history_appends_in_controller']!=0 or s['official_observer_accesses']!=0:
            raise ValueError('CONTROLLER_OBSERVER_HISTORY_BOUNDARY')
        if cp['selection']['sha256']!=member(root/'arms'/arm/'selection-ledger.json')['sha256']:
            raise ValueError('COMMIT_SELECTION_IDENTITY')
        histories=cp['history'];entry=read(root/'batch-entry.json')['identity']
        if len(histories)!=1:raise ValueError('HISTORY_RECEIPT_CARDINALITY')
        h=histories[0]
        if (cp['identity']['W']!=endpoint or h['weight_sha256']!=endpoint or h['layer']!=4 or
            h['history_append']!=1 or h['compute_ks']!=1 or h.get('compute_z',0)!=0 or h.get('solve',0)!=0 or
            h['before_sha256']!=entry['M'] or h['after_sha256']!=cp['identity']['M'] or cp['identity']['rng']!=entry['rng']):
            raise ValueError('SELECTED_HISTORY_MEMORY_RNG_BINDING')
        final.append(dict(arm=arm,**{k+'_count':summaries[arm][k]['count'] for k in ('RS','PS','NS')},
            RS_denominator=100,PS_denominator=200,NS_denominator=1000,**summaries[arm]['strict']))
        selection_rows.append(dict(arm=arm,accepted=s['accepted'],stop_reason=s['stop_reason'],
            actual_delta_norm=s['actual_delta_norm'],current_pass=s['current_pass'],
            history_pass=s['history_pass'],reference_pass=s['reference_pass'],full512=s['full512'],
            phi_native=s['phi_reference_native'],phi_gain=s['phi_reference_gain'],
            repaired_choices=s['repaired_choices'],trials=len(s['trials']),
            endpoint_sha256=endpoint,controller_header_endpoint_sha256=s['selected_weight_sha256'],
            endpoint_binding_level='LEDGER_COMMIT_OBSERVER_RAW_SHA; NO_TENSOR_RECONSTRUCTION',
            counters=json.dumps(s['counters'],sort_keys=True)))
        commit_rows.append(dict(arm=arm,history_appends=cp['history_appends'],
            history_seconds=cp['history_seconds'],commit_seconds=cp['seconds'],
            disk_checkpoint=cp['disk_checkpoint'],exact_resume=cp['exact_crash_resume']))
        for t in s['trials']:
            trial_rows.append(dict(arm=arm,trial=t['trial'],scale=t.get('scale'),
                accepted=t['accepted'],reason=t['reason'],actual_delta_norm=t.get('actual_delta_norm'),
                candidate_sha256=t.get('candidate_sha256'),phi_reference=t.get('phi_reference'),
                mismatches=t.get('mismatches'),loss=t.get('loss')))
    pairrows=[];pairids={};retention=[];conditioned=[]
    for arm in ARMS:
        for base in ('W0','OWN_NATIVE'):
            for metric in ('RS','PS','NS'):
                result=paired(rows[base][metric],rows[arm][metric]);ids={k:result.pop(k) for k in ('lost_ids','gained_ids')}
                pairrows.append(dict(base=base,arm=arm,metric=metric,**result))
                pairids[f'{base}:{arm}:{metric}']=ids
        original=rows['W0']['NS'];now=rows[arm]['NS']
        if [r['pair_id'] for r in original]!=[r['pair_id'] for r in now]:raise ValueError('W0_RETENTION_ID')
        denominator=sum(r['success'] for r in original)
        retained=sum(a['success'] and b['success'] for a,b in zip(original,now))
        retention.append(dict(arm=arm,W0_correct=denominator,retained=retained,lost=denominator-retained))
        own=rows['OWN_NATIVE']['NS'];entry=rows['ENTRY']['NS']
        groups={'full_N':list(range(len(now))),
            'W0_correct_native_broken':[i for i in range(len(now)) if original[i]['success'] and not own[i]['success']],
            'entry_correct_native_broken':[i for i in range(len(now)) if entry[i]['success'] and not own[i]['success']],
            'W0_native_stable':[i for i in range(len(now)) if original[i]['success'] and own[i]['success']]}
        for name,indices in groups.items():
            conditioned.append(dict(arm=arm,group=name,denominator=len(indices),
                native_correct=sum(own[i]['success'] for i in indices),selected_correct=sum(now[i]['success'] for i in indices),
                recovered=sum(not own[i]['success'] and now[i]['success'] for i in indices),
                additional_lost=sum(own[i]['success'] and not now[i]['success'] for i in indices),
                true_NLL_mean_change=(sum(now[i]['true_nll']-own[i]['true_nll'] for i in indices)/len(indices)) if indices else None,
                new_NLL_mean_change=(sum(now[i]['new_nll']-own[i]['new_nll'] for i in indices)/len(indices)) if indices else None))
    gate=independent_gate(summaries['N4'],summaries['DEC_MODES_CUM'],
                         read(root/'arms/DEC_MODES_CUM/selection-ledger.json'))
    storedgate=read(out/'B1-to-S3.json')
    if gate['checks']!=storedgate['checks'] or gate['pass_']!=storedgate['pass']:
        raise ValueError('INDEPENDENT_B1_GATE_DISAGREEMENT')
    for name,data in [('final-table',final),('selection',selection_rows),('history',commit_rows),
                      ('trials',trial_rows),('paired',pairrows),('W0-N-retention',retention),('N-conditioned',conditioned)]:
        csv_write(dest/(name+'.csv'),data)
    dump(dest/'paired-IDs.json',pairids);dump(dest/'stage-gate.json',gate)
    native=read(root/'native/native-binding.json')
    compute=read(root/'batch-compute.json')
    acct=read(accounting)
    if acct['job_id']!='51058' or acct['state']!='COMPLETED':raise ValueError('EXACT_PARENT_ACCOUNTING')
    nr=native['receipt'];hook=nr['z_hook']
    native_counts={k:v for k,v in nr.items() if isinstance(v,(str,int,float,bool)) or v is None}
    native_counts['z_hook']={k:v for k,v in hook.items() if k!='batches'}
    native_counts['z_hook']['clamp_hits']=sum(sum(b['clamp_hits']) for b in hook['batches'])
    native_counts['z_hook']['stop_reasons']=[reason for b in hook['batches'] for reason in b['stop_reasons']]
    dump(dest/'compute.json',dict(parent_accounting=acct,terminal=terminal,batch=compute,
        native_receipt=native_counts,native_binding=member(root/'native/native-binding.json'),prior_failed_GPU_seconds=1113,
        nested_timers_do_not_sum=True,allocation_is_not_utilization=True))
    dump(dest/'coverage.json',dict(native=read(root/'reference-native.json')['coverage'],
        requests=len(read(root/'batch-entry.json')['case_ids']),arms=list(ARMS),
        history_appends=sum(r['history_appends'] for r in commit_rows),
        full_numerical_validation=terminal['full_numerical_validation'],FD='USER_WAIVED_NOT_PASS',
        GPU_continuation='NOT_TESTED',exact_resume='NOT_AVAILABLE',new_review_GPU=0))
    dump(dest/'raw-inventory.json',dict(members=[member(p) for p in sorted(out.rglob('*')) if p.is_file()],
         raw_read_only=True,scope='THIS_ATTEMPT_ONLY',large_shared_assets='PRIOR_IDENTITY_REUSED'))
    dump(dest/'analysis-manifest.json',dict(code=member(__file__),reducer=member(Path(__file__).with_name('review_b1.py')),
        output=str(out),observations=inputs,independent_NLL_reducer=True,separate_agent_red=False))
    return final


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--accounting',required=True);a=p.parse_args()
    print(json.dumps(run(a.output,a.destination,a.accounting)))
