"""CPU-only raw NLL reducer. No imports of model, evaluator or runtime."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ARMS=('N4','EN_KL_Q','DEC_LINE','DEC_MODES_CUM')


def identity(row):
    return (row['case_id'],row['prompt_index'],row['prompt'],row['target'],
            tuple(row['target_token_ids']))


def reduce_raw(observation):
    count=observation['requests'];result={};paired={}
    order=None
    for tag,prefix,multiple in (('RS','rewrite',1),('PS','rephrase',2),('NS','locality',10)):
        new=observation['raw'][prefix+'_target_new']
        old=observation['raw'][prefix+'_target_true']
        if len(new)!=count*multiple or len(old)!=len(new):raise ValueError('RAW_CARDINALITY')
        if len(set(identity(row) for row in new))!=len(new):raise ValueError('DUPLICATE_RAW_ID')
        cases=[r['case_id'] for r in new[::multiple]]
        if len(set(cases))!=count:raise ValueError('DUPLICATE_CASE')
        if order is not None and order!=cases:raise ValueError('PANEL_ORDER')
        order=cases;rows=[]
        for index,(n,o) in enumerate(zip(new,old)):
            expected=(cases[index//multiple],index%multiple)
            if ((n['case_id'],n['prompt_index'])!=expected or
                (o['case_id'],o['prompt_index'])!=expected or n['prompt']!=o['prompt']):
                raise ValueError('PAIR_ID_ORDER')
            for r in (n,o):
                if not math.isfinite(r['nll']):raise ValueError('NONFINITE_NLL')
                if len(r['token_correct'])!=len(r['target_token_ids']):raise ValueError('TOKEN_CARDINALITY')
                if any(type(x) is not bool for x in r['token_correct']):raise ValueError('TOKEN_BOOL')
                if r['all_tokens_correct']!=all(r['token_correct']):raise ValueError('TOKEN_STRICT')
            success=o['nll']<n['nll'] if tag=='NS' else n['nll']<o['nll']
            pair_id=hashlib.sha256(json.dumps([identity(n),identity(o)],ensure_ascii=False).encode()).hexdigest()
            rows.append(dict(case_id=n['case_id'],prompt_index=n['prompt_index'],
                pair_id=pair_id,new_nll=n['nll'],true_nll=o['nll'],success=success,
                tie=n['nll']==o['nll'],new_strict=all(n['token_correct']),
                desired_margin=(n['nll']-o['nll']) if tag=='NS' else (o['nll']-n['nll'])))
        num=sum(r['success'] for r in rows)
        if observation['metrics'][tag]['numerator']!=num:raise ValueError('REPORTED_COUNT_MISMATCH')
        result[tag]=dict(count=num,denominator=len(rows),percent=100*num/len(rows),
            ties=sum(r['tie'] for r in rows))
        paired[tag]=rows
    result['strict']={}
    flags={k:[] for k in ('rewrite_strict','two_P_strict','R_two_P_strict','R_two_P_NLL_joint')}
    for i in range(count):
        r=paired['RS'][i];p=paired['PS'][2*i:2*i+2]
        values=(r['new_strict'],all(x['new_strict'] for x in p),
                r['new_strict'] and all(x['new_strict'] for x in p),
                r['success'] and all(x['success'] for x in p))
        for k,v in zip(flags,values):flags[k].append(v)
    for k,v in flags.items():
        result['strict'][k]=sum(v)
        if observation['strict'][k]!=sum(v):raise ValueError('REPORTED_STRICT_MISMATCH')
    return result,paired


def dump(path,value):
    data=json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(data)


def run(output,destination):
    output=Path(output);destination=Path(destination)
    if not (output/'terminal.json').exists() and not (output/'failure.json').exists():
        raise ValueError('TERMINAL_EVIDENCE_REQUIRED_NO_LIVE_REDUCTION')
    destination.mkdir(parents=True,exist_ok=False)
    first=[];allrows=[];reduced={};members=[]
    for arm in ARMS:
        path=output/'B1/observers/current'/f'{arm}.json'
        if not path.exists():
            first.append(dict(arm=arm,status='NOT_RECORDED',RS_count=None,PS_count=None,NS_count=None))
            continue
        data=path.read_bytes();obs=json.loads(data);summary,rows=reduce_raw(obs)
        if obs['requests']!=100:raise ValueError('EXACT_B1_100')
        reduced[arm]=summary
        members.append(dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
            endpoint=obs['selection_seal']['endpoint_weight_sha256'],request_order=obs['request_order']))
        first.append(dict(arm=arm,status='RAW_NLL_REDUCED',
            **{k+'_count':summary[k]['count'] for k in ('RS','PS','NS')}))
        for metric,group in rows.items():
            allrows.extend(dict(arm=arm,metric=metric,**r) for r in group)
    fields=('arm','status','RS_count','PS_count','NS_count')
    with (destination/'first-table.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(first)
    if allrows:
        with (destination/'paired-nll.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    dump(destination/'independent-reducer.json',dict(results=reduced,inputs=members,
        scope='RAW_TRUE_NEW_NLL_AND_TF_TOKEN_FLAGS; NOT_GPU_REPLAY',
        missing_not_imputed=True,new_GPU=0))
    return first


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--destination',required=True)
    a=p.parse_args();print(json.dumps(run(a.output,a.destination)))
