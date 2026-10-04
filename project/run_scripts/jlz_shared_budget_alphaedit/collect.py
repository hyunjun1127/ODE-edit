"""Model-free independent raw-row validation, partial coverage and paired tables."""
import argparse,csv,json,math,os,subprocess,traceback
from pathlib import Path
from .reducer import reduce_rows
from .common import *
from .telemetry import validate_relations
CHAINS=('V12_ALPHAEDIT',)

def read_rows(folder,expected,state_expected=None):
    paths=sorted(folder.glob('chunk-*.json')) or ([folder/'rows.json'] if (folder/'rows.json').exists() else [])
    rows=[]
    for path in paths:
        payload=json.loads(path.read_text())
        if state_expected is not None and 'state' in payload:require(payload['state']==state_expected,'RAW_ENDPOINT_STATE')
        rows.extend(payload['rows'])
    require(len(rows)==expected*13,'RAW_ROW_COVERAGE:'+str(folder))
    reduce_rows(rows)
    return rows

def validate_identity(rows,identities):
    for row in rows:
        expected=identities[row['identity']]
        for key in ('case_id','kind','prompt_index','new_token_identity','true_token_identity','new_token_count','true_token_count'):
            require(row[key]==expected[key],'RAW_IDENTITY:'+key)

def success(r):return r['true_nll']<r['new_nll'] if r['kind']=='N' else r['new_nll']<r['true_nll']

def paired(before,after):
    a={r['identity']:r for r in before};b={r['identity']:r for r in after}
    require(len(a)==len(before) and len(b)==len(after),'PAIRED_DUPLICATE')
    require(set(a)<=set(b),'PAIRED_MISSING')
    result={}
    for kind in ('R','P','N'):
        ids=[k for k,r in a.items() if r['kind']==kind]
        if not ids:continue
        old=[success(a[k]) for k in ids];new=[success(b[k]) for k in ids]
        result[kind]=dict(denominator=len(ids),before=sum(old),after=sum(new),
            lost=sum(x and not y for x,y in zip(old,new)),gained=sum(not x and y for x,y in zip(old,new)))
    return result

def describe(rows):
    summary=reduce_rows(rows)
    for kind,value in summary.items():
        group=[r for r in rows if r['kind']==kind];desired='true' if kind=='N' else 'new'
        vals=sorted(r[desired+'_nll'] for r in group)
        margins=sorted(r['true_nll']-r['new_nll'] for r in group)
        value.update(desired_NLL_mean=sum(vals)/len(vals),desired_NLL_tail={str(q):vals[round((len(vals)-1)*q)] for q in (.5,.9,.99)},
                     margin_quantiles={str(q):margins[round((len(margins)-1)*q)] for q in (.01,.5,.99)},
                     strict_ACC=value['strict_numerator']/value['strict_denominator'])
    return summary

def reduce_chain(folder,config,identities,w0,source=None):
    commits=[];current_rows=[];milestones={};prepost={};problems=[]
    packing={r['batch']:r for r in config['packing'] if r['phase']=='main'}
    for number in range(1,21):
        root=folder/f'batch-{number:02d}'
        if not (root/'commit.json').exists():continue
        commit=json.loads((root/'commit.json').read_text())
        require(commit['batch']==number and commit['ids']==packing[number]['ids'] and commit['native_pack']==packing[number]['identity'],'COMMIT_INPUT')
        require(commit['config']==digest(config) and commit['history_appends']==5 and commit['observer_no_mutation'] and not commit['checkpoint_saved'],'COMMIT_CONTRACT')
        if source is not None:require(commit['source']==source,'COMMIT_SOURCE')
        if commits:
            require(number==commits[-1]['batch']+1 and commits[-1]['after']==commit['before'],'CHAIN_JOIN')
            require(commits[-1]['RNG_after']==commit['RNG_before'] and commits[-1]['context_hash']==commit['context_hash'],'RNG_CONTEXT_JOIN')
        else:require(number==1,'MISSING_FIRST_COMMIT')
        pre=read_rows(root/'pre',100,commit['before']);post=read_rows(root/'post',100*number if number in MILESTONES else 100,commit['after'])
        validate_identity(pre,identities);validate_identity(post,identities)
        for rows,stored in ((pre,commit['pre']),(post,commit['post'])):
            independent=reduce_rows(rows)
            for family,values in independent.items():
                for key,value in values.items():
                    require(math.isclose(value,stored[family][key],rel_tol=1e-12,abs_tol=1e-12),'INDEPENDENT_STORED_AGGREGATE:'+key)
        wanted=set(commit['ids']);cur=[r for r in post if r['case_id'] in wanted]
        require(len(cur)==1300,'CURRENT_FROM_ALLSEEN')
        prepost[number]=paired(pre,cur);current_rows.extend(cur)
        if number in MILESTONES:milestones[number]=post
        fit=commit['fit']
        check=validate_relations(root/'events.jsonl',config['profile']['eligible_layers'],commit['ids'])
        require(check['request_evaluations']==fit['request_evaluations'] and check['request_updates']==fit['request_updates'],'FIT_LEDGER')
        require(fit['terminal_extra_forward']==fit['terminal_extra_backward']==0,'TERMINAL_EXTRA_CALL')
        events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
        history=[e for e in events if e['event']=='history_commit']
        require(len(history)==5 and len({e['layer_id'] for e in history})==5,'HISTORY_EVENT_COVERAGE')
        require(all(e['payload']['append_count']==1 and e['payload']['request_count']==100 for e in history),'HISTORY_ONCE')
        require(sum(e['event']=='write_layer_request' for e in events)==500,'REALIZATION_COVERAGE')
        require(sum(e['event']=='batch_commit' for e in events)==1,'COMMIT_EVENT')
        for row in pre+post:
            require(row['margin_true_minus_new']==row['true_nll']-row['new_nll'],'MARGIN_SIGN')
        operators=[]
        for l in config['profile']['eligible_layers']:
            p=root/f'alphaedit-operator-L{l}.json';operator=json.loads(p.read_text())
            require(operator['L2']==10 and operator['blue'] is False and operator['dtype']=='float32' and operator['covariance_ridge']==0,'ALPHAEDIT_OPERATOR_CONTRACT')
            operators.append(operator)
        commit['alphaedit_operator_observations']=operators
        commits.append(commit)
    result=dict(commits=len(commits),history_appends=sum(c['history_appends'] for c in commits),pre_post=prepost,
        status='COMPLETE' if len(commits)==20 else 'PARTIAL',metrics={},cohorts={},W0_retention={},atwrite_to_later={})
    ordered=[i for row in config['packing'] if row['phase']=='main' for i in row['ids']]
    for number,rows in milestones.items():
        result['metrics'][number]=describe(rows)
        past=[r for r in current_rows if r['case_id'] in set(ordered[:100*number])]
        result['atwrite_to_later'][number]=paired(past,rows)
        if w0:result['W0_retention'][number]=paired([r for r in w0 if r['case_id'] in set(ordered[:100*number])],rows)
        cohorts={}
        for birth in range(1,number+1):
            ids=set(ordered[(birth-1)*100:birth*100]);end=[r for r in rows if r['case_id'] in ids]
            old=[r for r in past if r['case_id'] in ids]
            cohorts[birth]=dict(metrics=describe(end),paired=paired(old,end))
        result['cohorts'][number]=cohorts
        result.setdefault('prefixes',{})[number]={n:describe([r for r in rows if r['case_id'] in set(ordered[:n])]) for n in (100,500,1000,1500) if n<=number*100}
        result.setdefault('active',{})[number]={label:describe([r for r in rows if bool(r['active_at_endpoint'])==flag])
            for label,flag in [('active',True),('superseded',False)] if any(bool(r['active_at_endpoint'])==flag for r in rows)}
    probe=folder/'batch-01/same-plan-ridge-probe'
    if (probe/'restore.json').exists():
        restore=json.loads((probe/'restore.json').read_text());require(restore['status']=='EXACT_RAM_RESTORED','PROBE_RESTORE_RECEIPT')
        reference=read_rows(probe/'post',100);validate_identity(reference,identities)
        alpha=read_rows(folder/'batch-01/post',100)
        result['same_plan_probe']=dict(restore=restore,ridge=describe(reference),alphaedit=describe(alpha),
            paired_ridge_to_alphaedit=paired(reference,alpha),new_plan_fit=0,not_main_denominator=True)
    elif (probe/'unavailable.json').exists():result['same_plan_probe']=json.loads((probe/'unavailable.json').read_text())
    else:result['same_plan_probe']={'status':'NOT_RECORDED'}
    result['candidate_total']=sum(c['fit']['request_evaluations'] for c in commits)
    result['update_total']=sum(c['fit']['request_updates'] for c in commits)
    result['main_seconds']=sum(c['seconds'] for c in commits)
    if (folder/'terminal.json').exists():
        result['terminal']=json.loads((folder/'terminal.json').read_text())
        if result['terminal']['status']!='COMPLETED':result['status']='PARTIAL'
    else:result['status']='PARTIAL'
    return result

def collect(attempt):
    config=json.loads((attempt/'config.json').read_text());out=attempt/'report';result={};errors={}
    lock=json.loads((attempt/'execution.lock.json').read_text())
    identities={r['identity']:r for r in json.loads(verify(config['observer_identity']).read_text())['rows']}
    w0=[]
    if config['W0']['mode']=='EXACT_STATIC_W0_REUSE':
        w0=read_rows(Path(config['W0']['folder']),2000);validate_identity(w0,identities)
    elif (attempt/'shared-W0/binding.json').exists():
        w0=read_rows(attempt/'shared-W0',2000);validate_identity(w0,identities)
    for chain in CHAINS:
        try:result[chain]=reduce_chain(attempt/('main-'+chain),config,identities,w0,lock['source_commit'])
        except Exception as exc:
            errors[chain]=dict(error=str(exc),traceback=traceback.format_exc());result[chain]=dict(status='TECHNICAL_REDUCTION_FAILED')
    status='COMPLETED' if len(w0)==26000 and not errors and all(r['status']=='COMPLETE' for r in result.values()) else 'PARTIAL'
    write(out/'summary.json',dict(status=status,chains=result,errors=errors,raw_CPU_reduction=True,new_model_calls=0))
    table=[]
    for chain,r in result.items():
        for kind,metric in r.get('metrics',{}).get(20,{}).items():
            table.append(dict(chain=chain,endpoint='W20',kind=kind,**metric))
    jobs=[]
    for name in ('pilot',*CHAINS):
        path=attempt/('submitted-'+name+'.json')
        if path.exists():
            job=json.loads(path.read_text())['job'];require(str(job).isdigit(),'ACCOUNTING_EXACT_ID');jobs.append(str(job))
    accounting=dict(status='NOT_AVAILABLE',job_ids=jobs)
    if jobs:
        try:
            query=subprocess.run(['sacct','-n','-P','-X','-j',','.join(jobs),
                '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES,AllocCPUS,NodeList'],capture_output=True,text=True,timeout=30)
            accounting.update(status='RECORDED' if query.returncode==0 and query.stdout.strip() else 'NOT_AVAILABLE',
                returncode=query.returncode,rows=query.stdout.strip().splitlines(),stderr=query.stderr[:1000],
                scope='exact registered parent accounting once by sealed CPUcollector; not agent monitoring')
        except (OSError,subprocess.TimeoutExpired) as exc:accounting['error']=str(exc)
    write(out/'allocation-cost.json',accounting)
    write(out/'baseline-availability.json',dict(status='NOT_AVAILABLE',reason='No GH matched first2k baseline comparison receipt bound; no new baseline runs',families=['AlphaEdit','AlphaEdit-BLUE','CAKE','MEMIT-H']))
    out.mkdir(parents=True,exist_ok=True)
    fields=['chain','endpoint','kind','numerator','denominator','rate','strict_numerator','strict_denominator','strict_ACC','token_micro','prompt_macro']
    with (out/'comparison-W20.csv').open('x',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(table)
    text='# JLZ v12 2000 edit 실행 결과\n\n상태: '+status+'\n\n'
    for chain,r in result.items():
        text+=f"- {chain}: {r['status']}, commit {r.get('commits','NOT_MEASURED')}/20.\n"
    text+='\n없는 endpoint/값은 NOT_MEASURED이며 0점이나 전체 완료로 처리하지 않았다. raw와 source는 local에 보존한다. 새로운 모델 평가 및 checkpoint 저장은 없다.\n'
    with (out/'report-ko.md').open('x') as f:f.write(text)
    inventory=[member(p) for p in sorted(attempt.rglob('*')) if p.is_file() and p.suffix in ('.json','.jsonl','.csv') and 'source' not in p.relative_to(attempt).parts and p.name!='inventory.json']
    write(out/'inventory.json',dict(files=inventory,no_tensor_bundle=True,source_lock=member(attempt/'execution.lock.json')))
    write(out/'terminal.json',dict(status=status,source=os.environ.get('ODEEDIT_SOURCE_COMMIT'),
          report=member(out/'report-ko.md'),inventory=member(out/'inventory.json'),errors=errors))
    return status

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);args=p.parse_args();print(collect(args.attempt))

if __name__=='__main__':main()
