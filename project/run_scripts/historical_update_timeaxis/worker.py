"""Persistent family worker. Both-family atomic gates, whole-state scoring, no submit."""
import argparse
import fcntl
import math
import subprocess
import time
import traceback
from .common import *
from .backend import Backend

def compare(a,b):
    assert len(a)==len(b)
    n=max((abs(x['nll']-y['nll']) for x,y in zip(a,b,strict=True)),default=0.)
    assert n<=2.5e-4,('NLL_FIDELITY',n)
    return n

class Run:
    def __init__(self,lock,family):
        self.lockpath=Path(lock);self.lock=read(lock);self.root=Path(self.lock['output']);self.family=family;self.out=self.root/family;self.out.mkdir(parents=True,exist_ok=True)
        self.binding=read(self.lock['T0']['path']);assert sha(self.lock['T0']['path'])==self.lock['T0']['sha256']
        self.identity={k:self.lock[k] for k in ('instruction_id','attempt','source_sha256','contract_sha256','T0_sha256','token_sha256')}
        assert self.identity['instruction_id']==INSTRUCTION
        for m in self.lock['source_members']:assert sha(m['path'])==m['sha256']
        assert self.binding['status']=='PASS' and self.binding['GPU_calls']==0
        self.stage='INIT';self.b=None;self.start=time.monotonic();self.facts=rows(DESIGN/'fact-ledger.csv')
        self.records=read(DATA/'counterfact.json');self.bycase={r['case_id']:r for r in self.records}
        self.tokens={(r['case_id'],r['panel'],r['prompt_index']):r for r in read(self.binding['token_manifest']['path'])}
        self.states={r['state_id']:r for r in rows(DESIGN/'state-bank.csv')};self.tasks=rows(DESIGN/'score-tasks.csv');self.evaluator_signature=digest(self.lock['evaluator_signature'])
        self.cachehits=0;self.cachemiss=0

    def gate(self,stage,detail):
        old=self.out/stage/'PASS.json'
        if old.exists():
            x=read(old);assert x['status']=='PASS' and x['identity']==self.identity;return record(old)
        return save(self.out/stage/'PASS.json',dict(status='PASS',stage=stage,identity=self.identity,detail=detail,job_id=os.environ.get('SLURM_JOB_ID'),elapsed_seconds=time.monotonic()-self.start))

    def barrier(self,stage):
        """In-program DAG join; never depends on chat agent or submits jobs."""
        begin=time.monotonic()
        while True:
            for f in FAMILIES:
                if (self.root/f/'FAILURE.json').exists():raise RuntimeError(('PEER_TECHNICAL_FAILURE',f,stage))
            ps=[self.root/f/stage/'PASS.json' for f in FAMILIES]
            if all(p.exists() for p in ps):
                for p in ps:
                    x=read(p);assert x['status']=='PASS' and x['identity']==self.identity
                return
            mapping=self.root.parent/'submission.json'
            if mapping.exists():
                ids=read(mapping)['family_jobs'];peer=next(f for f in FAMILIES if f!=self.family)
                response=subprocess.run(['sacct','-n','-X','-j',str(ids[peer]),'--format=State','--parsable2'],text=True,capture_output=True,timeout=20)
                if response.returncode==0:
                    states=response.stdout.split()
                    assert not any(s.split('|')[0].split('+')[0] in ('FAILED','CANCELLED','OUT_OF_MEMORY','TIMEOUT','NODE_FAIL','PREEMPTED','BOOT_FAIL') for s in states),('PEER_TERMINAL_FAILURE',peer,states)
            assert time.monotonic()-begin<self.lock['barrier_timeout_seconds'],('DAG_BARRIER_TIMEOUT',stage)
            time.sleep(15)

    def pairs(self,ids):
        pp=self.b.obs['evaluator'].counterfact_pairs([self.bycase[i] for i in ids])
        return {k:pp[k] for k in ('rewrite_target_new','rewrite_target_true','rephrase_target_new','rephrase_target_true')}

    def raw(self,ids,mb=16):return {k:self.b.evaluate(v,mb) for k,v in self.pairs(ids).items()}

    def check_raw(self,left,right):
        maxn=0.;maxm=0.;boundary=[]
        for k in left:maxn=max(maxn,compare(left[k],right[k]))
        for kind in ('rewrite','rephrase'):
            for n,t,n2,t2 in zip(left[kind+'_target_new'],left[kind+'_target_true'],right[kind+'_target_new'],right[kind+'_target_true'],strict=True):
                m=t['nll']-n['nll'];m2=t2['nll']-n2['nll'];maxm=max(maxm,abs(m-m2))
                if (m>0)!=(m2>0):
                    assert abs(m)<=5e-4 and abs(m2)<=5e-4,('NONBOUNDARY_FLIP',m,m2)
                    boundary.append(dict(case_id=n['case_id'],kind=kind,prompt_index=n['prompt_index'],left=m,right=m2))
        assert maxm<=5e-4,('MARGIN_FIDELITY',maxm)
        return dict(max_nll=maxn,max_margin=maxm,boundary_flips=boundary)

    def verify_endpoint(self,t):
        dest=self.out/'fidelity'/f'endpoint-{t:03d}.json'
        if dest.exists():return read(dest)
        fs=[r for r in rows(DESIGN/'fidelity-panel.csv') if int(r['birth_batch'])<=max(t,1)]
        ids=[int(r['case_id']) for r in sorted(fs,key=lambda r:r['rank_sha256'])[:16]]
        recipe=dict(kind='ACTUAL',actual_checkpoint=t)
        with self.b.state(recipe) as state:
            a=self.raw(ids);repeat=self.raw(ids);single=self.raw(ids,1)
            checks=[self.check_raw(a,repeat),self.check_raw(a,single)]
            original=[]
            if t:
                cell=1 if self.family==FAMILIES[0] else 2;p=BASE/f'main-cell-{cell}'/f'B{t:03d}'/'seen-full.json'
                assert sha(p)==self.binding['original_scores'][self.family][str(t)]['sha256']
                old=read(p)
                for kind,tag in [('rewrite','RS'),('rephrase','PS')]:
                    prior={(r['case_id'],r['prompt_index']):r for r in old['metrics'][tag]['rows']}
                    for n,tr in zip(a[kind+'_target_new'],a[kind+'_target_true'],strict=True):
                        o=prior[n['case_id'],n['prompt_index']];token=self.tokens[n['case_id'],'rewrite' if kind=='rewrite' else 'paraphrase',n['prompt_index']]
                        assert o['identity']==token['pair_identity'],'ORIGINAL_PAIR_IDENTITY'
                        dn=abs(n['nll']-o['new_nll']);dt=abs(tr['nll']-o['true_nll']);dm=abs(tr['nll']-n['nll']-o['margin'])
                        assert max(dn,dt)<=2.5e-4 and dm<=5e-4,('HISTORICAL_FIDELITY',t,n['case_id'],dn,dt,dm)
                        m=tr['nll']-n['nll'];flip=(m>0)!=o['success']
                        assert not flip or (abs(m)<=5e-4 and abs(o['margin'])<=5e-4),'ORIGINAL_NONBOUNDARY_FLIP'
                        original.append(dict(case_id=n['case_id'],panel=kind,prompt_index=n['prompt_index'],new_error=dn,true_error=dt,margin_error=dm,boundary_flip=flip,actual_new=n['nll'],actual_true=tr['nll'],original_new=o['new_nll'],original_true=o['true_nll'],original_identity=o['identity']))
                del old
        with self.b.state(recipe) as restored:after=self.raw(ids)
        checks.append(self.check_raw(a,after))
        result=dict(t=t,ids=ids,state=state,restored=restored,checks=checks,original=original,raw=a,repeat_raw=repeat,mb1_raw=single,restored_raw=after,original_status='NOT_AVAILABLE_W0' if not t else 'PASS',cost=self.b.cost())
        save(dest,result);print('ENDPOINT_FIDELITY_PASS',self.family,t,flush=True);return result

    def t1(self):
        self.stage='T1';self.b=Backend(self.binding,self.family)
        if (self.out/'T1/PASS.json').exists():
            self.gate('T1',{});self.barrier('T1');return
        eps=[self.verify_endpoint(t) for t in (0,1,20,50,100)];diagonals=[]
        ids=[int(r['case_id']) for r in rows(DESIGN/'fidelity-panel.csv')[:8]]
        for i in range(12):
            a,b=TIMES[i:i+2];recipe=dict(kind='ACTUAL',actual_checkpoint=a)
            expected=self.b.endpoint(a)
            with self.b.state(recipe,force_removal=(b,[i])) as arithmetic:
                assert arithmetic['weight_hashes']=={k:tensor_sha(v) for k,v in expected.items()},('DIAGONAL_BYTES',a,b)
                first=self.raw(ids)
            with self.b.state(recipe) as direct:second=self.raw(ids)
            check=self.check_raw(first,second);diagonals.append(dict(start=a,anchor=b,arithmetic=arithmetic,direct=direct,check=check))
            del expected
        self.gate('T1',dict(endpoints=[0,1,20,50,100],diagonals=diagonals,cost=self.b.cost()));self.barrier('T1')

    def cached_raw(self,ids,state):
        raw={};receipts=[]
        for kind,ps in self.pairs(ids).items():
            raw[kind]=[]
            for offset in range(0,len(ps),16):
                group=ps[offset:offset+16]
                layout=[dict(case_id=p.case_id,kind=p.kind,prompt_index=p.prompt_index,encoded=self.b.obs['evaluator']._encode_pair(self.b.tok,p)) for p in group]
                key=digest([state['state_weight_hash'],layout,self.evaluator_signature]);directory=self.root/'score-cache'/key[:2];directory.mkdir(parents=True,exist_ok=True)
                path=directory/(key+'.json')
                with (directory/(key+'.lock')).open('a') as guard:
                    fcntl.flock(guard,fcntl.LOCK_EX)
                    if path.exists():
                        x=read(path);assert x['key']==key and x['state_weight_hash']==state['state_weight_hash'] and x['layout_sha256']==digest(layout) and x['rows_sha256']==digest(x['rows']);self.cachehits+=1
                    else:
                        rr=self.b.evaluate(group)
                        # raw prompt strings remain local only; exact original evaluator result.
                        x=dict(key=key,state_weight_hash=state['state_weight_hash'],layout_sha256=digest(layout),evaluator_signature=self.evaluator_signature,rows=rr,rows_sha256=digest(rr))
                        save(path,x);self.cachemiss+=1
                    raw[kind].extend(x['rows']);receipts.append(dict(path=str(path),sha256=sha(path),key=key,layout_sha256=x['layout_sha256']))
        return raw,receipts

    def score_task(self,task):
        dest=self.out/'tasks'/task['task_id'];receipt=dest/'PASS.json'
        if receipt.exists():
            x=read(receipt);assert x['identity']==self.identity and sha(x['scores']['path'])==x['scores']['sha256'];return
        sid=task['state_id'];recipe=self.states[sid]
        if recipe['kind']=='ACTUAL':self.verify_endpoint(int(recipe['actual_checkpoint']))
        ids=[int(f['case_id']) for f in self.facts if f['cohort_id']==task['cohort_id'] and (task['case_selector']=='whole_cohort' or boolstr(f['pilot_selected']))]
        assert historical_digest(ids)==task['case_order_sha256']
        sentinel_ids=sorted(ids,key=lambda i:digest(['sentinel',sid,i]))[:8]
        sent_path=self.out/'sentinels'/(sid+'.json')
        with self.b.state(recipe) as state:
            raw,cache=self.cached_raw(ids,state);checks=None
            if recipe['kind']=='COUNTERFACTUAL' and not sent_path.exists():
                first=self.raw(sentinel_ids);second=self.raw(sentinel_ids);checks=self.check_raw(first,second)
            score=[]
            for kind in ('rewrite','rephrase'):
                for n,t in zip(raw[kind+'_target_new'],raw[kind+'_target_true'],strict=True):
                    panel='rewrite' if kind=='rewrite' else 'paraphrase';token=self.tokens[n['case_id'],panel,n['prompt_index']]
                    assert n['target_token_ids']==token['target_ids'] and t['target_token_ids']==token['competitor_ids']
                    score.append(dict(schema_version=1,row_type='score',case_id=n['case_id'],panel=panel,prompt_index=n['prompt_index'],
                        target_version=token['target_version'],prompt_token_hash=token['prompt_token_hash'],target_token_hash=token['target_token_hash'],competitor_token_hash=token['competitor_token_hash'],
                        evaluator_signature=self.evaluator_signature,valid=True,missing_reason=None,state_id=sid,state_weight_hash=state['state_weight_hash'],
                        target_nll=n['nll'],competitor_nll=t['nll'],margin=t['nll']-n['nll'],target_token_count=len(n['token_correct']),competitor_token_count=len(t['token_correct']),
                        target_strict_tf=n['all_tokens_correct'],competitor_strict_tf=t['all_tokens_correct'],target_token_correct=sum(n['token_correct']),competitor_token_correct=sum(t['token_correct']),
                        target_predictions=n['token_predictions'],competitor_predictions=t['token_predictions'],execution_receipt=str(receipt)))
            assert len(score)==int(task['prompt_count']) and len({(r['case_id'],r['panel'],r['prompt_index']) for r in score})==len(score)
        if checks is not None:
            with self.b.state(recipe) as restored:third=self.raw(sentinel_ids)
            restorecheck=self.check_raw(first,third);save(sent_path,dict(state=state,restore_state=restored,ids=sentinel_ids,repeat=checks,restore=restorecheck))
        sr=save(dest/'scores.json',score)
        save(receipt,dict(status='PASS',identity=self.identity,task=task,state=state,scores=sr,cache=cache,cost=self.b.cost()))
        print('SCORE_TASK_PASS',self.family,task['task_id'],sid,len(score),flush=True)

    def phase(self,stage,flag):
        self.stage=stage
        before=self.b.cost()
        tasks=[t for t in self.tasks if boolstr(t[flag]) and self.states[t['state_id']]['family'] in (self.family,'SHARED')]
        # Common W0 rows share exact cache keys; sidecar task receipts remain family-specific.
        for t in tasks:self.score_task(t)
        after=self.b.cost();forecast=None
        if stage=='T2P':
            total_padded=0
            for task in self.tasks:
                if self.states[task['state_id']]['family'] not in (self.family,'SHARED') or not (boolstr(task['main']) or boolstr(task['pairs'])):continue
                ids=[int(f['case_id']) for f in self.facts if f['cohort_id']==task['cohort_id']]
                for pp in self.pairs(ids).values():
                    lengths=[sum(map(len,self.b.obs['evaluator']._encode_pair(self.b.tok,p)))-1 for p in pp]
                    total_padded+=sum(max(lengths[i:i+16])*len(lengths[i:i+16]) for i in range(0,len(lengths),16))
            padded=after['padded_tokens']-before['padded_tokens'];seconds=after['forward_seconds']-before['forward_seconds']
            forecast=dict(pilot_measured_padded_tokens=padded,pilot_measured_forward_seconds=seconds,planned_full_pair_padded_tokens_upper=total_padded,
                estimated_forward_seconds_upper=seconds/padded*total_padded if padded else None,excludes='load/materialization/hash/restore/cache/observer I-O; reuse reduces unique computation',not_hardcap=True)
        self.gate(stage,dict(task_count=len(tasks),task_ids=[t['task_id'] for t in tasks],cache_hits=self.cachehits,cache_misses=self.cachemiss,cost=after,pilot_forecast=forecast))
        self.barrier(stage)

    def run(self):
        try:
            if (self.out/'terminal.json').exists():
                x=read(self.out/'terminal.json');assert x['identity']==self.identity and x['status']=='COMPLETED';return
            self.t1();self.phase('T2P','pilot');self.phase('T2F','main');self.phase('T3B','pairs')
            self.b.copy(self.b.w0);assert self.b.hashes()==self.binding['w0_selected'];self.b.unchanged()
            save(self.out/'terminal.json',dict(status='COMPLETED',identity=self.identity,cost=self.b.cost(),job_id=os.environ.get('SLURM_JOB_ID')))
        except BaseException as e:
            save(self.out/'FAILURE.json',dict(status='TECHNICAL_FAILED',identity=self.identity,stage=self.stage,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),cost=None if self.b is None else self.b.cost(),job_id=os.environ.get('SLURM_JOB_ID')))
            raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--family',choices=FAMILIES,required=True);a=p.parse_args();Run(a.lock,a.family).run()
