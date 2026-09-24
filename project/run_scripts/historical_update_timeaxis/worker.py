"""Persistent family worker. Both-family atomic gates, whole-state scoring, no submit."""
import argparse
import fcntl
import math
import subprocess
import time
import traceback
from .common import *
from .backend import Backend
from .fidelity import (compare, compare_raw, historical_row, policy, execution_identity,
                       check_gate, completion, COMPLETE, json_evidence)

class Run:
    def __init__(self,lock,family):
        self.lockpath=Path(lock);self.lock=read(lock);self.root=Path(self.lock['output']);self.family=family;self.out=self.root/family;self.out.mkdir(parents=True,exist_ok=True)
        self.binding=read(self.lock['T0']['path']);assert sha(self.lock['T0']['path'])==self.lock['T0']['sha256']
        self.policy=policy(self.lock);self.identity=execution_identity(self.lock);self.diagnostics={}
        self.reuse=read(self.lock['reuse_bridge']['path'])
        assert sha(self.lock['reuse_bridge']['path'])==self.lock['reuse_bridge']['sha256']
        assert self.reuse['new_policy_sha256']==self.policy['numerical_policy_sha256']
        assert self.identity['instruction_id']==INSTRUCTION
        for m in self.lock['source_members']:assert sha(m['path'])==m['sha256']
        assert self.binding['status']=='PASS' and self.binding['GPU_calls']==0
        self.stage='INIT';self.b=None;self.start=time.monotonic();self.facts=rows(DESIGN/'fact-ledger.csv')
        self.records=read(DATA/'counterfact.json');self.bycase={r['case_id']:r for r in self.records}
        self.tokens={(r['case_id'],r['panel'],r['prompt_index']):r for r in read(self.binding['token_manifest']['path'])}
        self.states={r['state_id']:r for r in rows(DESIGN/'state-bank.csv')};self.tasks=rows(DESIGN/'score-tasks.csv');self.evaluator_signature=digest(self.lock['evaluator_signature'])
        self.cachehits=0;self.cachemiss=0

    def evidence(self,path,value):
        return save(path,dict(**value,**self.policy,identity=self.identity))

    def warnings(self):
        return dict(warning_count=sum(v['warning_count'] for v in self.diagnostics.values()),
                    comparisons=self.diagnostics)

    def diagnostic(self,name,left,right):
        result=compare_raw(left,right)
        self.evidence(self.out/'numerical-diagnostics'/(name+'.json'),dict(comparison=name,detail=result))
        self.diagnostics[name]=dict(warning_count=result['warning_count'],max_nll=result['max_nll'],max_margin=result['max_margin'])
        return result

    def gate(self,stage,detail):
        old=self.out/stage/'PASS.json'
        if old.exists():
            x=read(old);check_gate(x,self.identity);return record(old)
        return self.evidence(self.out/stage/'PASS.json',dict(status='PASS',stage=stage,
            pass_semantics='STRUCTURAL_COMPLETION_NOT_NUMERICAL_CERTIFICATION',detail=detail,
            numerical_diagnostics=self.warnings(),job_id=os.environ.get('SLURM_JOB_ID'),elapsed_seconds=time.monotonic()-self.start))

    def barrier(self,stage):
        """In-program DAG join; never depends on chat agent or submits jobs."""
        begin=time.monotonic()
        while True:
            for f in FAMILIES:
                if (self.root/f/'FAILURE.json').exists():raise RuntimeError(('PEER_TECHNICAL_FAILURE',f,stage))
            ps=[self.root/f/stage/'PASS.json' for f in FAMILIES]
            if all(p.exists() for p in ps):
                for p in ps:
                    check_gate(read(p),self.identity)
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

    def raw(self,ids,mb=16,evidence=None):
        result={}
        for k,v in self.pairs(ids).items():
            sink=None if evidence is None else lambda rr,k=k:self.evidence(
                evidence/(k+'.json'),dict(ids=ids,microbatch=mb,rows=json_evidence(rr),
                                         evidence_semantics='RAW_BEFORE_FINITE_AND_NUMERICAL_DIAGNOSTICS'))
            result[k]=self.b.evaluate(v,mb,evidence_sink=sink)
        return result

    def check_raw(self,left,right):
        return compare_raw(left,right)

    def verify_endpoint(self,t):
        dest=self.out/'fidelity'/f'endpoint-{t:03d}.json'
        if dest.exists():
            result=read(dest);assert result['identity']==self.identity;return result
        prior=self.reuse['endpoints'][self.family].get(str(t))
        if prior:
            assert sha(prior['path'])==prior['sha256'],'REUSE_ENDPOINT_SHA'
            old=read(prior['path'])
            expected=self.binding['w0_selected'] if t==0 else self.binding['checkpoints'][self.family][str(t)]['weights']
            for s in (old['state'],old['restored']):
                assert s['restore']=='EXACT_BYTES' and s['weight_hashes']==expected,'REUSE_STATE_IDENTITY'
            checks=[self.diagnostic(f'endpoint-{t:03d}-{label}',old['raw'],old[field]) for label,field in
                    [('repeat','repeat_raw'),('MB16-MB1','mb1_raw'),('restore','restored_raw')]]
            result=dict(t=t,reused_from=prior,reuse_bridge=self.lock['reuse_bridge'],checks=checks,
                        original_evidence_status='PRIOR_SAVED_RECEIPT_NOT_NEW_GPU_CHECK',new_GPU_evaluations=0)
            self.evidence(dest,result);return result
        fs=[r for r in rows(DESIGN/'fidelity-panel.csv') if int(r['birth_batch'])<=max(t,1)]
        ids=[int(r['case_id']) for r in sorted(fs,key=lambda r:r['rank_sha256'])[:16]]
        recipe=dict(kind='ACTUAL',actual_checkpoint=t)
        prefix=self.out/'diagnostic-raw'/f'endpoint-{t:03d}'
        with self.b.state(recipe) as state:
            a=self.raw(ids,evidence=prefix/'raw');repeat=self.raw(ids,evidence=prefix/'repeat');single=self.raw(ids,1,evidence=prefix/'mb1')
            checks=[self.diagnostic(f'endpoint-{t:03d}-repeat',a,repeat),self.diagnostic(f'endpoint-{t:03d}-MB16-MB1',a,single)]
            original=[]
            if t:
                cell=1 if self.family==FAMILIES[0] else 2;p=BASE/f'main-cell-{cell}'/f'B{t:03d}'/'seen-full.json'
                assert sha(p)==self.binding['original_scores'][self.family][str(t)]['sha256']
                old=read(p)
                for kind,tag in [('rewrite','RS'),('rephrase','PS')]:
                    prior={(r['case_id'],r['prompt_index']):r for r in old['metrics'][tag]['rows']}
                    for n,tr in zip(a[kind+'_target_new'],a[kind+'_target_true'],strict=True):
                        o=prior[n['case_id'],n['prompt_index']];token=self.tokens[n['case_id'],'rewrite' if kind=='rewrite' else 'paraphrase',n['prompt_index']]
                        original.append(historical_row(n,tr,o,token))
                del old
            self.evidence(prefix/'historical.json',dict(original_source=None if not t else record(p),rows=original))
            self.diagnostics[f'endpoint-{t:03d}-historical']=dict(warning_count=sum(r['warning_count'] for r in original))
        with self.b.state(recipe) as restored:after=self.raw(ids,evidence=prefix/'restored')
        checks.append(self.diagnostic(f'endpoint-{t:03d}-restore',a,after))
        result=dict(t=t,ids=ids,state=state,restored=restored,checks=checks,original=original,raw=a,repeat_raw=repeat,mb1_raw=single,restored_raw=after,original_status='NOT_AVAILABLE_W0' if not t else 'RECORDED_NOT_CERTIFIED',cost=self.b.cost())
        self.evidence(dest,result);print('ENDPOINT_DIAGNOSTICS_RECORDED',self.family,t,flush=True);return result

    def t1(self):
        self.stage='T1';self.b=Backend(self.binding,self.family)
        if (self.out/'T1/PASS.json').exists():
            self.gate('T1',{});self.barrier('T1');return
        eps=[self.verify_endpoint(t) for t in (0,1,20,50,100)];diagonals=[]
        ids=[int(r['case_id']) for r in rows(DESIGN/'fidelity-panel.csv')[:8]]
        prior=self.reuse['diagonals'].get(self.family)
        if prior:
            assert sha(prior['path'])==prior['sha256'],'REUSE_DIAGONAL_SHA'
            diagonals=read(prior['path'])['detail']['diagonals'];assert len(diagonals)==12
            self.gate('T1',dict(endpoints=[0,1,20,50,100],diagonals=diagonals,diagonals_reused_from=prior,
                               raw_diagonal_comparison='NOT_RECORDED_IN_ORIGINAL_RECEIPT',reuse_bridge=self.lock['reuse_bridge'],cost=self.b.cost()))
            self.barrier('T1');return
        for i in range(12):
            a,b=TIMES[i:i+2];recipe=dict(kind='ACTUAL',actual_checkpoint=a)
            expected=self.b.endpoint(a)
            with self.b.state(recipe,force_removal=(b,[i])) as arithmetic:
                assert arithmetic['weight_hashes']=={k:tensor_sha(v) for k,v in expected.items()},('DIAGONAL_BYTES',a,b)
                first=self.raw(ids,evidence=self.out/'diagnostic-raw'/f'diagonal-{i:02d}'/'arithmetic')
            with self.b.state(recipe) as direct:second=self.raw(ids,evidence=self.out/'diagnostic-raw'/f'diagonal-{i:02d}'/'direct')
            check=self.diagnostic(f'diagonal-{i:02d}',first,second);diagonals.append(dict(start=a,anchor=b,arithmetic=arithmetic,direct=direct,check=check))
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
                        self.evidence(path,x);self.cachemiss+=1
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
                first=self.raw(sentinel_ids,evidence=self.out/'diagnostic-raw'/sid/'first')
                second=self.raw(sentinel_ids,evidence=self.out/'diagnostic-raw'/sid/'repeat')
                checks=self.diagnostic(sid+'-repeat',first,second)
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
            with self.b.state(recipe) as restored:third=self.raw(sentinel_ids,evidence=self.out/'diagnostic-raw'/sid/'restore')
            restorecheck=self.diagnostic(sid+'-restore',first,third);self.evidence(sent_path,dict(state=state,restore_state=restored,ids=sentinel_ids,repeat=checks,restore=restorecheck))
        sr=save(dest/'scores.json',score)
        self.evidence(receipt,dict(status='PASS',pass_semantics='STRUCTURAL_COMPLETION_NOT_NUMERICAL_CERTIFICATION',task=task,state=state,scores=sr,cache=cache,numerical_diagnostics=self.warnings(),cost=self.b.cost()))
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
                x=read(self.out/'terminal.json');assert x['identity']==self.identity and x['status'] in COMPLETE;return
            self.t1();self.phase('T2P','pilot');self.phase('T2F','main');self.phase('T3B','pairs')
            self.b.copy(self.b.w0);assert self.b.hashes()==self.binding['w0_selected'];self.b.unchanged()
            self.evidence(self.out/'terminal.json',dict(status=completion(self.warnings()['warning_count']),
                numerical_diagnostics=self.warnings(),cost=self.b.cost(),job_id=os.environ.get('SLURM_JOB_ID')))
        except BaseException as e:
            self.evidence(self.out/'FAILURE.json',dict(status='TECHNICAL_FAILED',stage=self.stage,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),numerical_diagnostics=self.warnings(),cost=None if self.b is None else self.b.cost(),job_id=os.environ.get('SLURM_JOB_ID')))
            raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--family',choices=FAMILIES,required=True);a=p.parse_args();Run(a.lock,a.family).run()
