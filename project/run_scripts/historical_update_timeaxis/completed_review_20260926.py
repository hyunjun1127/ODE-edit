"""Read-only CPU review of immutable attempt-r3; independent arithmetic, no torch.

Original runner/reducer is never imported or executed. Only this review's output
directories are written. Raw identities and per-case products stay local.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

ROOT=Path('/data/janghj/ODE-edit/local/historical-update-timeaxis/20260924-v1')
RUN=ROOT/'attempt-r3-record-only'
RAW=RUN/'output'
DESIGN=ROOT/'inputs/design-package/members'
REPO=Path(__file__).resolve().parents[3]
LOCAL=ROOT/'completed-review-20260926-v1'
REPORT=REPO/'experiment-reports/servers/server4/historical-update-timeaxis-20260924-v1/completed-review-20260926-v1'
AUDIT=REPO/'audits/servers/server4/historical-update-timeaxis-20260924-v1/completed-review-20260926-v1'
FAMILIES=('BASE_ALPHAEDIT','BASE_MEMIT')
KEYS=tuple(f'model.layers.{i}.mlp.down_proj.weight' for i in range(4,9))
PANELS=('rewrite','paraphrase_0','paraphrase_1')
CAT=('rewrite_target_new','rewrite_target_true','rephrase_target_new','rephrase_target_true')
FILES={}


def digest(x,ascii=False):
    return hashlib.sha256(json.dumps(x,ensure_ascii=ascii,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def bytes_read(p):
    p=Path(p);s=p.stat();b=p.read_bytes();ss=p.stat()
    assert (s.st_size,s.st_mtime_ns)==(ss.st_size,ss.st_mtime_ns),'CHANGED_DURING_REVIEW'
    h=hashlib.sha256(b).hexdigest()
    if str(p) in FILES:assert FILES[str(p)]['sha256']==h
    FILES[str(p)]=dict(path=str(p),bytes=len(b),sha256=h,mtime_ns=s.st_mtime_ns)
    return b


def read(p):return json.loads(bytes_read(p))
def rows(p):
    import io
    return list(csv.DictReader(io.StringIO(bytes_read(p).decode())))
def istrue(v):return v is True or v=='True'
def active(f,t):return not istrue(f['same_batch_conflict']) and (not f['first_later_conflict_batch'] or int(f['first_later_conflict_batch'])>t)
def panel(r):return 'rewrite' if r['panel']=='rewrite' else 'paraphrase_'+str(r['prompt_index'])
def key(r):return (r['case_id'],r['panel'],r['prompt_index'])
def same(a,b,label):assert a==b,(label,a,b)
def finite(*values):assert all(math.isfinite(v) for v in values),'NONFINITE'


def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def csvout(p,rr):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);rr=list(rr)
    with p.open('x',newline='') as f:
        if rr:
            w=csv.DictWriter(f,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr)


def rate(n,d):return n/d if d else None


def contribution_terms(mt,bt,mb,bb):
    finite(mt,bt,mb,bb)
    ct=mt-bt;cb=mb-bb;dm=mt-mb;db=bt-bb;dc=ct-cb
    assert abs(dm-db-dc)<=1e-10
    return ct,cb,dm,db,dc


def pair_terms(m,u,v,uv):
    finite(m,u,v,uv)
    dm=m-v;db=u-uv;interaction=dm-db
    assert abs(interaction-((m-u)-(v-uv)))<=1e-10
    return dm,db,interaction


def summarize_scores(ss):
    n=len(ss)
    return dict(prompts=n,success=sum(s['margin']>0 for s in ss),ties=sum(s['margin']==0 for s in ss),
        native_preference=rate(sum(s['margin']>0 for s in ss),n),
        target_nll_mean=sum(s['target_nll'] for s in ss)/n if n else None,
        true_nll_mean=sum(s['competitor_nll'] for s in ss)/n if n else None,
        margin_mean=sum(s['margin'] for s in ss)/n if n else None,
        target_correct_tokens=sum(s['target_token_correct'] for s in ss),target_tokens=sum(s['target_token_count'] for s in ss),
        target_token_micro=rate(sum(s['target_token_correct'] for s in ss),sum(s['target_token_count'] for s in ss)),
        target_prompt_macro=sum(s['target_token_correct']/s['target_token_count'] for s in ss)/n if n else None,
        target_strict_count=sum(s['target_strict_tf'] for s in ss),target_strict_rate=rate(sum(s['target_strict_tf'] for s in ss),n),
        true_correct_tokens=sum(s['competitor_token_correct'] for s in ss),true_tokens=sum(s['competitor_token_count'] for s in ss),
        true_token_micro=rate(sum(s['competitor_token_correct'] for s in ss),sum(s['competitor_token_count'] for s in ss)),
        true_prompt_macro=sum(s['competitor_token_correct']/s['competitor_token_count'] for s in ss)/n if n else None,
        true_strict_count=sum(s['competitor_strict_tf'] for s in ss),true_strict_rate=rate(sum(s['competitor_strict_tf'] for s in ss),n))


def first_table():
    states={r['state_id']:r for r in rows(DESIGN/'state-bank.csv')}
    facts={int(r['case_id']):r for r in rows(DESIGN/'fact-ledger.csv')}
    results=[]
    for family in FAMILIES:
        final=[]
        for p in sorted((RAW/family/'tasks').glob('*/PASS.json')):
            x=read(p);task=x['task'];s=states[task['state_id']]
            if task['case_selector']!='whole_cohort' or s['kind']!='ACTUAL' or int(s['actual_checkpoint'])!=100:continue
            rr=read(x['scores']['path']);same(FILES[x['scores']['path']]['sha256'],x['scores']['sha256'],'SCORES_SHA')
            for r in rr:
                finite(r['target_nll'],r['competitor_nll']);same(r['margin'],r['competitor_nll']-r['target_nll'],'MARGIN')
            final.extend(rr)
        same(len(final),30000,'T100_ALL_REQUESTS');same(len({key(s) for s in final}),30000,'UNIQUE')
        for subset in ('all_facts','active_t100'):
            ss=[s for s in final if subset=='all_facts' or active(facts[s['case_id']],100)]
            for pan in PANELS:
                results.append(dict(family=family,subset=subset,panel=pan,**summarize_scores([s for s in ss if panel(s)==pan])))
    csvout(REPORT/'first-endpoint-table.csv',results)
    print(json.dumps({'first_table':results},ensure_ascii=False),flush=True)


class Review:
    def __init__(self):
        self.start=time.monotonic();self.lock=read(RUN/'execution.lock.json')
        same(FILES[str(RUN/'execution.lock.json')]['sha256'],'f99939e6edda921c6320110c44a4a0b03e5c8911e2937c8955ea5e98d9ba7ebc','LOCK')
        self.bind=read(self.lock['T0']['path']);same(FILES[self.lock['T0']['path']]['sha256'],self.lock['T0']['sha256'],'T0')
        self.tokens={key(r):r for r in read(self.bind['token_manifest']['path'])}
        same(FILES[self.bind['token_manifest']['path']]['sha256'],self.bind['token_manifest']['sha256'],'TOKEN_SHA')
        self.facts=rows(DESIGN/'fact-ledger.csv');self.fact={int(r['case_id']):r for r in self.facts}
        self.states={r['state_id']:r for r in rows(DESIGN/'state-bank.csv')}
        self.tasks={r['task_id']:r for r in rows(DESIGN/'score-tasks.csv')}
        self.cells=rows(DESIGN/'main-cells.csv');self.pairs=rows(DESIGN/'pair-cells.csv')
        data=read(self.bind['dataset']['path']);same(FILES[self.bind['dataset']['path']]['sha256'],self.bind['dataset']['sha256'],'DATASET')
        same([r['case_id'] for r in data],[int(f['case_id']) for f in self.facts],'ORDER')
        self.data={r['case_id']:r for r in data}
        for f,r in zip(self.facts,data):same(digest(r,True),f['raw_record_sha256'],'CASE_HASH')
        for m in self.lock['source_members']:
            bytes_read(m['path']);same(FILES[m['path']]['sha256'],m['sha256'],'SOURCE_CLOSURE')
        for m in self.bind['design_members']:
            bytes_read(m['path']);same(FILES[m['path']]['sha256'],m['sha256'],'DESIGN')
        for k in ('model_config','model_index'):
            r=self.bind[k];bytes_read(r['path']);same(FILES[r['path']]['sha256'],r['sha256'],k)
        reused=[]
        for family,items in self.bind['checkpoints'].items():
            for batch,r in items.items():
                s=Path(r['path']).stat();same((s.st_size,s.st_ino,s.st_mtime_ns),(r['bytes'],r['inode'],r['mtime_ns']),'CP_STAT')
                reused.append(dict(family=family,batch=batch,path=r['path'],bytes=r['bytes'],prior_sha256=r['sha256'],verification='PRIOR_FULL_SHA_PLUS_CURRENT_STAT'))
        same(len(reused),24,'CP24');csvout(REPORT/'checkpoint-input-reuse.csv',reused)
        for r in self.bind['model_shards']:
            s=Path(r['path']).stat();same((s.st_size,s.st_mtime_ns),(r['bytes'],r['mtime_ns']),'MODEL_STAT')
        self.identity=read(RAW/FAMILIES[0]/'terminal.json')['identity']
        for k in ('instruction_id','attempt','source_sha256','contract_sha256','T0_sha256','token_sha256'):same(self.identity[k],self.lock[k],k)
        same(self.identity['numerical_policy_sha256'],digest(self.lock['numerical_policy']),'POLICY')
        self.scores={};self.task_counts=[];self.cache_seen=set();self.cache_rows=0;self.cache_references=0
        self.state_seen={};self.gates=[];self.cost=[]

    def gate(self,r):
        same(r['status'],'PASS','STRUCTURAL_PASS');same(r['identity'],self.identity,'IDENTITY')
        same(r['pass_semantics'],'STRUCTURAL_COMPLETION_NOT_NUMERICAL_CERTIFICATION','PASS_SEMANTICS')
        same(r['numerical_certification'],'NOT_ESTABLISHED','CERTIFICATION')

    def state(self,r,sid):
        s=self.states[sid];actual=s['kind']=='ACTUAL'
        t=int(s['actual_checkpoint'] if actual else s['construction_endpoint'])
        remove=[] if actual else [int(i) for i in s['removed_cohort_indices'].split(';') if i]
        same(r['endpoint'],t,'STATE_ENDPOINT');same(r['removed'],remove,'REMOVED_U')
        same(set(r['weight_hashes']),set(KEYS),'WHOLE_FIVE_WEIGHTS');same(r['state_weight_hash'],digest(r['weight_hashes']),'STATE_HASH')
        same(r['restore'],'EXACT_BYTES','RESTORE')
        family=s['family']
        expected=self.bind['w0_selected'] if t==0 else self.bind['checkpoints'][family][str(t)]['weights']
        same(r['entry_hash'],expected,'ENTRY_BYTES')
        if actual:same(r['weight_hashes'],expected,'ACTUAL_WEIGHT_BYTES')
        if sid in self.state_seen:same(self.state_seen[sid],r['state_weight_hash'],'STATE_REPEATED_HASH')
        self.state_seen[sid]=r['state_weight_hash']

    def cache(self,refs,state):
        out={k:[] for k in CAT}
        for ref in refs:
            self.cache_references+=1
            c=read(ref['path']);same(FILES[ref['path']]['sha256'],ref['sha256'],'CACHE_FILE_SHA')
            same(c['state_weight_hash'],state['state_weight_hash'],'CACHE_WEIGHT')
            same(c['rows_sha256'],digest(c['rows']),'CACHE_ROWS_SHA')
            same(c['evaluator_signature'],digest(self.lock['evaluator_signature']),'EVALUATOR')
            layout=[]
            for row in c['rows']:
                kind=row['kind'];pan='rewrite' if kind.startswith('rewrite') else 'paraphrase'
                token=self.tokens[row['case_id'],pan,row['prompt_index']]
                target=token['target_ids' if kind.endswith('_new') else 'competitor_ids']
                same(row['target_token_ids'],target,'TARGET_IDS');finite(row['nll'])
                preds=row['token_predictions'];same(len(preds),len(target),'PREDICTION_CARDINALITY')
                correct=[a==b for a,b in zip(preds,target)];same(row['token_correct'],correct,'TF_MICRO');same(row['all_tokens_correct'],all(correct),'TF_STRICT')
                original=self.data[row['case_id']];rw=original['requested_rewrite']
                prompt=rw['prompt'].format(rw['subject']) if pan=='rewrite' else original['paraphrase_prompts'][row['prompt_index']]
                same(row['prompt'],prompt,'PROMPT');same(row['target'],rw['target_new' if kind.endswith('_new') else 'target_true']['str'],'TARGET_TEXT')
                layout.append(dict(case_id=row['case_id'],kind=kind,prompt_index=row['prompt_index'],encoded=[token['prompt_ids'],target]))
                out[kind].append(row)
            same(c['layout_sha256'],digest(layout),'LAYOUT');same(ref['layout_sha256'],c['layout_sha256'],'LAYOUT_REF')
            same(c['key'],digest([state['state_weight_hash'],layout,c['evaluator_signature']]),'CACHE_KEY')
            if ref['path'] not in self.cache_seen:self.cache_rows+=len(c['rows']);self.cache_seen.add(ref['path'])
        return out

    def load_scores(self):
        for family in FAMILIES:
            by={};whole=pilot=0
            expected={t['task_id'] for t in self.tasks.values() if self.states[t['state_id']]['family'] in (family,'SHARED')}
            paths=sorted((RAW/family/'tasks').glob('*/PASS.json'));same({p.parent.name for p in paths},expected,'ALL_SCORE_TASKS')
            for j,p in enumerate(paths):
                r=read(p);self.gate(r);task=r['task'];same(task,self.tasks[task['task_id']],'TASK_BINDING')
                sid=task['state_id'];self.state(r['state'],sid)
                ids=[int(f['case_id']) for f in self.facts if f['cohort_id']==task['cohort_id'] and (task['case_selector']=='whole_cohort' or istrue(f['pilot_selected']))]
                same(digest(ids,True),task['case_order_sha256'],'TASK_ORDER')
                rr=read(r['scores']['path']);same(FILES[r['scores']['path']]['sha256'],r['scores']['sha256'],'SCORE_FILE')
                same(len(rr),int(task['prompt_count']),'PROMPT_DENOMINATOR');same(len({key(s) for s in rr}),len(rr),'DUPLICATE_SCORES')
                raw=self.cache(r['cache'],r['state']);expected_rows=[]
                for kind in ('rewrite','rephrase'):
                    expected_order=[(i,k) for i in ids for k in (range(1) if kind=='rewrite' else range(2))]
                    for side in ('new','true'):same([(x['case_id'],x['prompt_index']) for x in raw[kind+'_target_'+side]],expected_order,'RAW_ORDER')
                    for n,t in zip(raw[kind+'_target_new'],raw[kind+'_target_true'],strict=True):
                        pan='rewrite' if kind=='rewrite' else 'paraphrase';tok=self.tokens[n['case_id'],pan,n['prompt_index']]
                        expected_rows.append(dict(case_id=n['case_id'],panel=pan,prompt_index=n['prompt_index'],target_version=tok['target_version'],
                            prompt_token_hash=tok['prompt_token_hash'],target_token_hash=tok['target_token_hash'],competitor_token_hash=tok['competitor_token_hash'],
                            evaluator_signature=digest(self.lock['evaluator_signature']),valid=True,missing_reason=None,state_id=sid,state_weight_hash=r['state']['state_weight_hash'],
                            target_nll=n['nll'],competitor_nll=t['nll'],margin=t['nll']-n['nll'],target_token_count=len(n['token_correct']),competitor_token_count=len(t['token_correct']),
                            target_token_correct=sum(n['token_correct']),competitor_token_correct=sum(t['token_correct']),target_strict_tf=n['all_tokens_correct'],competitor_strict_tf=t['all_tokens_correct'],
                            target_predictions=n['token_predictions'],competitor_predictions=t['token_predictions']))
                for a,b in zip(rr,expected_rows,strict=True):
                    for k in b:same(a[k],b[k],('INDEPENDENT_SCORE',task['task_id'],k))
                    if task['case_selector']=='whole_cohort':
                        k=(sid,*key(b));assert k not in by,'DUPLICATE_STATE_PROMPT'
                        by[k]={k:v for k,v in b.items() if k not in ('target_predictions','competitor_predictions')}
                if task['case_selector']=='whole_cohort':whole+=len(rr)
                else:pilot+=len(rr)
                if j%50==0:print('CPU_SCORE_REVIEW',family,j,len(paths),flush=True)
            self.scores[family]=by;self.task_counts.append(dict(family=family,tasks=len(paths),whole_cohort_score_rows=whole,pilot_subset_score_rows=pilot))
        csvout(REPORT/'score-coverage.csv',self.task_counts)

    def score(self,family,state,case,pan,index):return self.scores[family][state,case,pan,index]

    def join(self):
        """Rebuild every scalar from verified score/cache data, not collector arithmetic."""
        import pandas as pd
        cc=[];pp=[]
        with (RAW/'T4/contributions.jsonl').open() as old:
            for family in FAMILIES:
                for c in self.cells:
                    if c['family']!=family:continue
                    fs=[f for f in self.facts if f['cohort_id']==c['cohort_id']]
                    for f in fs:
                        case=int(f['case_id']);anchor=int(c['anchor']);t=int(c['eval_t'])
                        for pan,idx in (('rewrite',0),('paraphrase',0),('paraphrase',1)):
                            ss={k:self.score(family,c[k+'_state'],case,pan,idx) for k in ('M_t','B_t','M_b','B_b')}
                            mt,bt,mb,bb=(ss[k]['margin'] for k in ('M_t','B_t','M_b','B_b'))
                            ct,cb,dm,db,dc=contribution_terms(mt,bt,mb,bb)
                            x=json.loads(next(old))
                            expected=dict(cell_id=c['cell_id'],family=family,cohort_id=c['cohort_id'],case_id=case,panel=pan,prompt_index=idx,
                                update_start=int(c['update_start']),anchor=anchor,eval_t=t,birth_batch=int(f['birth_batch']),
                                M_t=mt,B_t=bt,C_t=ct,M_b=mb,B_b=bb,C_b=cb,delta_M=dm,delta_B=db,delta_C=dc,
                                active_at_anchor=active(f,anchor),active_at_t=active(f,t),accounting_residual=dm-db-dc,
                                contribution_state='decreased' if dc<-.1 else ('increased' if dc>.1 else 'stable'))
                            for k,v in expected.items():same(x[k],v,('COLLECTOR_CONTRIBUTION',c['cell_id'],k))
                            for k in ('target_nll','competitor_nll','target_strict_tf','target_token_correct','target_token_count',
                                      'competitor_strict_tf','competitor_token_correct','competitor_token_count'):
                                expected[k]=ss['M_t'][k];same(x[k],expected[k],('COLLECTOR_TF',k))
                            for k in ('target_version','prompt_token_hash','target_token_hash','competitor_token_hash','evaluator_signature'):
                                assert len({s[k] for s in ss.values()})==1;assert all(s[k]==x[k] for s in ss.values())
                            for k,v in ss.items():same(x['source_score_keys'][k],digest([v['state_weight_hash'],v['case_id'],v['panel'],v['prompt_index'],v['prompt_token_hash'],v['target_token_hash'],v['evaluator_signature']]),'SOURCE_SCORE_KEY')
                            expected.update(subject_relation_group=f['subject_relation_group'],repeated_group=istrue(f['repeated_group']),
                                            age=t-anchor,panel_name=pan if pan=='rewrite' else 'paraphrase_'+str(idx),success=mt>0,
                                            target_prompt_accuracy=ss['M_t']['target_token_correct']/ss['M_t']['target_token_count'])
                            cc.append(expected)
            assert not old.read().strip(),'EXTRA_CONTRIBUTION_ROWS'
        with (RAW/'T4/pairs.jsonl').open() as old:
            for family in FAMILIES:
                for c in self.pairs:
                    if c['family']!=family:continue
                    for f in (f for f in self.facts if f['cohort_id']==c['past_cohort']):
                        case=int(f['case_id'])
                        for pan,idx in (('rewrite',0),('paraphrase',0),('paraphrase',1)):
                            ss={k:self.score(family,c[k+'_state'],case,pan,idx) for k in ('M','minus_U','minus_V','minus_UV')}
                            m,u,v,uv=(ss[k]['margin'] for k in ('M','minus_U','minus_V','minus_UV'))
                            dm,db,interaction=pair_terms(m,u,v,uv)
                            x=json.loads(next(old));e=dict(pair_id=c['pair_id'],family=family,past_cohort=c['past_cohort'],future_cohort=c['future_cohort'],
                                case_id=case,panel=pan,prompt_index=idx,eval_t=100,M=m,minus_U=u,minus_V=v,minus_UV=uv,D_M=dm,D_B=db,interaction=interaction,
                                active_at_t=active(f,100),selection=c['selection'])
                            for k,vv in e.items():same(x[k],vv,('COLLECTOR_PAIR',k))
                            e.update(subject_relation_group=f['subject_relation_group'],panel_name=pan if pan=='rewrite' else 'paraphrase_'+str(idx),
                                     past_anchor=int(c['past_anchor']),future_end=int(c['future_end']),competitor_exposure_sum=int(c['competitor_exposure_sum']),
                                     target_exposure_sum=int(c['target_exposure_sum']))
                            pp.append(e)
            assert not old.read().strip(),'EXTRA_PAIR_ROWS'
        same(len(cc),333600,'MAIN_ROWS');same(len(pp),48000,'PAIR_ROWS')
        same(len({(r['cell_id'],r['case_id'],r['panel'],r['prompt_index']) for r in cc}),333600,'MAIN_UNIQUE')
        same(len({(r['pair_id'],r['case_id'],r['panel'],r['prompt_index']) for r in pp}),48000,'PAIR_UNIQUE')
        self.df=pd.DataFrame(cc);self.pf=pd.DataFrame(pp)
        print('CPU_INDEPENDENT_JOINS',len(cc),len(pp),flush=True)

    def primary_tables(self):
        import numpy as np
        import pandas as pd
        df=self.df;out=[]
        original={(r['family'],r['cohort_id'],int(r['eval_t']),r['panel'],float(r['epsilon']),r['subset']):r for r in rows(RAW/'T4/primary-tables.csv')}
        for (family,cohort,t,pan),g in df.groupby(['family','cohort_id','eval_t','panel_name'],sort=True):
            peer=df[(df.family!=family)&(df.cohort_id==cohort)&(df.eval_t==t)&(df.panel_name==pan)].set_index('case_id')
            for eps in (.025,.05,.1,.2):
                e=g.active_at_anchor&(g.M_b>0)&(g.C_b>eps)
                both=e&g.case_id.map(peer.active_at_anchor&(peer.M_b>0)&(peer.C_b>eps))
                for subset,eligible in (('initial_effective',e),('all_anchor_success',g.active_at_anchor&(g.M_b>0)),('all_facts',pd.Series(True,index=g.index)),('both_arm_eligible',both)):
                    a=g[eligible&g.active_at_t];ret=a[a.success];lost=a[~a.success]
                    row=dict(family=family,cohort_id=cohort,anchor=int(g.anchor.iloc[0]),eval_t=int(t),panel=pan,epsilon=eps,subset=subset,
                        total_raw=len(g),anchor_eligible=int(eligible.sum()),active_denominator=len(a),censored=int(eligible.sum())-len(a),retained=len(ret),lost=len(lost),
                        retained_decreased=int((ret.delta_C < -eps).sum()),retained_stable=int((ret.delta_C.abs()<=eps).sum()),retained_increased=int((ret.delta_C>eps).sum()),
                        lost_decreased=int((lost.delta_C < -eps).sum()),lost_stable=int((lost.delta_C.abs()<=eps).sum()),lost_increased=int((lost.delta_C>eps).sum()),
                        lost_maintained_or_increased=int((lost.delta_C>=-eps).sum()),retained_reversal=int((ret.C_t < -eps).sum()))
                    row.update(weak_given_retained=rate(row['retained_decreased'],len(ret)),lost_maintained_rate=rate(row['lost_maintained_or_increased'],len(lost)),
                               reversal_given_retained=rate(row['retained_reversal'],len(ret)),weak_of_all_active=rate(row['retained_decreased'],len(a)),
                               lost_stable_increase_of_all_active=rate(row['lost_maintained_or_increased'],len(a)),reversal_of_all_active=rate(row['retained_reversal'],len(a)),
                               M_mean=float(a.M_t.mean()) if len(a) else None,B_mean=float(a.B_t.mean()) if len(a) else None,C_mean=float(a.C_t.mean()) if len(a) else None,
                               delta_B_mean=float(a.delta_B.mean()) if len(a) else None,delta_C_mean=float(a.delta_C.mean()) if len(a) else None,
                               delta_C_p05=float(a.delta_C.quantile(.05)) if len(a) else None,delta_C_p95=float(a.delta_C.quantile(.95)) if len(a) else None,
                               target_tf_token_micro=rate(int(a.target_token_correct.sum()),int(a.target_token_count.sum())),
                               target_tf_prompt_macro=float(a.target_prompt_accuracy.mean()) if len(a) else None,target_tf_strict=float(a.target_strict_tf.mean()) if len(a) else None)
                    old=original[family,cohort,int(t),pan,eps,subset]
                    for k,v in row.items():
                        if v is None:same(old[k],'',('COLLECTOR_NA',k))
                        elif isinstance(v,(int,float)):assert abs(float(old[k])-v)<=1e-12,(k,old[k],v)
                        else:same(old[k],v,('COLLECTOR_PRIMARY',k))
                    out.append(row)
        same(len(out),7488,'PRIMARY_ROWS');self.primary=pd.DataFrame(out)
        csvout(REPORT/'primary-recomputed.csv',out)
        final=[]
        for (family,pan,eps,subset),g in self.primary[(self.primary.eval_t==100)&self.primary.anchor.between(20,90)].groupby(['family','panel','epsilon','subset']):
            row=dict(family=family,panel=pan,epsilon=eps,subset=subset,cohorts=len(g))
            for k in ('total_raw','anchor_eligible','active_denominator','censored','retained','lost','retained_decreased','retained_stable','retained_increased',
                      'lost_decreased','lost_stable','lost_increased','lost_maintained_or_increased','retained_reversal'):row[k]=int(g[k].sum())
            row.update(retained_rate=rate(row['retained'],row['active_denominator']),weak_given_retained=rate(row['retained_decreased'],row['retained']),
                       maintained_given_lost=rate(row['lost_maintained_or_increased'],row['lost']),reversal_given_retained=rate(row['retained_reversal'],row['retained']),
                       micro_retention=rate(row['retained'],row['active_denominator']),macro_retention=float((g.retained/g.active_denominator.replace(0,np.nan)).mean()))
            final.append(row)
        csvout(REPORT/'primary-t100-eight-cohorts.csv',final)
        print('CPU_PRIMARY_FINAL',json.dumps([r for r in final if r['panel']=='rewrite' and r['epsilon']==.1 and r['subset']=='initial_effective']),flush=True)

    def detailed_tables(self):
        import numpy as np
        import pandas as pd
        df=self.df;pf=self.pf
        endpoints=[];transitions=[]
        for family in FAMILIES:
            for sid,s in self.states.items():
                if s['kind']!='ACTUAL' or s['family'] not in (family,'SHARED'):continue
                t=int(s['actual_checkpoint']);rs=[r for (state,*_),r in self.scores[family].items() if state==sid and (t==0 or int(self.fact[r['case_id']]['birth_batch'])<=t)]
                if not rs:continue
                for subset in ('all_measured','active_at_t'):
                    subset_rows=[r for r in rs if subset=='all_measured' or active(self.fact[r['case_id']],max(1,t))]
                    for pan in PANELS:endpoints.append(dict(family=family,checkpoint=t,state_id=sid,subset=subset,panel=pan,**summarize_scores([r for r in subset_rows if panel(r)==pan])))
            for (cohort,t,pan),g in df[df.family==family].groupby(['cohort_id','eval_t','panel_name']):
                for subset in ('raw','active_both'):
                    a=g if subset=='raw' else g[g.active_at_anchor&g.active_at_t]
                    old=a.M_b>0;new=a.M_t>0
                    transitions.append(dict(family=family,cohort=cohort,anchor=int(g.anchor.iloc[0]),eval_t=int(t),panel=pan,subset=subset,n=len(a),
                        anchor_success=int(old.sum()),final_success=int(new.sum()),retained=int((old&new).sum()),lost=int((old&~new).sum()),gained=int((~old&new).sum()),
                        both_failed=int((~old&~new).sum()),lost_case_ids_sha256=digest(a.loc[old&~new,'case_id'].tolist()),gained_case_ids_sha256=digest(a.loc[~old&new,'case_id'].tolist())))
        csvout(REPORT/'endpoint-trajectory.csv',endpoints);csvout(REPORT/'anchor-final-transitions.csv',transitions)
        paired=[]
        final=df[(df.eval_t==100)]
        for pan in PANELS:
            a=final[(final.family==FAMILIES[0])&(final.panel_name==pan)].set_index('case_id')
            b=final[(final.family==FAMILIES[1])&(final.panel_name==pan)].set_index('case_id').loc[a.index]
            same(a.index.tolist(),b.index.tolist(),'PAIRED_FAMILY_ID')
            for subset in ('all10000','active9784'):
                mask=np.ones(len(a),bool) if subset=='all10000' else a.active_at_t.to_numpy()
                aa=a[mask];bb=b[mask];ap=aa.M_t>0;bp=bb.M_t>0
                paired.append(dict(panel=pan,subset=subset,n=len(aa),both_success=int((ap&bp).sum()),alpha_only=int((ap&~bp).sum()),memit_only=int((~ap&bp).sum()),both_failed=int((~ap&~bp).sum()),
                    alpha_minus_memit_target_nll=float((aa.target_nll-bb.target_nll).mean()),alpha_minus_memit_true_nll=float((aa.competitor_nll-bb.competitor_nll).mean()),
                    alpha_minus_memit_margin=float((aa.M_t-bb.M_t).mean())))
        csvout(REPORT/'family-paired-t100.csv',paired)
        tails=[];signed=[];boundary=[];worked=[]
        for (family,pan),g in df[(df.eval_t==100)&df.anchor.between(20,90)].groupby(['family','panel_name']):
            a=g[g.active_at_anchor&g.active_at_t&(g.M_b>0)&(g.C_b>.1)]
            for name in ('M_t','B_t','C_t','delta_M','delta_B','delta_C','target_nll','competitor_nll'):
                q=a[name];tails.append(dict(family=family,panel=pan,metric=name,n=len(a),mean=float(q.mean()),q01=float(q.quantile(.01)),q05=float(q.quantile(.05)),q50=float(q.quantile(.5)),q95=float(q.quantile(.95)),q99=float(q.quantile(.99)),minimum=float(q.min()),maximum=float(q.max())))
            for kept in (True,False):
                z=a[a.success==kept]
                signed.append(dict(family=family,panel=pan,retained=kept,n=len(z),decreased=int((z.delta_C<-.1).sum()),
                    decreased_positive_deltaB=int(((z.delta_C<-.1)&(z.delta_B>0)).sum()),
                    decreased_deltaB_covers_deltaC=int(((z.delta_C<-.1)&(z.delta_B>=-z.delta_C)).sum()),
                    deltaB_positive_deltaC_positive=int(((z.delta_B>0)&(z.delta_C>0)).sum()),deltaB_negative_deltaC_negative=int(((z.delta_B<0)&(z.delta_C<0)).sum())))
            for limit in (0.,.0005):
                z=a[(a.M_t.abs()>limit)&(a.M_b.abs()>limit)]
                boundary.append(dict(family=family,panel=pan,exclude_margin_abs_le=limit,denominator=len(z),excluded=len(a)-len(z),retained=int(z.success.sum()),lost=int((~z.success).sum())))
            if pan=='rewrite':
                selectors={'retained_min_deltaC':a[a.success].sort_values(['delta_C','case_id']).head(1),
                           'lost_max_deltaC':a[~a.success].sort_values(['delta_C','case_id'],ascending=[False,True]).head(1),
                           'retained_min_Ct':a[a.success].sort_values(['C_t','case_id']).head(1)}
                for label,z in selectors.items():
                    for _,r in z.iterrows():worked.append(dict(family=family,selection=label,case_id=int(r.case_id),cohort=r.cohort_id,anchor=int(r.anchor),eval_t=100,
                        M_b=r.M_b,B_b=r.B_b,C_b=r.C_b,M_t=r.M_t,B_t=r.B_t,C_t=r.C_t,delta_M=r.delta_M,delta_B=r.delta_B,delta_C=r.delta_C,retained=bool(r.success)))
        csvout(REPORT/'continuous-tails.csv',tails);csvout(REPORT/'signed-movements.csv',signed);csvout(REPORT/'boundary-exclusion.csv',boundary);csvout(REPORT/'worked-examples.csv',worked)
        pairrows=[]
        for (pid,pan),g in pf.groupby(['pair_id','panel_name']):
            for subset in ('all_fixed','active_t100'):
                a=g if subset=='all_fixed' else g[g.active_at_t]
                row=dict(pair_id=pid,family=g.family.iloc[0],panel=pan,subset=subset,past_anchor=int(g.past_anchor.iloc[0]),future_end=int(g.future_end.iloc[0]),
                    selection=g.selection.iloc[0],competitor_exposure_sum=int(g.competitor_exposure_sum.iloc[0]),target_exposure_sum=int(g.target_exposure_sum.iloc[0]),n=len(a))
                for k in ('D_M','D_B','interaction'):
                    row[k+'_mean']=float(a[k].mean());row[k+'_q05']=float(a[k].quantile(.05));row[k+'_q50']=float(a[k].quantile(.5));row[k+'_q95']=float(a[k].quantile(.95))
                row.update(interaction_negative=int((a.interaction<0).sum()),interaction_positive=int((a.interaction>0).sum()),interaction_zero=int((a.interaction==0).sum()),
                           actual_success=int((a.M>0).sum()),withoutV_success=int((a.minus_V>0).sum()),
                           removeV_gained=int(((a.M<=0)&(a.minus_V>0)).sum()),removeV_lost=int(((a.M>0)&(a.minus_V<=0)).sum()))
                pairrows.append(row)
        csvout(REPORT/'pair-effects-recomputed.csv',pairrows)
        # Store detailed transitions locally; no prompts/large per-case data in Git.
        seq=df.sort_values(['family','case_id','panel_name','eval_t']).copy();groups=seq.groupby(['family','case_id','panel_name'])
        for name in ('M_t','B_t','C_t'):seq['increment_'+name]=groups[name].diff()
        assert ((seq.increment_M_t-seq.increment_B_t-seq.increment_C_t).dropna().abs()<=1e-10).all()
        previous=groups.success.shift();seq['recovered']=(previous==False)&seq.success;seq['newly_lost']=(previous==True)&~seq.success
        cols=['family','cohort_id','case_id','panel_name','eval_t','active_at_t','increment_M_t','increment_B_t','increment_C_t','recovered','newly_lost']
        seq[cols].to_csv(LOCAL/'paired-chronological-rows.csv',index=False)
        g=seq[seq.active_at_t].groupby(['family','eval_t','panel_name']).agg(n=('case_id','size'),recovered=('recovered','sum'),newly_lost=('newly_lost','sum'),dM=('increment_M_t','mean'),dB=('increment_B_t','mean'),dC=('increment_C_t','mean')).reset_index()
        g.to_csv(REPORT/'chronological-summary.csv',index=False)
        # Request joint indicators are not equivalent to prompt strict accuracy.
        joint=[]
        for (family,t),g in df.groupby(['family','eval_t']):
            p=g.pivot(index='case_id',columns='panel_name',values='success');tf=g.pivot(index='case_id',columns='panel_name',values='target_strict_tf')
            same(len(p),100*t,'ALLSEEN_CASES')
            joint.append(dict(family=family,checkpoint=int(t),cases=len(p),rewrite_and_twoP=int(p.all(axis=1).sum()),twoP_both=int(p[['paraphrase_0','paraphrase_1']].all(axis=1).sum()),
                              TF_rewrite_and_twoP=int(tf.all(axis=1).sum()),TF_twoP_both=int(tf[['paraphrase_0','paraphrase_1']].all(axis=1).sum())))
        csvout(REPORT/'request-joint.csv',joint)
        # Fixed-age summaries retain the same prespecified eight comparable cohorts.
        p=self.primary[(self.primary.epsilon==.1)&(self.primary.subset=='initial_effective')&self.primary.anchor.between(20,90)].copy()
        p['age']=p.eval_t-p.anchor
        p[(p.age.isin([10,50]))|(p.eval_t==100)].to_csv(REPORT/'fixed-age-primary.csv',index=False)
        # Fact-level drift uses all three prompts, not a rewrite-only substitute.
        drift=df.assign(dc2=df.delta_C**2).groupby(['family','cohort_id','eval_t','case_id']).agg(square_sum=('dc2','sum'),prompts=('dc2','size')).reset_index()
        assert (drift.prompts==3).all();drift['rms']=np.sqrt(drift.square_sum/3)
        old=rows(RAW/'T4/panel-rms-drift.csv');same(len(old),len(drift),'RMS_CARDINALITY')
        for a,b in zip(old,drift.to_dict('records'),strict=True):
            for k in ('family','cohort_id'):same(a[k],b[k],'RMS_ID')
            for k in ('eval_t','case_id','prompts'):same(int(a[k]),b[k],'RMS_ID')
            assert abs(float(a['rms'])-b['rms'])<=1e-12,'RMS_VALUE'
        drift.groupby(['family','cohort_id','eval_t']).agg(n=('case_id','size'),mean_rms=('rms','mean'),p95_rms=('rms',lambda x:x.quantile(.95)),max_rms=('rms','max')).reset_index().to_csv(REPORT/'fact-rms-drift.csv',index=False)

    def diagnostics_and_stages(self):
        stages=[];costs=[];diags=[];warning_rows=[];sentinels=[];endpoints=[]
        for family in FAMILIES:
            previous={}
            for stage in ('T1','T2P','T2F','T3B'):
                p=RAW/family/stage/'PASS.json';x=read(p);self.gate(x);d=x['detail'];cost=d['cost']
                stages.append(dict(family=family,stage=stage,status='COMPLETED',semantics=x['pass_semantics'],sha256=FILES[str(p)]['sha256'],task_count=d.get('task_count',''),warnings=x['numerical_diagnostics']['warning_count']))
                c=dict(family=family,stage=stage,**cost)
                for k in ('seconds','forward_seconds','forward_calls','materialize_seconds','restore_seconds','selected_H2D_bytes','target_sequences','padded_tokens'):
                    c['stage_delta_'+k]=cost[k]-previous.get(k,0)
                previous=cost;costs.append(c)
            terminal=read(RAW/family/'terminal.json');same(terminal['identity'],self.identity,'TERMINAL_IDENTITY')
            assert terminal['status'] in ('COMPLETED','COMPLETED_WITH_NUMERICAL_WARNINGS')
            for p in sorted((RAW/family/'fidelity').glob('endpoint-*.json')):
                x=read(p);same(x['identity'],self.identity,'FIDELITY_IDENTITY');reused='reused_from' in x
                y=read(x['reused_from']['path']) if reused else x
                if reused:same(FILES[x['reused_from']['path']]['sha256'],x['reused_from']['sha256'],'REUSE_SHA')
                expected=self.bind['w0_selected'] if x['t']==0 else self.bind['checkpoints'][family][str(x['t'])]['weights']
                for name in ('state','restored'):
                    same(y[name]['weight_hashes'],expected,'ENDPOINT_BYTES');same(y[name]['restore'],'EXACT_BYTES','ENDPOINT_RESTORE')
                endpoints.append(dict(family=family,checkpoint=x['t'],reused=reused,cases=len(y['ids']),state_weight_hash=y['state']['state_weight_hash'],restore='RECORDED_EXACT_BYTES',new_GPU_evaluations=0 if reused else 'ACTUAL'))
                # Independently pair saved raw categories; no numerical policy code is imported.
                for label,field in (('repeat','repeat_raw'),('MB16-MB1','mb1_raw'),('restore','restored_raw')):
                    d=read(RAW/family/'numerical-diagnostics'/f"endpoint-{x['t']:03d}-{label}.json")['detail']
                    nr=[];mr=[]
                    for cat in CAT:
                        for a,b in zip(y['raw'][cat],y[field][cat],strict=True):
                            for k in ('case_id','kind','prompt_index','target_token_ids'):same(a[k],b[k],'FIDELITY_RAW_ID')
                            nr.append(abs(a['nll']-b['nll']))
                    for kind in ('rewrite','rephrase'):
                        aa=y['raw'];bb=y[field]
                        for a,b,c,e in zip(aa[kind+'_target_new'],aa[kind+'_target_true'],bb[kind+'_target_new'],bb[kind+'_target_true'],strict=True):
                            mr.append(abs((b['nll']-a['nll'])-(e['nll']-c['nll'])))
                    same(max(nr),d['max_nll'],'RAW_DIAGNOSTIC_NLL');same(max(mr),d['max_margin'],'RAW_DIAGNOSTIC_MARGIN')
            for p in sorted((RAW/family/'sentinels').glob('*.json')):
                x=read(p);same(x['identity'],self.identity,'SENTINEL_IDENTITY');same(len(x['ids']),8,'SENTINEL8')
                self.state(x['state'],p.stem);self.state(x['restore_state'],p.stem)
                sentinels.append(dict(family=family,state_id=p.stem,cases=8,repeat_nll=x['repeat']['max_nll'],restore_nll=x['restore']['max_nll'],weight_hash=x['state']['state_weight_hash'],restore=x['state']['restore']))
            for p in sorted((RAW/family/'numerical-diagnostics').glob('*.json')):
                x=read(p);same(x['identity'],self.identity,'DIAGNOSTIC_IDENTITY');d=x['detail']
                for kind in ('nll_rows','margin_rows'):
                    for r in d[kind]:
                        finite(r['left'],r['right']);same(r['absolute_error'],abs(r['left']-r['right']),'NUMERIC_DIFF');same(r['excess'],max(0.,r['absolute_error']-r['limit']),'EXCESS')
                same(d['max_nll'],max([r['absolute_error'] for r in d['nll_rows']],default=0.),'MAX_NLL')
                same(d['max_margin'],max([r['absolute_error'] for r in d['margin_rows']],default=0.),'MAX_MARGIN')
                same(d['warning_count'],len(d['warnings']),'WARNING_COUNT')
                diags.append(dict(family=family,comparison=x['comparison'],max_nll=d['max_nll'],max_margin=d['max_margin'],warning_count=d['warning_count'],flip_count=len(d['boundary_flips']),rows=len(d['nll_rows'])))
                for r in d['warnings']:warning_rows.append(dict(family=family,comparison=x['comparison'],code=r['code'],case_id=r['case_id'],category=r.get('category',r.get('kind')),prompt_index=r['prompt_index'],absolute_error=r['absolute_error'],limit=r['limit'],excess=r['excess'],flip='FLIP' in r['code']))
            for p in sorted((RAW/family/'diagnostic-raw').glob('endpoint-*/historical.json')):
                x=read(p)
                for r in x['rows']:
                    same(r['new_error'],abs(r['actual_new']-r['original_new']),'HISTORICAL_NEW')
                    same(r['true_error'],abs(r['actual_true']-r['original_true']),'HISTORICAL_TRUE')
                    same(r['margin_error'],abs(r['actual_true']-r['actual_new']-r['original_margin']),'HISTORICAL_MARGIN')
                    for code in r['warnings']:warning_rows.append(dict(family=family,comparison=p.parent.name+'-historical',code=code,case_id=r['case_id'],category=r['panel'],prompt_index=r['prompt_index'],absolute_error=max(r['new_error'],r['true_error']),limit=r['nll_limit'],excess=r['nll_excess'],flip=r['boundary_flip']))
            same(sum(r['warning_count'] for r in diags if r['family']==family)+sum(1 for r in warning_rows if r['family']==family and r['code'].startswith('HISTORICAL')),terminal['numerical_diagnostics']['warning_count'],'TERMINAL_WARNING_COUNT')
        same(len(sentinels),148,'ALL_REMOVAL_SENTINELS');same(len(endpoints),26,'ENDPOINTS_W0_SHARED_LOGICAL')
        for family in FAMILIES:
            same(len([x for x in sentinels if x['family']==family]),74,'SENTINELS_EACH')
        t3=read(RAW/'T4/T3A-PASS.json');self.gate(t3)
        for ref in t3['source_receipts']:
            if ref['path'] not in FILES:bytes_read(ref['path'])
            same(FILES[ref['path']]['sha256'],ref['sha256'],'JOIN_RECEIPT')
        end=read(RAW/'T4/terminal.json');same(end['identity'],self.identity,'T4_IDENTITY')
        for stage in ('T3A','T4'):stages.append(dict(family='BOTH',stage=stage,status='COMPLETED_WITH_NUMERICAL_WARNINGS',semantics='STRUCTURAL_COMPLETION_NOT_NUMERICAL_CERTIFICATION',sha256=FILES[str(RAW/('T4/T3A-PASS.json' if stage=='T3A' else 'T4/terminal.json'))]['sha256'],task_count='',warnings=5))
        csvout(REPORT/'stage-coverage.csv',stages);csvout(REPORT/'compute.csv',costs);csvout(REPORT/'numerical-diagnostics.csv',diags)
        csvout(REPORT/'numerical-warnings.csv',warning_rows);csvout(REPORT/'state-sentinel-coverage.csv',sentinels);csvout(REPORT/'endpoint-fidelity-coverage.csv',endpoints)
        self.max_margin=max(d['max_margin'] for d in diags)
        print('CPU_DIAGNOSTICS',len(diags),len(warning_rows),'max_margin',self.max_margin,flush=True)

    def bootstrap_and_plots(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # Same declared seed/cluster unit; independently formed cluster sufficient statistics.
        original={(x['family'],x['cohort'],int(x['t'])):x for x in rows(RAW/'T4/cluster-bootstrap-primary-rewrite.csv')}
        results=[]
        for (family,cohort,t),g in self.df[self.df.panel_name=='rewrite'].groupby(['family','cohort_id','eval_t']):
            a=g[g.active_at_anchor&g.active_at_t&(g.M_b>0)&(g.C_b>.1)]
            group=a.assign(one=1,weak=a.success&(a.delta_C<-.1)).groupby('subject_relation_group').agg(dc=('delta_C','sum'),n=('one','sum'),weak=('weak','sum'),ret=('success','sum'))
            vals=group.to_numpy(float);n=len(vals);samples=[];rng=np.random.default_rng(20260924)
            if n:
                for _ in range(40):
                    totals=vals[rng.integers(n,size=(50,n))].sum(axis=1)
                    weak=np.divide(totals[:,2],totals[:,3],out=np.full(50,np.nan),where=totals[:,3]>0)
                    samples.extend(zip(totals[:,0]/totals[:,1],weak))
                b=np.array(samples)
                out=dict(delta_C_low=float(np.quantile(b[:,0],.025)),delta_C_high=float(np.quantile(b[:,0],.975)),weak_low=float(np.nanquantile(b[:,1],.025)) if np.isfinite(b[:,1]).any() else None,weak_high=float(np.nanquantile(b[:,1],.975)) if np.isfinite(b[:,1]).any() else None)
            else:out=dict(delta_C_low=None,delta_C_high=None,weak_low=None,weak_high=None)
            old=original[family,cohort,int(t)]
            same(int(old['n']),len(a),'BOOT_N');same(int(old['clusters']),n,'BOOT_CLUSTERS')
            for k,v in out.items():
                if v is None:same(old[k],'','BOOT_NA')
                else:assert abs(float(old[k])-v)<1e-12,('BOOT_CI',k,old[k],v)
            results.append(dict(family=family,cohort=cohort,t=int(t),n=len(a),clusters=n,replicates=2000,seed=20260924,**out))
        csvout(REPORT/'bootstrap-recomputed.csv',results)
        # Five deterministic figures, made only from independently verified values.
        images=REPORT/'figures';images.mkdir(exist_ok=True)
        def finish(fig,name):
            fig.tight_layout();fig.savefig(images/name,dpi=140,metadata={'Software':'historical-timeaxis CPU review'});plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(11,4))
        for family in FAMILIES:
            p=self.primary[(self.primary.family==family)&(self.primary.panel=='rewrite')&(self.primary.epsilon==.1)&(self.primary.subset=='initial_effective')&self.primary.anchor.between(20,90)]
            gg=p.groupby('eval_t')[['retained','active_denominator','retained_decreased']].sum();label=family.removeprefix('BASE_')
            axs[0].plot(gg.index,gg.retained/gg.active_denominator,label=label,marker='o');axs[1].plot(gg.index,gg.retained_decreased/gg.retained,label=label,marker='o')
        for ax,title in zip(axs,['Retained / eligible active','Contribution weakened / retained']):ax.set(xlabel='Checkpoint t (cohort mix changes)',ylabel='Fraction',title=title);ax.legend();ax.grid(alpha=.2)
        finish(fig,'01-retention-and-contribution.png')
        fig,axs=plt.subplots(1,2,figsize=(11,4))
        for ax,family in zip(axs,FAMILIES):
            a=self.df[(self.df.family==family)&(self.df.panel_name=='rewrite')&(self.df.eval_t==100)&self.df.anchor.between(20,90)&self.df.active_at_anchor&self.df.active_at_t&(self.df.M_b>0)&(self.df.C_b>.1)]
            for good,color,label in [(True,'#2676aa','Retained'),(False,'#cf643e','Lost')]:
                z=a[a.success==good];ax.scatter(z.delta_C,z.delta_B,s=4,alpha=.25,c=color,label=label)
            ax.axhline(0,c='gray',lw=.5);ax.axvline(0,c='gray',lw=.5);ax.set(xlabel='C_t - C_b',ylabel='B_t - B_b',title=family);ax.legend()
        finish(fig,'02-signed-decomposition.png')
        fig,axs=plt.subplots(1,2,figsize=(11,4))
        for ax,family in zip(axs,FAMILIES):
            p=self.primary[(self.primary.family==family)&(self.primary.panel=='rewrite')&(self.primary.epsilon==.1)&(self.primary.subset=='initial_effective')&self.primary.anchor.between(20,90)]
            mat=p.pivot(index='anchor',columns='eval_t',values='weak_given_retained');im=ax.imshow(mat,aspect='auto',vmin=0,vmax=1,cmap='viridis');ax.set_xticks(range(len(mat.columns)),mat.columns);ax.set_yticks(range(len(mat.index)),mat.index);ax.set(title=family,xlabel='t',ylabel='anchor b');fig.colorbar(im,ax=ax)
        finish(fig,'03-cohort-heatmap.png')
        fig,ax=plt.subplots(figsize=(10,5));p=self.pf[self.pf.panel_name=='rewrite'].groupby(['family','pair_id']).agg(dm=('D_M','mean'),db=('D_B','mean'),i=('interaction','mean'))
        x=np.arange(len(p));ax.bar(x-.24,p.dm,.24,label='D_M');ax.bar(x,p.db,.24,label='D_B');ax.bar(x+.24,p.i,.24,label='Interaction');ax.set_xticks(x,[z[1] for z in p.index],rotation=75,fontsize=7);ax.set(ylabel='Mean nats / target token',title='Fixed 16 pair cells; all fixed rewrite cases');ax.legend();ax.axhline(0,c='gray',lw=.5)
        finish(fig,'04-pair-effects.png')
        fig,ax=plt.subplots(figsize=(8,4))
        for family in FAMILIES:
            p=self.primary[(self.primary.family==family)&(self.primary.eval_t==100)&(self.primary.panel=='rewrite')&(self.primary.subset=='initial_effective')&self.primary.anchor.between(20,90)].groupby('epsilon')[['retained_decreased','retained']].sum()
            ax.plot(p.index,p.retained_decreased/p.retained,marker='o',label=family)
        ax.set(xlabel='Prespecified epsilon (nats / target token)',ylabel='Weakened / retained',title='Threshold sensitivity; eight comparable cohorts at t100');ax.legend();ax.grid(alpha=.2)
        finish(fig,'05-epsilon-sensitivity.png')

    def inventory(self):
        # Hash all result files once; large raw index remains local, directory roots compact.
        groups=collections.defaultdict(list)
        for p in sorted(RAW.rglob('*')):
            if not p.is_file():continue
            if str(p) not in FILES:
                st=p.stat();h=hashlib.sha256()
                with p.open('rb') as f:
                    for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
                same((st.st_size,st.st_mtime_ns),(p.stat().st_size,p.stat().st_mtime_ns),'IMMUTABLE_OUTPUT')
                FILES[str(p)]=dict(path=str(p),bytes=st.st_size,sha256=h.hexdigest(),mtime_ns=st.st_mtime_ns)
            rel=p.relative_to(RAW);groups[rel.parts[0]].append(dict(path=str(rel),bytes=FILES[str(p)]['bytes'],sha256=FILES[str(p)]['sha256']))
        inventory=[dict(group=k,files=len(v),bytes=sum(x['bytes'] for x in v),root_sha256=digest(v)) for k,v in sorted(groups.items())]
        save(LOCAL/'raw-full-sha-inventory.json',dict(root=str(RAW),groups=dict(groups)))
        save(AUDIT/'artifact-inventory-summary.json',dict(groups=inventory,raw_inventory_path=str(LOCAL/'raw-full-sha-inventory.json'),raw_inventory_sha256=hashlib.sha256((LOCAL/'raw-full-sha-inventory.json').read_bytes()).hexdigest(),model_CP='PRIOR_FULL_SHA_PLUS_CURRENT_STAT_NOT_REHASHED'))
        compact=[x for x in FILES.values() if '/output/' not in x['path'] and 'dataset' not in x['path']]
        save(AUDIT/'analysis-input-manifest.json',dict(identity=self.identity,source_and_authority=compact,verified_files=len(FILES),cache_files=len(self.cache_seen),cache_raw_rows=self.cache_rows,state_count=len(self.state_seen),task_counts=self.task_counts,analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        save(AUDIT/'independent-review-checks.json',dict(status='PASS_CPU_STORED_EVIDENCE_REDUCTION',new_GPU=0,model_loaded=False,original_runtime_modified=False,independent_agent_used=False,
            independent_reducer=True,main_rows=len(self.df),pair_rows=len(self.pf),main_cells=156,pair_cells=16,primary_rows=len(self.primary),accounting_max=float(self.df.accounting_residual.abs().max()),
            cache_rows=self.cache_rows,states=len(self.state_seen),numerical_certification='NOT_ESTABLISHED',seconds=time.monotonic()-self.start))
        print('CPU_REVIEW_COMPLETE',json.dumps(inventory),flush=True)


def main():
    global REPORT,AUDIT,LOCAL
    p=argparse.ArgumentParser();p.add_argument('--first',action='store_true');p.add_argument('--review-output-root',type=Path);a=p.parse_args()
    if a.review_output_root:
        root=a.review_output_root.resolve();assert not root.exists(),'NEW_OUTPUT_ROOT_REQUIRED'
        root.mkdir(parents=True);LOCAL=root;REPORT=root/'report';AUDIT=root/'audit';REPORT.mkdir();AUDIT.mkdir()
    if a.first:first_table()
    else:
        r=Review();r.load_scores();r.join();r.primary_tables();r.detailed_tables();r.diagnostics_and_stages();r.bootstrap_and_plots();r.inventory()
        print(json.dumps(dict(task_counts=r.task_counts,cache_rows=r.cache_rows,unique_cache_files=len(r.cache_seen),state_count=len(r.state_seen),seconds=time.monotonic()-r.start)),flush=True)

if __name__=='__main__':main()
