"""Independent CPU joins/arithmetic, censoring, fixed tables, five code-only figures."""
import argparse
import collections
import csv
import math
import traceback
from .common import *
from .fidelity import policy, execution_identity, check_gate, COMPLETE, completion

IDENTITY=('schema_version','case_id','panel','prompt_index','target_version','prompt_token_hash','target_token_hash','competitor_token_hash','evaluator_signature','valid','missing_reason')

def contribution(c,f,ss):
    mt,bt,mb,bb=(ss[k]['margin'] for k in ('M_t','B_t','M_b','B_b'))
    ct=mt-bt;cb=mb-bb;dm=mt-mb;db=bt-bb;dc=ct-cb;res=dm-db-dc
    assert abs(res)<=1e-10 and all(math.isfinite(x) for x in (mt,bt,mb,bb,ct,cb,dc))
    row={k:ss['M_t'][k] for k in IDENTITY}
    assert all(all(x[k]==row[k] for k in IDENTITY) for x in ss.values()),'CROSS_STATE_TOKEN_IDENTITY'
    row.update(row_type='contribution',cell_id=c['cell_id'],family=c['family'],cohort_id=c['cohort_id'],update_start=int(c['update_start']),
        anchor=int(c['anchor']),birth_batch=int(f['birth_batch']),eval_t=int(c['eval_t']),active_at_anchor=active(f,int(c['anchor'])),active_at_t=active(f,int(c['eval_t'])),
        same_batch_conflict=boolstr(f['same_batch_conflict']),first_later_conflict_batch=int(f['first_later_conflict_batch']) if f['first_later_conflict_batch'] else None,
        subject_relation_group=f['subject_relation_group'],repeated_group=boolstr(f['repeated_group']),M_t=mt,B_t=bt,C_t=ct,M_b=mb,B_b=bb,C_b=cb,
        delta_M=dm,delta_B=db,delta_C=dc,epsilon=.1,contribution_state='decreased' if dc<-.1 else ('increased' if dc>.1 else 'stable'),accounting_residual=res,
        source_score_keys={k:digest([v['state_weight_hash'],v['case_id'],v['panel'],v['prompt_index'],v['prompt_token_hash'],v['target_token_hash'],v['evaluator_signature']]) for k,v in ss.items()},
        target_nll=ss['M_t']['target_nll'],competitor_nll=ss['M_t']['competitor_nll'],target_strict_tf=ss['M_t']['target_strict_tf'],
        target_token_correct=ss['M_t']['target_token_correct'],target_token_count=ss['M_t']['target_token_count'],
        competitor_strict_tf=ss['M_t']['competitor_strict_tf'],competitor_token_correct=ss['M_t']['competitor_token_correct'],competitor_token_count=ss['M_t']['competitor_token_count'])
    return row

def pair(c,f,ss):
    m,u,v,uv=(ss[k]['margin'] for k in ('M','minus_U','minus_V','minus_UV'))
    row={k:ss['M'][k] for k in IDENTITY};assert all(all(x[k]==row[k] for k in IDENTITY) for x in ss.values())
    row.update(row_type='pair',pair_id=c['pair_id'],family=c['family'],past_cohort=c['past_cohort'],future_cohort=c['future_cohort'],eval_t=100,
        M=m,minus_U=u,minus_V=v,minus_UV=uv,D_M=m-v,D_B=u-uv,interaction=(m-v)-(u-uv),active_at_t=active(f,100),
        source_score_keys={k:digest([x['state_weight_hash'],x['case_id'],x['panel'],x['prompt_index'],x['prompt_token_hash'],x['target_token_hash'],x['evaluator_signature']]) for k,x in ss.items()},
        selection=c['selection'],subject_relation_group=f['subject_relation_group'])
    assert abs(row['interaction']-((m-u)-(v-uv)))<=1e-10
    return row

def jsonl(p,values):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial')
    with tmp.open('x') as f:
        for r in values:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        f.flush();os.fsync(f.fileno())
    os.link(tmp,p);tmp.unlink();return record(p)

def csvout(p,values):
    values=list(values);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',newline='') as f:
        if values:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    return record(p)

def lookup(root,family,identity=None):
    by={};receipts=[]
    for p in sorted((root/family/'tasks').glob('*/PASS.json')):
        r=read(p);task=r['task']
        if identity is not None:check_gate(r,identity)
        if task['case_selector']!='whole_cohort':continue
        assert sha(r['scores']['path'])==r['scores']['sha256']
        sr=read(r['scores']['path']);assert len(sr)==int(task['prompt_count'])
        for s in sr:
            k=(s['state_id'],s['case_id'],s['panel'],s['prompt_index'])
            assert k not in by;assert s['valid'] and s['missing_reason'] is None and math.isfinite(s['margin'])
            assert s['margin']==s['competitor_nll']-s['target_nll'];by[k]=s
        receipts.append(record(p))
    return by,receipts

def joins(root,identity=None):
    facts=rows(DESIGN/'fact-ledger.csv');contrib=[];pairs=[];inventory=[]
    for family in FAMILIES:
        by,rr=lookup(root,family,identity);inventory.extend(rr)
        for c in rows(DESIGN/'main-cells.csv'):
            if c['family']!=family:continue
            for f in (r for r in facts if r['cohort_id']==c['cohort_id']):
                for panel,idx in [('rewrite',0),('paraphrase',0),('paraphrase',1)]:
                    ss={k:by[c[k+'_state'],int(f['case_id']),panel,idx] for k in ('M_t','B_t','M_b','B_b')}
                    contrib.append(contribution(c,f,ss))
        for c in rows(DESIGN/'pair-cells.csv'):
            if c['family']!=family:continue
            for f in (r for r in facts if r['cohort_id']==c['past_cohort']):
                for panel,idx in [('rewrite',0),('paraphrase',0),('paraphrase',1)]:
                    ss={k:by[c[k+'_state'],int(f['case_id']),panel,idx] for k in ('M','minus_U','minus_V','minus_UV')}
                    pairs.append(pair(c,f,ss))
        del by
    assert len(contrib)==333600 and len(pairs)==48000
    assert len({(r['cell_id'],r['case_id'],r['panel'],r['prompt_index']) for r in contrib})==333600
    return contrib,pairs,inventory

def tables(out,cc,pp):
    import numpy as np,pandas as pd
    df=pd.DataFrame(cc);pf=pd.DataFrame(pp);df['success']=df.M_t>0;df['eligible']=df.active_at_anchor&(df.M_b>0)&(df.C_b>.1)
    df['age']=df.eval_t-df.anchor;df['panel_name']=np.where(df.panel=='rewrite','rewrite','paraphrase_'+df.prompt_index.astype(str))
    paired=df.groupby(['cohort_id','eval_t','case_id','panel_name']).eligible.transform('all');df['both_arm_eligible']=paired
    summaries=[]
    for (family,cohort,t,panel),g in df.groupby(['family','cohort_id','eval_t','panel_name'],sort=True):
        for epsilon in (.025,.05,.1,.2):
            for subset in ('initial_effective','all_anchor_success','all_facts','both_arm_eligible'):
                eligible=g.active_at_anchor&(g.M_b>0)&(g.C_b>epsilon)
                if subset=='all_anchor_success':eligible=g.active_at_anchor&(g.M_b>0)
                if subset=='all_facts':eligible=pd.Series(True,index=g.index)
                if subset=='both_arm_eligible':
                    peers=df[(df.cohort_id==cohort)&(df.eval_t==t)&(df.panel_name==panel)]
                    ee=peers.active_at_anchor&(peers.M_b>0)&(peers.C_b>epsilon)
                    ids=peers.assign(ee=ee).groupby('case_id').ee.all();eligible=g.case_id.map(ids)
                a=g[eligible&g.active_at_t];ret=a[a.success];lost=a[~a.success];n=len(a)
                def rate(x,d):return x/d if d else None
                weak=int((ret.delta_C < -epsilon).sum());strength=int((lost.delta_C>=-epsilon).sum());reverse=int((ret.C_t < -epsilon).sum())
                summaries.append(dict(family=family,cohort_id=cohort,anchor=int(g.anchor.iloc[0]),eval_t=int(t),panel=panel,epsilon=epsilon,subset=subset,
                    total_raw=len(g),anchor_eligible=int(eligible.sum()),active_denominator=n,censored=int(eligible.sum())-n,retained=len(ret),lost=len(lost),
                    retained_decreased=weak,retained_stable=int((ret.delta_C.abs()<=epsilon).sum()),retained_increased=int((ret.delta_C>epsilon).sum()),
                    lost_decreased=int((lost.delta_C < -epsilon).sum()),lost_stable=int((lost.delta_C.abs()<=epsilon).sum()),lost_increased=int((lost.delta_C>epsilon).sum()),
                    weak_given_retained=rate(weak,len(ret)),lost_maintained_or_increased=strength,lost_maintained_rate=rate(strength,len(lost)),
                    retained_reversal=reverse,reversal_given_retained=rate(reverse,len(ret)),weak_of_all_active=rate(weak,n),lost_stable_increase_of_all_active=rate(strength,n),reversal_of_all_active=rate(reverse,n),
                    M_mean=float(a.M_t.mean()) if n else None,B_mean=float(a.B_t.mean()) if n else None,C_mean=float(a.C_t.mean()) if n else None,
                    delta_B_mean=float(a.delta_B.mean()) if n else None,delta_C_mean=float(a.delta_C.mean()) if n else None,
                    delta_C_p05=float(a.delta_C.quantile(.05)) if n else None,delta_C_p95=float(a.delta_C.quantile(.95)) if n else None,
                    target_tf_token_micro=float(a.target_token_correct.sum()/a.target_token_count.sum()) if n else None,
                    target_tf_prompt_macro=float((a.target_token_correct/a.target_token_count).mean()) if n else None,
                    target_tf_strict=float(a.target_strict_tf.mean()) if n else None))
    csvout(out/'primary-tables.csv',summaries)
    # Full fixed panel means remain separate from conditional eligible tables.
    trajectory=df.groupby(['family','cohort_id','anchor','eval_t','age','panel_name']).agg(n=('case_id','size'),M=('M_t','mean'),B=('B_t','mean'),C=('C_t','mean'),dB=('delta_B','mean'),dC=('delta_C','mean'),native_retention=('success','mean')).reset_index()
    trajectory.to_csv(out/'trajectory.csv',index=False)
    drift=df.assign(dc2=df.delta_C**2).groupby(['family','cohort_id','eval_t','case_id']).agg(rms=('dc2',lambda x:float(np.sqrt(x.mean()))),prompts=('dc2','size')).reset_index();assert (drift.prompts==3).all();drift.to_csv(out/'panel-rms-drift.csv',index=False)
    ordered=df.sort_values(['family','case_id','panel_name','eval_t']);group=ordered.groupby(['family','case_id','panel_name'])
    for name in ('M_t','B_t','C_t'):ordered['increment_'+name]=group[name].diff()
    ordered['prior_success']=group.success.shift();ordered['recovered']=(ordered.prior_success==False)&ordered.success;ordered['lost_now']=(ordered.prior_success==True)&~ordered.success
    ordered['prior_contribution_state']=group.contribution_state.shift();ordered['contribution_transition']=ordered.prior_contribution_state.fillna('anchor')+'→'+ordered.contribution_state
    cols=['family','cohort_id','case_id','panel_name','eval_t','active_at_t','increment_M_t','increment_B_t','increment_C_t','recovered','lost_now','contribution_transition']
    ordered[cols].to_csv(out/'chronological-increments.csv',index=False)
    fixed=trajectory[((trajectory.age.isin([10,50]))|(trajectory.eval_t==100))&trajectory.anchor.between(20,90)]
    fixed.to_csv(out/'fixed-age.csv',index=False)
    # Per-cell cluster bootstrap: paired B/C and success labels share sampled clusters.
    bootstrap=[]
    for keys,g in df[df.panel=='rewrite'].groupby(['family','cohort_id','eval_t']):
        a=g[g.eligible&g.active_at_t];n=len(a)
        if not n:bootstrap.append(dict(family=keys[0],cohort=keys[1],t=int(keys[2]),n=0,clusters=0,delta_C_low=None,delta_C_high=None,weak_low=None,weak_high=None));continue
        vals=a.assign(weak=a.success&(a.delta_C<-.1),one=1).groupby('subject_relation_group').agg(dc=('delta_C','sum'),n=('one','sum'),weak=('weak','sum'),ret=('success','sum')).to_numpy(float)
        rng=np.random.default_rng(20260924);stats=[]
        for _ in range(40):
            ix=rng.integers(0,len(vals),(50,len(vals)));s=vals[ix].sum(1)
            stats.extend(zip(s[:,0]/s[:,1],np.divide(s[:,2],s[:,3],out=np.full(50,np.nan),where=s[:,3]>0)))
        arr=np.array(stats);bootstrap.append(dict(family=keys[0],cohort=keys[1],t=int(keys[2]),n=n,clusters=len(vals),delta_C_low=float(np.quantile(arr[:,0],.025)),delta_C_high=float(np.quantile(arr[:,0],.975)),weak_low=float(np.nanquantile(arr[:,1],.025)) if np.isfinite(arr[:,1]).any() else None,weak_high=float(np.nanquantile(arr[:,1],.975)) if np.isfinite(arr[:,1]).any() else None))
    csvout(out/'cluster-bootstrap-primary-rewrite.csv',bootstrap)
    summary=pd.DataFrame(summaries);primary=summary[(summary.epsilon==.1)&(summary.subset=='initial_effective')&summary.anchor.between(20,90)]
    aggregate=[]
    for keys,g in primary.groupby(['family','eval_t','panel']):
        for averaging in ('micro','macro'):
            valid=g[g.active_denominator>0];rate=(float(valid.retained.sum()/valid.active_denominator.sum()) if averaging=='micro' else float((valid.retained/valid.active_denominator).mean())) if len(valid) else None
            aggregate.append(dict(family=keys[0],t=int(keys[1]),panel=keys[2],averaging=averaging,cohorts=len(g),eligible_active=int(g.active_denominator.sum()),retained_rate=rate))
    csvout(out/'micro-macro.csv',aggregate)
    pairsummary=pf.groupby(['family','pair_id','past_cohort','future_cohort','selection','panel','prompt_index']).agg(n=('case_id','size'),active=('active_at_t','sum'),D_M=('D_M','mean'),D_B=('D_B','mean'),I=('interaction','mean'),I_q05=('interaction',lambda x:x.quantile(.05)),I_q95=('interaction',lambda x:x.quantile(.95))).reset_index();pairsummary.to_csv(out/'pair-summary.csv',index=False)
    figures(out,df,trajectory,fixed,ordered,pairsummary)
    return dict(contribution_rows=len(df),pair_rows=len(pf),primary_rows=len(summary),max_accounting_residual=float(df.accounting_residual.abs().max()),bootstrap_replicates=2000,bootstrap_seed=20260924)

def figures(out,df,tr,fixed,inc,pairs):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axs=plt.subplots(2,3,figsize=(13,7))
    for i,f in enumerate(FAMILIES):
        x=tr[(tr.family==f)&(tr.panel_name=='rewrite')]
        for _,g in x.groupby('cohort_id'):
            for j,k in enumerate(('M','B','C')):axs[i,j].plot(g.eval_t,g[k],alpha=.65);axs[i,j].set_title(f+' '+k)
    fig.tight_layout();fig.savefig(out/'01-trajectories.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(13,7))
    for i,f in enumerate(FAMILIES):
        x=df[(df.family==f)&(df.panel=='rewrite')&df.active_at_t].copy();x['weakened']=x.delta_C<-.1;x['reverse']=(x.C_b>.1)&(x.C_t<-.1)
        for j,k in enumerate(('success','weakened','reverse')):
            grid=x.pivot_table(index='anchor',columns='eval_t',values=k,aggfunc='mean');im=axs[i,j].imshow(grid,aspect='auto',vmin=0,vmax=1);axs[i,j].set_title(f+' '+k);axs[i,j].set_xticks(range(len(grid.columns)),grid.columns,rotation=90);axs[i,j].set_yticks(range(len(grid.index)),grid.index);fig.colorbar(im,ax=axs[i,j])
    fig.tight_layout();fig.savefig(out/'02-time-grid.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,f in zip(axs,FAMILIES):
        x=df[(df.family==f)&(df.panel=='rewrite')];ax.scatter(x.delta_B,x.delta_C,c=x.success.astype(int),s=1,alpha=.25,rasterized=True);ax.axhline(0,color='black');ax.axvline(0,color='black');ax.set(title=f,xlabel='delta B',ylabel='delta C')
    fig.tight_layout();fig.savefig(out/'03-delta-scatter.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(13,7))
    for i,f in enumerate(FAMILIES):
        for j,age in enumerate((10,50,100)):
            x=fixed[(fixed.family==f)&(fixed.panel_name=='rewrite')&((fixed.age==age) if age!=100 else (fixed.eval_t==100))];axs[i,j].bar(x.anchor,x.dC,width=6);axs[i,j].set_title(f+(' age '+str(age) if age!=100 else ' t100'))
    fig.tight_layout();fig.savefig(out/'04-fixed-age.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,7))
    for i,f in enumerate(FAMILIES):
        x=inc[(inc.family==f)&(inc.panel=='rewrite')].groupby('eval_t')[['increment_B_t','increment_C_t']].mean();x.plot(ax=axs[i,0]);p=pairs[(pairs.family==f)&(pairs.panel=='rewrite')];axs[i,1].bar(range(len(p)),p.I);axs[i,1].set_xticks(range(len(p)),p.pair_id.str.replace(f+'__',''),rotation=90);axs[i,1].set_title('whole U/V interaction')
    fig.tight_layout();fig.savefig(out/'05-increments-pairs.png',dpi=140);plt.close(fig)

def worker_completion(root,identity):
    terminal=[]
    for f in FAMILIES:
        fp=root/f/'terminal.json';fail=root/f/'FAILURE.json'
        x=read(fp) if fp.exists() else None
        if x is not None:assert x['identity']==identity,'TERMINAL_IDENTITY'
        terminal.append(dict(family=f,status=x['status'] if x else 'TECHNICAL_FAILED_OR_NOT_COMPLETED',
            numerical_diagnostics=x.get('numerical_diagnostics',{}) if x else {},
            failure=read(fail) if fail.exists() else None))
    ready=all(r['status'] in COMPLETE and r['failure'] is None for r in terminal)
    if ready:
        for f in FAMILIES:
            for stage in ('T1','T2P','T2F','T3B'):check_gate(read(root/f/stage/'PASS.json'),identity)
    return ready,terminal


def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);args=p.parse_args();lock=read(args.lock);root=Path(lock['output']);out=root/'T4';out.mkdir(parents=True,exist_ok=True)
    pol=policy(lock);identity=execution_identity(lock)
    try:
        ready,terminal=worker_completion(root,identity)
        if not ready:
            save(out/'terminal.json',dict(status='TECHNICAL_BLOCKED',identity=identity,**pol,workers=terminal,science_gate_bypassed=False));return
        nw=sum(w['numerical_diagnostics']['warning_count'] for w in terminal)
        cc,pp,rr=joins(root,identity);cr=jsonl(out/'contributions.jsonl',cc);pr=jsonl(out/'pairs.jsonl',pp)
        summary=tables(out,cc,pp);save(out/'T3A-PASS.json',dict(status='PASS',identity=identity,**pol,
            pass_semantics='STRUCTURAL_COMPLETION_NOT_NUMERICAL_CERTIFICATION',numerical_warning_count=nw,summary=summary,source_receipts=rr))
        text='# Historical update timeaxis 사실 보고\n\n두 BASE 전체 구간 U를 사용한 관측 전용 실행입니다. 새로운 편집·native fitting·history append·checkpoint 저장은 없습니다.\n\n'
        text+=f"실제 완료: 156 main cells / {len(cc)} prompt×time rows, 16 pair cells / {len(pp)} paired prompt rows.\n\n"
        text+=f"수치 재현 정책: `{pol['numerical_fidelity_policy']}` / numerical_certification=`NOT_ESTABLISHED`. 기록된 수치 경고 {nw}건(비교별 중복 가능). 완료 상태 `{completion(nw)}`는 계산·구조 완결성을 뜻하며 수치 허용치 PASS가 아닙니다. 사용자 waiver SHA `{pol['waiver_sha256']}`. 기존 기준 NLL 0.00025 / margin 0.0005 및 실제 오차·flip은 family별 numerical-diagnostics/diagnostic-raw에 보존합니다.\n\n"
        text+='세부 표: [primary](primary-tables.csv), [trajectory](trajectory.csv), [pair](pair-summary.csv), [fixed-age](fixed-age.csv), [bootstrap](cluster-bootstrap-primary-rewrite.csv).\n\n'
        text+='기준 ε=.10, 민감도 .025/.05/.10/.20. 원시 NLL strict >0와 TF 정확도는 별도입니다. 검열은 해당 시점 최초 충돌을 사용합니다. Cluster CI는 단일 고정 chain의 문항 구성 민감도이며 독립 순서 반복이 아닙니다. U를 처음부터 제거한 학습 이력이 아니라 실제 후속 parameter를 고정한 제거입니다.\n\n'
        text+='원자료·score cache·토큰·receipt는 local-only이며 이 보고는 과학적 채택 결론을 내리지 않습니다. NO_BROADCAST_NOT_REQUIRED: 동일 host 원본과 파생 관측을 사용합니다.\n'
        (out/'report-ko.md').write_text(text)
        save(out/'artifact-index.json',[record(q) for q in sorted(out.iterdir()) if q.is_file() and q.name!='artifact-index.json'])
        # Final success is the last create-once write, after report and inventory.
        save(out/'terminal.json',dict(status=completion(nw),identity=identity,**pol,numerical_warning_count=nw,summary=summary,contributions=cr,pairs=pr,workers=terminal,save_checkpoints=False,
            report=record(out/'report-ko.md'),artifact_index=record(out/'artifact-index.json')))
    except BaseException as e:
        save(out/'REDUCTION_FAILURE.json',dict(status='TECHNICAL_FAILED',identity=identity,**pol,error=str(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
