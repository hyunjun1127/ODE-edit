"""Compact arithmetic summaries and same-state candidate comparisons."""
import collections,csv,datetime
from reducer import *
from performance import summary,transition
def rows(name):return list(csv.DictReader((REPORT/name).open()))
def main():
    selections=rows('selection.csv');fits=rows('fit-cost.csv');scores=rows('score-cost.csv');cands=rows('candidate.csv');history=rows('history.csv')
    sched=load(LOCAL/'scheduler-r1.json');alloc=[];intervals=[]
    mapping=['C45678','N4','F48','G48','C4','C48']
    for line in sched['stdout'].splitlines():
        x=line.split('|')
        if '.' in x[0]:continue
        arm=mapping[int(x[0].split('_')[1])];assert x[3]=='janghj' and x[4]=='COMPLETED' and x[5]=='0:0' and x[11]=='server4' and 'gres/gpu=1' in x[9]
        t=load(ROOT/'arms'/arm/'attempt-v1/output/terminal.json')
        batchline=next(z.split('|') for z in sched['stdout'].splitlines() if z.startswith(x[0]+'.batch|'))
        alloc.append(dict(arm=arm,array_id=x[0],physical_job_id=x[1],owner=x[3],state=x[4],exit=x[5],start_KST=x[6],end_KST=x[7],allocated_GPU_seconds=int(x[8]),allocated_GPU_hours=int(x[8])/3600,MaxRSS=batchline[10],program_seconds=t['seconds'],peak_GPU_allocated=t['peak_GPU_allocated'],peak_GPU_reserved=t['peak_GPU_reserved'],peak_host_RSS_KiB=t['peak_host_RSS_KiB']))
        intervals += [(x[6],1),(x[7],-1)]
    active=peak=0
    for _,delta in sorted(intervals,key=lambda x:(x[0],x[1])):active+=delta;peak=max(peak,active)
    assert peak<=2 and active==0
    mech=[];cost=[];pruning=[];localpairs=[];gates=[];genericrows=[]
    for arm in ARMS:
        ss=[r for r in selections if r['arm']==arm];ff=[r for r in fits if r['arm']==arm];cc=[r for r in cands if r['arm']==arm];sc=[r for r in scores if r['arm']==arm]
        stops=collections.Counter(x['stop'] for x in ss);n4=sum(x['own_n4']=='True' for x in ss)
        failcounts=collections.Counter(reason for x in cc for reason in x['reasons'].split(';') if reason)
        mech.append(dict(arm=arm,batches=10,own_N4=n4,selected_nonN4=10-n4,selected_support1=sum(int(x['support'])==1 for x in ss),selected_support2=sum(int(x['support'])==2 for x in ss),selected_support3plus=sum(int(x['support'])>=3 for x in ss),selected_infeasible=sum(x['feasible']=='False' for x in ss),completed_candidates=len(cc),feasible_candidates=sum(x['feasible']=='True' for x in cc),incomplete=sum(int(x['incomplete']) for x in ss),coverage_adequate=sum(x['adequate']=='True' for x in ss),coverage_applicable=arm.startswith('C'),stop_reasons=json.dumps(stops,sort_keys=True),quality_failures=json.dumps(failcounts,sort_keys=True)))
        cost.append(dict(arm=arm,native_targets=sum(int(x['targets']) for x in ff),native_Adam=sum(int(x['Adam']) for x in ff),native_loss=sum(int(x['loss']) for x in ff),native_clamp=sum(int(x['clamp']) for x in ff),native_solves=len(ff),native_keys=len(ff),online_scores=sum(int(x['endpoints'])+1 for x in ss),history_appends=50,history_keys=50,native_inclusive_seconds=sum(float(x['seconds']) for x in ff),native_target_seconds=sum(float(x['target_seconds']) for x in ff),native_solve_seconds=sum(float(x['solve_seconds']) for x in ff),native_key_seconds=sum(float(x['key_seconds']) for x in ff),online_forward_calls=sum(int(x['forwards']) for x in sc),online_input_tokens=sum(int(x['input_tokens']) for x in sc),online_scored_tokens=sum(int(x['scored_tokens']) for x in sc),online_backward_calls=sum(int(x['backwards']) for x in sc),teacher_reads=sum(int(x['teacher_reads']) for x in sc),online_scoring_seconds=sum(float(x['seconds']) for x in sc),teacher_read_seconds=sum(float(x['teacher_read_seconds']) for x in sc),pure_writer='NOT_SEPARATED',state_io='NOT_SEPARATED',observer_forward_token_work='NOT_RECORDED_SEPARATELY'))
        for b in range(1,11):
            bd=ROOT/'arms'/arm/f'attempt-v1/output/B{b:03d}';sel=load(bd/'selection.json');s=sel['selected'];p=sel['stop'].get('pruning')
            if p:pruning.append(dict(arm=arm,batch=b,**{k:json.dumps(v) if isinstance(v,list) else v for k,v in p.items()}))
            layers=[4] if arm in ['N4','C4'] else [4,5,6,7,8] if arm=='C45678' else [4,8]
            for l,g in zip(layers,s['gates']):gates.append(dict(arm=arm,batch=b,layer=l,gate=g,actual_nonzero=l in s['active_layers']))
            if b in [1,5,10]:
                bindings=[load(q) for q in sorted((bd/'candidate-observer').glob('*-binding.json'))];n4=sel['candidates'][0]
                nr=load(next(x['rows']['path'] for x in bindings if x['state_token']==n4['state_token']));sr=load(bd/'selected-current.json')
                for tag in MULT:
                    t,d=transition(nr['metrics'][tag]['rows'],sr['metrics'][tag]['rows'],tag,arm=arm,batch=b,comparison='OWN_N4_TO_SELECTED_SAME_ENTRY');localpairs.append(t)
            # Recompute per-doc D, not a new teacher/forward pass.
            for q in sorted((bd/'episode/scores').glob('*.json')):
                x=load(q)['metrics']['details']['generic'];row0=x['rows'][0]
                if not genericrows:print('GENERIC_ROW_KEYS',list(row0))
                key=next((k for k in ['kl','D','value','loss'] if k in row0),None)
                if key is not None:
                    mean=sum(z[key] for z in x['rows'])/len(x['rows']);assert abs(mean-x['D'])<1e-12
                genericrows.append(dict(path=str(q),doc_count=len(x['rows']),row_scalar_key=key,mean_recomputed=key is not None))
    csvout('allocation.csv',alloc);csvout('mechanism-summary.csv',mech);csvout('actual-cost.csv',cost);csvout('pruning-summary.csv',pruning);csvout('selected-gates.csv',gates);csvout('own-n4-to-selected.csv',localpairs)
    writejson(LOCAL/'generic-reduction-checks.json',genericrows)
    writejson(REPORT/'cost-summary.json',dict(new_science_allocated_GPU_seconds=sum(x['allocated_GPU_seconds'] for x in alloc),new_science_GPU_hours=sum(x['allocated_GPU_hours'] for x in alloc),technical_prior_49421_GPU_seconds=3379,teacher_reuse_prior_GPU_seconds=98,review_GPU_seconds=0,observed_exact_six_peak_allocation=peak,cap=2,other_task_intervals='NOT_QUERIED',total_actual={k:sum(x[k] for x in cost) for k in ['native_targets','native_Adam','native_loss','native_clamp','native_solves','online_scores','history_appends']},planned_upper=load(ROOT/'execution.lock.json')['planned_science_upper_bounds'],allocation_is_utilization=False))
if __name__=='__main__':main()
