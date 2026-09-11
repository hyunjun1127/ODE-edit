"""Source evidence, all recorded native rewrite retention, descriptive comparisons."""
import itertools,json
from .common import *

def main():
    cumulative=csvread(OUT/'cumulative-metrics.csv');current=csvread(OUT/'current-metrics.csv');layers=csvread(OUT/'layer-updates.csv');cost=csvread(OUT/'batch-cost.csv')
    summary=csvread(OUT/'first-final-summary.csv');ids=[x['case_id'] for x in sample()['records']]
    rw=[];nativecohort=[]
    for arm in ['BASE_MEMIT','BASE_ALPHAEDIT']:
        prior=[]
        for b in range(1,101):
            r=root(arm)/f'B{b:03d}';ev=read(r/'seen-rewrite.json');c=read(r/'commit.json');now=read(r/'current.json')
            prior+=now['metrics']['RS']['rows']
            assert ev['state']==c['endpoint'] and ev['requests']==b*100
            rows=ev['metrics']['RS']['rows'];assert [(x['case_id'],x['prompt_index']) for x in rows]==[(i,0) for i in ids[:b*100]]
            z=reduce_rows(rows,'RS');assert z['numerator']==ev['metrics']['RS']['numerator']
            rw.append(dict(arm=arm,batch=b,**z,**p99(rows),**{'atwrite_'+k:v for k,v in transition(prior,rows).items()}))
            for cohort in range(1,b+1):
                part=rows[(cohort-1)*100:cohort*100]
                nativecohort.append(dict(arm=arm,batch=b,cohort=cohort,metric='RS',numerator=sum(x['success'] for x in part),denominator=100,rate=np.mean([x['success'] for x in part])))
    csvwrite(OUT/'native-all-batches-seen-rewrite.csv',rw);csvwrite(OUT/'native-all-batches-cohort-retention.csv',nativecohort)
    comparisons=[]
    pairs=[]
    for method in ['MEMIT','AlphaEdit']:
        names=[a for a in ARMS if family(a)==method]
        pairs += [(a,b,'WITHIN_'+method) for a,b in itertools.combinations(names,2)]
    pairs += [('BASE_MEMIT','BASE_ALPHAEDIT','NATIVE_CROSS_FAMILY')]
    for variant in ['ORIGINAL']+[f'L{l}_ONLY' for l in range(4,9)]:pairs.append(('MEMIT_'+variant,'AlphaEdit_'+variant,'CROSS_FAMILY_MATCHED_VARIANT'))
    for a,b,kind in pairs:
        for batch in SCHEDULE:
            for tag in MULT:
                x=next(r for r in cumulative if r['arm']==a and int(r['batch'])==batch and r['metric']==tag);y=next(r for r in cumulative if r['arm']==b and int(r['batch'])==batch and r['metric']==tag)
                assert x['denominator']==y['denominator']
                q=dict(arm_before=a,arm_after=b,comparison=kind,batch=batch,metric=tag,before_num=x['numerator'],after_num=y['numerator'],denominator=x['denominator'])
                for k in ['rate']+[f'{side}_nll_prompt_{stat}' for side in ['new','true'] for stat in ['mean','median','p90','max']]+[f'margin_prompt_{stat}' for stat in ['mean','median','p90','max']]:q[k+'_delta']=float(y[k])-float(x[k])
                comparisons.append(q)
    csvwrite(OUT/'paired-checkpoint-deltas.csv',comparisons)
    cp=csvread(INHERITED/'checkpoint-tensors.csv')+csvread(OUT/'new-checkpoint-tensors.csv')
    ci=csvread(INHERITED/'chain-integrity.csv')+csvread(OUT/'new-chain-integrity.csv')
    cfg=csvread(INHERITED/'source-config-compatibility.csv')+csvread(OUT/'new-source-config-compatibility.csv')
    configs=[];compat=[];pre=read(OLD/'preedit-runtime-summary.json')['runtime'];prew=pre.get('W0',pre.get('entry',{}))
    # Native entry supplies all five selected weights; old W0 publication identity is inherited separately.
    reference=read(root('BASE_ALPHAEDIT')/'runtime.json')
    for arm in ARMS:
        rt=read(root(arm)/'runtime.json');lk=read(attempt(arm)/'execution.lock.json');hp=rt['hparams']
        assert all(reference['W0']['weights'][k]['sha256']==v['sha256'] for k,v in rt['W0']['weights'].items())
        assert (rt['model_revision'],rt['dtype'],rt['attention'],rt['torch'],rt['transformers'],rt['evaluator_tokenizer'])==(reference['model_revision'],reference['dtype'],reference['attention'],reference['torch'],reference['transformers'],reference['evaluator_tokenizer'])
        configs.append(dict(arm=arm,display=label(arm),family=family(arm),blue=hp['blue'],layers=str(hp['layers']),L2=hp.get('L2','NA'),mom2_update_weight=hp.get('mom2_update_weight','NA'),v_weight_decay=hp['v_weight_decay'],v_lr=hp['v_lr'],v_num_grad_steps=hp['v_num_grad_steps'],v_loss_layer=hp['v_loss_layer'],clamp_norm_factor=hp['clamp_norm_factor'],seed=lk['seed'],target_policy='batch entry fixed L8, once/request' if arm.startswith('BASE') else 'current W at each selected layer, once/request/layer',divisor='5,4,3,2,1' if arm.startswith('BASE') else '1',model_revision=rt['model_revision'],dtype=rt['dtype'],native_scalar_policy=rt['native_scalar_policy'],context_sha256=next(r['context_sha256'] for r in ci if r['arm']==arm)))
        compat.append(dict(arm=arm,W0_selected_match_all5_native=True,model_evaluator_runtime_equal=True,context_sha256=configs[-1]['context_sha256'],sample_root=lk['sample_root'],dataset_source_sha256=sha(lk['dataset']),shared_fixed10k_target_order='ENTRY_REQUEST_HASH_ALL100_BATCHES',historical_full_model_byte_parity='NOT_TESTED',nonselected='POINTER_VERSION_ONLY',W0_source='SH2 SEALED PUBLICATION; no remote raw/job read'))
    csvwrite(OUT/'configured-policy-comparison.csv',configs);csvwrite(OUT/'compatibility.csv',compat);csvwrite(OUT/'source-config-compatibility.csv',cfg);csvwrite(OUT/'checkpoint-tensors.csv',cp);csvwrite(OUT/'chain-integrity.csv',ci)
    compute=[];layer_summary=[];associations=[]
    for arm in ARMS:
        rr=[r for r in cost if r['arm']==arm];ss=next(r for r in summary if r['arm']==arm);cc=[r for r in cp if r['arm']==arm]
        q=dict(arm=arm,job=ss['job'],gpu_hours=float(ss['gpu_hours']),wall_seconds=float(ss['wall_seconds']),raw_bytes=int(ss['raw_bytes']),checkpoint_bytes=sum(int(r['file_bytes']) for r in {(r['batch'],r['file_sha256']):r for r in cc}.values()),peak_gpu_bytes=max(int(r['peak_gpu_bytes']) for r in rr))
        for k in ['edit_seconds','target_seconds','key_seconds','solve_seconds','evaluation_seconds','compute_z','solve_calls','history_append_passes']:q[k]=sum(float(r[k]) for r in rr)
        q['unseparated_load_snapshot_io_guard_seconds']=q['wall_seconds']-q['edit_seconds']-q['evaluation_seconds']
        q['pure_snapshot_history_seconds']='NOT_RECORDED';q['native_JVP_calls']='NOT_RECORDED; source no JV controller'
        compute.append(q)
        ll=[r for r in layers if r['arm']==arm]
        for l in sorted({int(r['layer']) for r in ll}):
            v=[r for r in ll if int(r['layer'])==l];row=dict(arm=arm,layer=l,unit='actual stored-weight batch-net Frobenius magnitude; 100 batches')
            for f in ['update_norm','magnitude_share','relative_norm']:
                row.update({f+'_'+k:x for k,x in stats([float(r[f]) for r in v]).items()})
            row['sum_batch_net_lengths']=sum(float(r['update_norm']) for r in v)
            row['path_length_share']=row['sum_batch_net_lengths']/sum(float(r['update_norm']) for r in ll)
            layer_summary.append(row)
        for scope,data,batches in [('CURRENT',current,list(range(1,101))),('CUMULATIVE',cumulative,SCHEDULE)]:
            for tag in MULT:
                for feature in ['update_norm','sum_batch_net_lengths']:
                    x=[sum(float(r[feature]) for r in ll if int(r['batch'])==b) for b in batches]
                    y=[float(next(r['rate'] for r in data if r['arm']==arm and int(r['batch'])==b and r['metric']==tag)) for b in batches]
                    from scipy.stats import spearmanr
                    rho=float(spearmanr(x,y).statistic) if np.std(x)>0 and np.std(y)>0 else 'UNDEFINED_CONSTANT'
                    associations.append(dict(arm=arm,scope=scope,metric=tag,feature=feature,n=len(x),spearman=rho,interpretation='DESCRIPTIVE_TIME_DEPENDENT; NO_CAUSALITY_NO_PVALUE'))
    csvwrite(OUT/'compute-summary.csv',compute);csvwrite(OUT/'layer-summary.csv',layer_summary);csvwrite(OUT/'mechanism-performance-associations.csv',associations)
    for r in summary:r['status']='TERMINAL_VALID_EVIDENCE_VERIFIED';r['validation']='REUSED_V3' if r['arm'] in OLDARMS else 'NEW_FULL_RAW_CP_CPU'
    csvwrite(OUT/'final-summary.csv',summary)
    for method in ['MEMIT','AlphaEdit']:
        for name in ['final-summary','cumulative-metrics','current-metrics','prompt-transitions','compute-summary','layer-summary']:
            csvwrite(OUT/(method+'-'+name+'.csv'),[r for r in csvread(OUT/(name+'.csv')) if family(r['arm'])==method])
    for name in ['w0-final-paired-transitions.csv','w0-final-paired-cohorts.csv','w0-distributions.csv','preedit-compute.csv']:
        csvwrite(OUT/name,csvread(OLD/name))
    csvwrite(OUT/'technical-exclusions.csv',csvread(INHERITED/'technical-exclusions.csv')+[dict(job='42656',status='PRE_EDIT_PENDING_CANCELLED_USER_TRANSFER_TO_SH2',canonical_denominator=0,committed_prefix_requests=0,evidence='local/fixed10k-native-baselines/attempt-v1/preedit-transfer-receipt.json')])
    findings=[]
    blue=Path(read(attempt(NEWARMS[0])/'execution.lock.json')['blue_root'])
    paths=[blue/'memit/memit_main.py',blue/'AlphaEdit/AlphaEdit_main.py',attempt(NEWARMS[0])/'lifelong/runtime.py',attempt(NEWARMS[0])/'lifelong/method.py',attempt('BASE_MEMIT')/'native/runtime.py',attempt('BASE_MEMIT')/'native/method.py',attempt('BASE_MEMIT')/'native/observation.py']
    for p in paths:
        for i,line in enumerate(p.read_text().splitlines(),1):
            if any(term in line for term in ['if hparams.blue','z_layer =','resid = targets','cache_c[i','weights[k]','w[...] +=','indices=','projector=','expected=[','signature(weights','seen-rewrite','prior=','FINAL_W0_RESTORE','mom2_update_weight * cov.double()']):
                findings.append(dict(path=str(p),sha256=sha(p),line=i,expression=line.strip()))
    csvwrite(OUT/'source-findings.csv',findings)

if __name__=='__main__':main()
