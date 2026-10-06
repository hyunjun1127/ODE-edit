"""Read sealed scalar observations; publish comparable W&B histories, never train.

USER 2026-10-07 explicitly approved live automatic synchronization. This finite
bridge owns separate comparison runs: it never concurrently writes a live
scientific run, queries Slurm, reads tensors, or mutates an experiment directory.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import time
from pathlib import Path
from project.run_scripts.jlz_interference_l1 import validate_rows
from project.run_scripts.jlz_realization.observe import reduce_rows

ENTITY='wkdguswns2256'
PROJECT='layer allocation'
MULT={'R':1,'P':2,'N':10}

def check(ok,reason):
    if not ok:raise ValueError(reason)

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def atomic(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n');os.replace(tmp,path)

def metric_row(prefix,groups,requests):
    result={};rates=[]
    check(set(groups)==set(MULT),'FAMILY_COVERAGE')
    for k,m in MULT.items():
        g=groups[k];n=g['denominator'];num=g['numerator'];tokens=g['desired_token_count']
        check(n==m*requests==g['strict_denominator'],'ENDPOINT_DENOMINATOR')
        check(type(num)is int and 0<=num<=n and tokens>0,'NUMERATOR')
        rate=num/n;check(abs(rate-g['rate'])<1e-12,'STORED_RATE');rates.append(rate)
        fields=dict(success_pct=100*rate,success_count=num,count=n,
            strict_acc_pct=100*g['strict_numerator']/n,
            token_acc_pct=100*g['desired_token_correct']/tokens,
            prompt_acc_pct=100*g['prompt_macro'],true_nll=g['true_nll_mean'],new_nll=g['new_nll_mean'],
            margin_true_minus_new=g['true_nll_mean']-g['new_nll_mean'])
        check(all(type(v)in(int,float) and math.isfinite(v) for v in fields.values()),'FINITE_SCALARS')
        result.update({f'{prefix}/{k}/{name}':v for name,v in fields.items()})
    result[prefix+'/success_harmonic_pct']=300/sum(1/r for r in rates) if all(rates) else 0.
    return result

def observation(folder,state,identities,ids,endpoint):
    rows=[]
    for p in sorted(folder.glob('chunk-*.json')):
        obj=read(p);check(obj['state']==state and obj['optimizer_feedback'] is False,'OBS_STATE_FEEDBACK')
        rows.extend(obj['rows'])
    summary=validate_rows(rows,identities,ids,endpoint)
    stored=read(folder/'summary.json')
    check(stored['summary']==summary and stored['no_mutation'] and stored['state']==state,'OBS_REDUCTION')
    return rows,summary

class Target:
    def __init__(self,binding):
        self.b=binding;self.root=Path(binding['attempt']);self.cell=binding['cell'];self.out=self.root/self.cell
        check(sha(self.root/'config.json')==binding['config_sha256'],'CONFIG_BINDING')
        check(sha(self.root/'execution.lock.json')==binding['lock_sha256'],'LOCK_BINDING')
        self.lock=read(self.root/'execution.lock.json');self.c=read(self.root/'config.json')
        check(self.lock['source_commit']==binding['source'],'SOURCE_BINDING')
        self.mc=self.c['models'][binding['model']]
        identity=self.mc['observer_identity'];check(sha(identity['path'])==identity['sha256'],'TOKEN_BINDING')
        self.identities=read(identity['path'])['rows'];self.packs=self.mc['packs']
        self.last=0;self.previous=None;self.previous_rng=None;self.rows={};self.hashes={}

    def collect(self):
        changed=False
        for n in range(self.last+1,21):
            folder=self.out/f'batch-{n:02d}';p=folder/'commit.json'
            if not p.exists():break
            c=read(p);ids=self.packs[n-1]['ids'];seen=[i for pack in self.packs[:n] for i in pack['ids']]
            check(c['batch']==n and c['arm']==self.cell and c['source']==self.b['source'],'COMMIT_IDENTITY')
            check(c['task']==self.c['task_id'] and c['native_pack']==self.packs[n-1]['identity'],'TASK_PACK')
            check(c['ids']==ids and c['seen_requests']==100*n and len(ids)==100,'ORDER_HORIZON')
            check(c['observer_no_mutation'] and c['history_appends']==5 and c['fit_count']==1,'SEALED_COMMIT')
            check(c['RNG_before']==c['RNG_after'],'RNG_NONMUTATION')
            if self.previous is not None:
                check(c['before']==self.previous and c['RNG_before']==self.previous_rng,'OWN_NEXT_ENTRY')
            _,pre=observation(folder/'pre',c['before'],self.identities,ids,f'B{n}_PRE')
            cumulative=n in (5,10,15,20)
            check(c['post_scope']==('ALL_SEEN' if cumulative else 'CURRENT'),'ENDPOINT_SCOPE')
            postrows,post=observation(folder/'post',c['after'],self.identities,seen if cumulative else ids,f'W{n}')
            current=reduce_rows([r for r in postrows if r['case_id'] in set(ids)])
            check(pre==c['pre'] and post==c['post'] and current==c['post_current'],'COMMIT_RAW_REDUCTION')
            row={'edits':100*n,'progress/batch':n}
            row.update(metric_row('current/pre',pre,100));row.update(metric_row('current/post',current,100))
            if cumulative:row.update(metric_row('all_seen/post',post,100*n))
            self.rows[n]=row;self.hashes[n]=sha(p);self.last=n;changed=True
            self.previous=c['after'];self.previous_rng=c['RNG_after']
        return changed

    def publish(self,spool):
        import wandb
        if not self.rows:return None
        tracking=read(self.out/'tracking/receipt.json');parent=tracking['run_id']
        api=wandb.Api(timeout=30);parent_run=api.run(f'{ENTITY}/{PROJECT}/{parent}')
        check(parent_run.config['source_sha']==self.b['source'] and
              str(parent_run.config['job_id'])==str(self.b['job']),'REMOTE_PARENT_IDENTITY')
        rid='comparison-'+parent
        existing=list(api.runs(f'{ENTITY}/{PROJECT}',filters={'name':rid}))
        remote={}
        if existing:
            check(len(existing)==1 and existing[0].config['parent_run_id']==parent,'COMPANION_IDENTITY')
            for row in existing[0].scan_history(keys=['progress/batch']):
                n=int(row['progress/batch']);check(n not in remote,'DUPLICATE_REMOTE_BATCH');remote[n]=row
            # Fetch all keys separately: W&B keys filter would drop sparse milestones.
            remote={int(r['progress/batch']):r for r in existing[0].scan_history() if 'progress/batch' in r}
            for n,row in remote.items():
                check(n in self.rows,'REMOTE_BEYOND_SEALED_PREFIX')
                check(all(k in row and math.isclose(row[k],v,rel_tol=1e-12,abs_tol=1e-9)
                          for k,v in self.rows[n].items()),'REMOTE_VALUE_CONFLICT')
        metadata=dict(uploaded_batch=self.last,completed_edits=self.last*100,
            sealed_commit_sha256=digest(self.hashes),sync_role='LIVE_COMPARISON_BRIDGE',
            uploader_finished_is_not_experiment_finished=True,
            source_terminal_observed=(self.out/'terminal.json').exists(),new_model_evaluations=0)
        missing=[n for n in self.rows if n not in remote]
        if not missing:
            existing[0].summary.update(metadata)
            return dict(run_id=rid,parent_run_id=parent,uploaded_batch=self.last,verified=True,
                all_seen_batches=[n for n in self.rows if n in (5,10,15,20)])
        settings=wandb.Settings(console='off',disable_code=True,disable_git=True,save_code=False,
            disable_job_creation=True,x_disable_meta=True,x_disable_stats=True,x_disable_machine_info=True,
            init_timeout=45)
        name=f"{self.b['writer'].upper()} {self.cell} · {self.b['job']} [comparison]"
        run=wandb.init(entity=ENTITY,project=PROJECT,id=rid,name=name,group=self.c['task_id'],
            resume='must' if existing else 'never',job_type='sealed-metric-comparison',dir=str(spool),settings=settings,
            config=dict(model_family=self.b['model'],arm=self.cell,writer=self.b['writer'],
                parent_run_id=parent,source_sha=self.b['source'],job_id=str(self.b['job']),
                comparison_set='price-first2000',comparison_role='READONLY_SCALAR_MIRROR',
                metric_unit='percent; counts and NLL unscaled',raw_model_evaluation=False,
                source_config_sha256=self.b['config_sha256'],observation_identity=self.mc['observation_identity'],
                runtime_comparison='Historical baselines may differ; same metric schema is not same runtime'))
        try:
            run.define_metric('edits')
            for prefix in ('current/*','all_seen/*','progress/*'):run.define_metric(prefix,step_metric='edits')
            for n in missing:run.log(self.rows[n])
            run.summary.update(metadata)
        finally:run.finish()
        actual={int(r['progress/batch']):r for r in wandb.Api(timeout=30).run(
            f'{ENTITY}/{PROJECT}/{rid}').scan_history() if 'progress/batch'in r}
        check(set(actual)==set(self.rows),'REMOTE_COVERAGE')
        for n,row in self.rows.items():
            check(all(k in actual[n] and math.isclose(actual[n][k],v,rel_tol=1e-12,abs_tol=1e-9)
                      for k,v in row.items()),'REMOTE_READBACK')
        return dict(run_id=rid,parent_run_id=parent,uploaded_batch=self.last,verified=True,
            all_seen_batches=[n for n in self.rows if n in (5,10,15,20)])

def main():
    p=argparse.ArgumentParser();p.add_argument('--bindings',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--publish',action='store_true')
    p.add_argument('--watch',action='store_true');a=p.parse_args()
    check(not a.watch or a.publish,'WATCH_REQUIRES_PUBLISH')
    a.out.mkdir(parents=True,exist_ok=True);cfg=read(a.bindings)
    with (a.out/'process.lock').open('a') as lockfile:
        fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
        targets=[Target(b) for b in cfg['targets']];started=time.time();results={};blocked={};synced={}
        atomic(a.out/'process.json',dict(pid=os.getpid(),started_epoch=started,bindings_sha256=sha(a.bindings),
            watch=a.watch,interval_seconds=60,max_hours=168,science_jobs_mutated=False))
        while True:
            for target in targets:
                key=target.b['writer']+':'+target.cell
                if key in blocked:continue
                try:
                    changed=target.collect()
                    if a.publish and target.last>synced.get(key,0):
                        results[key]=target.publish(a.out);synced[key]=target.last
                    elif not a.publish:results[key]={'validated_batches':target.last}
                except (ValueError,KeyError,AssertionError) as error:
                    blocked[key]=dict(status='IDENTITY_OR_METRIC_BLOCKED',type=type(error).__name__,reason=str(error)[:160])
                except Exception as error:
                    # Network/IO unavailable: preserve prefix, retry transport only;
                    # no scientific operation or scheduler retry exists here.
                    results[key]=dict(status='LOGGING_DEGRADED',error_type=type(error).__name__,validated_batches=target.last)
            finished=all(t.last==20 or (t.out/'terminal.json').exists() or t.b['writer']+':'+t.cell in blocked for t in targets)
            status=dict(results=results,blocked=blocked,updated_epoch=time.time(),finished=finished,
                watch=a.watch,no_new_evaluation=True,no_slurm=True,source_read_only=True)
            atomic(a.out/'status.json',status)
            if not a.watch or finished or time.time()-started>=168*3600:break
            time.sleep(60)
        print(json.dumps(status,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
