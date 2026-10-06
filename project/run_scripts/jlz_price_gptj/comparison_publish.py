"""GPT-J comparison publication: prior bridge method with job identity binding.
Provenance: jlz_interference_l1/comparison_bridge.py; only future GPT-J caller.
No shared helper changes or prior run rename/backfill.
"""
import math
from project.run_scripts.jlz_interference_l1.comparison_bridge import (
    read,check,digest,ENTITY,PROJECT)

def publish(self,spool):
    import wandb
    if not self.rows:return None
    tracking=read(self.out/'tracking/receipt.json');parent=tracking['run_id']
    api=wandb.Api(timeout=30);parent_run=api.run(f'{ENTITY}/{PROJECT}/{parent}')
    check(parent_run.config['source_sha']==self.b['source'] and
          str(parent_run.config['job_id'])==str(self.b['job']),'REMOTE_PARENT_IDENTITY')
    # This is a read-only mirror of the science allocation, not the collector's
    # GPU0 job. Bind the science job via the exact submission and parent run.
    job=str(self.b['job'])
    submission=read(self.root/'submission.json')
    check(str(submission['jobs'][self.cell])==job,'SUBMISSION_JOB_IDENTITY')
    identity={k:parent_run.config[k] for k in
        ('job_id','array_job_id','array_task_id','step_id','job_display_id')
        if k in parent_run.config}
    check(identity.get('job_display_id') is not None,'PARENT_JOB_DISPLAY_REQUIRED')
    identity.update(execution_backend='slurm',identity_source='VERIFIED_SCIENCE_SUBMISSION')
    display=identity['job_display_id']
    rid='comparison-'+parent
    existing=list(api.runs(f'{ENTITY}/{PROJECT}',filters={'name':rid}))
    remote={}
    for item in existing:
        check(str(item.config.get('job_id'))==job and
              item.config.get('job_display_id')==display and
              ('job'+display) in item.name,'COMPANION_JOB_IDENTITY')
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
    name=f"{self.b['writer'].upper()} {self.cell} · job{display} [comparison]"
    run=wandb.init(entity=ENTITY,project=PROJECT,id=rid,name=name,group=self.c['task_id'],
        resume='must' if existing else 'never',job_type='sealed-metric-comparison',dir=str(spool),settings=settings,
        config=dict(model_family=self.b['model'],arm=self.cell,writer=self.b['writer'],
            parent_run_id=parent,source_sha=self.b['source'],**identity,
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
    remote_run=wandb.Api(timeout=30).run(f'{ENTITY}/{PROJECT}/{rid}')
    check(('job'+display) in remote_run.name and
          all(remote_run.config.get(k)==v for k,v in identity.items()),'REMOTE_JOB_IDENTITY')
    actual={int(r['progress/batch']):r for r in remote_run.scan_history() if 'progress/batch'in r}
    check(set(actual)==set(self.rows),'REMOTE_COVERAGE')
    for n,row in self.rows.items():
        check(all(k in actual[n] and math.isclose(actual[n][k],v,rel_tol=1e-12,abs_tol=1e-9)
                  for k,v in row.items()),'REMOTE_READBACK')
    return dict(run_id=rid,parent_run_id=parent,uploaded_batch=self.last,verified=True,
        all_seen_batches=[n for n in self.rows if n in (5,10,15,20)])

