"""Independent post-run CPU audit for frozen2632 FP32 results with Slurm step.

No edits to old source/result/tracking. Full scientific collector retained;
only environment reconstruction gains independently cross-checked step fields.
Run with PYTHONPATH pointing to the run's frozen official source package.
"""
import argparse
from pathlib import Path
from official.diagnostics.fe_qwen_current import author_basic_eot_fp32 as old
from official.runners.server1.common import read,verify,member,require
from official.experiments.prepare import write_new
from official.tracking.schema import job_identity,run_name

SCHEMA='author-basic-eot-fp32-postrun-audit-v1'
FROZEN_SHA='3b0dc0cf355b2ae21e2d57dec12264ed782ab7e979bd19b8e729aec291541180'


def crosscheck_tracking(b,runtime,proof,startup_identity,receipt):
    """Reconstruct only recorded scheduler fields; never infer/drop a step."""
    tracking=runtime['tracking'];job=job_identity(tracking)
    execution=b['execution'];require(tracking['server']==execution['server']
        and str(job['job_id'])==str(execution['job_id']),'TRACKING_EXECUTION_JOB')
    require(runtime['identity']==proof['identity']==proof['checkpoint']['identity'],'SCIENCE_RUNTIME_CP_IDENTITY')
    require(str(proof['identity']['job_id'])==str(job['job_id'])
        and proof['identity']['source']==execution['source_commit'],'SCIENCE_JOB_SOURCE')
    # Frozen2632 science identity stores job_id (not scheduler step). Its
    # immutable tracking identity and startup readback store full job identity.
    for value in (runtime['identity'],proof['checkpoint']['identity']):
        if 'job_identity' in value:require(value['job_identity']==job,'NESTED_SCIENCE_JOB_IDENTITY')
    require(startup_identity['startup_remote_identity_verified'] is True
        and startup_identity['config']==tracking and startup_identity['job_identity']==job,'IMMUTABLE_STARTUP_IDENTITY')
    startup=receipt['startup_readback']
    require(startup['status']=='READY_ONLINE' and startup['config']==tracking
        and startup['job_identity']==job and receipt['job_identity']==job,'RECEIPT_STARTUP_IDENTITY')
    require(startup_identity['run_id']==receipt['run_id']==startup['run_id']
        and startup_identity['url']==startup['url']
        and startup_identity['run_name']==startup['run_name']==run_name(tracking),'TRACKING_RUN_ID_NAME')
    final=receipt.get('result',{})
    for key,expected in (('job_identity',job),('run_id',receipt['run_id']),('run_name',startup['run_name'])):
        if key in final:require(final[key]==expected,'FINAL_TRACKING_'+key)
    if job['execution_backend']=='slurm':
        require(execution['server']=='server4','SLURM_HOST')
        env={'SLURM_JOB_ID':job['job_id']}
        for key,envkey in (('step_id','SLURM_STEP_ID'),('array_job_id','SLURM_ARRAY_JOB_ID'),('array_task_id','SLURM_ARRAY_TASK_ID')):
            if key in job:env[envkey]=job[key]
    else:
        require(job['execution_backend']=='kubernetes' and execution['server']=='rent','RENT_HOST')
        env={'ODEEDIT_RENT_JOB_ID':job['job_id']}
    require(old.tracking(b,env)==tracking,'EXACT_RECONSTRUCTED_TRACKING_CONFIG')
    return dict(job_identity=job,reconstructed_environment=env,run_id=receipt['run_id'],
        tracking_status=receipt['status'],dropped_points=receipt['dropped_points'],
        step_id_ignored=False,remote_readback_requeried=False)


def collect(bindings,output):
    out=Path(output);require(not out.exists() and not out.is_symlink(),'NEW_AUDIT_ONLY')
    inputs=read(bindings);require(inputs['schema']==SCHEMA,'AUDITOR_SCHEMA')
    require(member(old.__file__)['sha256']==FROZEN_SHA,'EXACT_FROZEN2632_COLLECTOR')
    path=verify(inputs['result']);proof=read(path)
    require(proof['status']==old.STATUS and proof['edits']==2000 and proof['batches']==20,'ACTUAL_W20')
    require(proof['source']['sha256']==FROZEN_SHA,'FP32_SOURCE')
    b=read(verify(proof['bindings']));cfg,resolved,stream=old.validate(b)
    require(proof['identity']==old.identity(b,resolved) and proof['dataset']==b['dataset'],'RESULT_BINDING')
    runtime=read(verify(proof['runtime']))
    checked=crosscheck_tracking(b,runtime,proof,read(verify(inputs['tracking_identity'])),read(verify(inputs['tracking_receipt'])))
    require(runtime['identity']==proof['identity'] and runtime['native_config']==resolved
        and runtime['model_dtype']=='torch.float32' and runtime['runtime']==b['runtime']
        and runtime['attention_backend']==b['runtime']['attention_backend']
        and runtime['tf32']==b['runtime']['allow_tf32'] and runtime['cudnn_allow_tf32']==b['runtime']['cudnn_allow_tf32'],'ACTUAL_FP32_RUNTIME')
    # Same tokenizer/query/raw independent reducer and checkpoint audit as2632.
    # No AutoModel, CUDA, forward, solver, W&B connection, or history mutation.
    import transformers
    tok=transformers.AutoTokenizer.from_pretrained(b['model']['snapshot'],trust_remote_code=True,local_files_only=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    reduced=old.common.audit_chain(proof,proof['commits'],stream,b['dataset'],tok,list(range(1,21)),('W0_first2000',2000))
    old.audit_checkpoint(proof,stream,resolved)
    report=dict(status='AUTHOR_BASIC_EOT_FP32_W20_CPU_AUDITED',audit_schema=SCHEMA,auditor_source=member(__file__),
        audit_bindings=member(bindings),result=member(path),dataset=b['dataset'],profile=old.PROFILE,
        W0=reduced[(0,'W0_first2000')],W20=reduced[(20,'all_seen/post')],commits=20,endpoints=25,recovery_claim=False,
        tracking_crosscheck=checked,tracking_identity=inputs['tracking_identity'],tracking_receipt=inputs['tracking_receipt'],
        old_in_job_collector_status='NOT_REWRITTEN_OR_RELABELLED',original_files_modified=False)
    write_new(out,report);return member(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('bindings');parser.add_argument('output')
    args=parser.parse_args();print(collect(args.bindings,args.output))
