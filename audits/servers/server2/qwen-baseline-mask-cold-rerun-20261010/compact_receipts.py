"""Read sealed local metadata only; emit compact publication, no raw/CP copying."""
from pathlib import Path
from official.experiments.prepare import read,write_new,file_sha
from official.runners.server2.qwen_mask_ft_eval import config as ft_config
from official.tracking import schema

BASE=Path('/mnt/raid5/janghj/ODE-edit/local/qwen-baseline-mask-cold-rerun-20261010')
ROOT=BASE/'registration-r1'
OUT=Path(__file__).resolve().parent
released=read(ROOT/'released.json');lock=read(ROOT/'source-lock.json')
profile=read(ROOT/'mask-profile.json');rows=[]
for j in released['jobs']:
    row={k:j[k] for k in ('job_id','cell','kind','name','dependencies','source','config_sha256')}
    row.update(state_at_release='PENDING',W_B_stage='NOT_STARTED_DEPENDENCY_PENDING',
        script_sha256=file_sha(ROOT/'scripts'/f"{j['cell']}-{j['kind']}.sh"))
    if j['kind']=='gpu' and j['cell']!='qwen25-zsre-ft-eval':
        c=read(ROOT/'configs'/f"{j['cell']}.json")
        row.update(old_job_id=profile['old_to_cells'][j['cell']],dataset=c['dataset'],method=c['method'],
            generation_schedule=c.get('generation_schedule','NOT_APPLICABLE'),
            context_sha256=None,context_status='PENDING_FIRST_ACTUAL_NATIVE_CALL',
            context_receipt=str(ROOT/'runs'/j['cell']/'native-context-identity.json'),
            checkpoint_path=str(ROOT/'runs'/j['cell']/'checkpoint'),
            source_config_file_sha256=file_sha(ROOT/'configs'/f"{j['cell']}.json"))
    elif j['cell']=='qwen25-zsre-ft-eval':
        cfg=ft_config(read(ROOT/'ft-eval-inputs.json'),lock['code_commit'],ROOT.name+'-ft-eval-job'+j['job_id'])
        # Exact runtime config projection for the known actual job; no online call.
        bound=schema.config(schema.bind_job_identity(cfg,{'SLURM_JOB_ID':j['job_id']}))
        row.update(config_sha256=cfg['config_sha'],dataset='zsre',method='FT',
            original_job_id='61900',mode='SAVED_W20_EVAL_ONLY',
            checkpoint=read(ROOT/'ft-eval-inputs.json')['row']['checkpoint'],
            config_projection=cfg,config_projection_schema='CPU_PASS_NOT_ONLINE',
            run_name=schema.run_name(bound))
    rows.append(row)
write_new(OUT/'submission.json',dict(instruction_id=profile['instruction_id'],
    accepted_turn='01a1214b-7e89-7673-90eb-8e7877f56555',
    stage='ALL_HELD_INSPECTED_RELEASED_RESOURCE_DEPENDENCY_PENDING',
    snapshot_utc=released['at'],source=lock['code_commit'],official_tree_sha256=lock['official_tree_sha256'],
    source_lock_sha256=file_sha(ROOT/'source-lock.json'),input_lock_sha256=file_sha(ROOT/'input-lock.json'),
    release_receipt_sha256=file_sha(ROOT/'released.json'),root=str(ROOT),rows=rows,
    GPU_cap=4,qualification='NOT_RUN_USER_DISABLED',CPU_tests=52,
    W_B_project_precheck=read(ROOT/'wandb-project-precheck.json')['status'],
    actual_run_remote_readback='NOT_STARTED_PENDING',
    archive='8 GPU0 one-shot KEEP receipts; receiver not bound, CF deferred consumer pending; transfer/delete0',
    old_source_raw_CP_KEEP=True,recurring_monitor=False,automatic_retry=False))
control=BASE/'control-r1';cancel=read(control/'cancellation.json')
write_new(OUT/'cancellation.json',dict(
    at=cancel['at'],cancelled=[dict(job_id=x['job']['job_id'],cell=x['job']['cell'],kind=x['job']['kind'],
        owner='janghj',node='server2',source=x['job']['source'],
        before=x['before']['fields']['JobState'],after=x['after']['fields']['JobState']) for x in cancel['cancelled']],
    kept_terminal=[dict(job_id=x['job']['job_id'],state=x['before']['fields']['JobState']) for x in cancel['kept']],
    protected_GPU=cancel['protected_GPU'],protected_eval_source_unchanged=True,
    protected_eval_resource_lanes=read(control/'protected-eval-resource-release.json')['cap4_lanes'],
    original_receipt_sha256=file_sha(control/'cancellation.json'),
    postqueue_cancelled_targets_active=0,postqueue_check='single exact-ID squeue after release returned no rows; timestamp not separately recorded',
    raw_CP_log_history_KEEP=True))
print('compact submission and cancellation receipts written')
