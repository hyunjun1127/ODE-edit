"""Compact OOM provenance and actual one-job registration receipt, no raw upload."""
from pathlib import Path
import json
from datetime import datetime,timezone
from official.experiments.prepare import file_sha

OUT=Path(__file__).resolve().parent
LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/qwen-zsre-sphere-oom-rerun-20261010')
OLD=Path('/mnt/raid5/janghj/ODE-edit/local/qwen-baseline-mask-cold-rerun-20261010/registration-r1')
def read(p):return json.loads(Path(p).read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')

logs=OLD/'logs/qwen25-zsre-sphere-62087.out'
lines=logs.read_text().splitlines()
first=next(line for line in lines if line.startswith('torch.OutOfMemoryError:'))
pointer=read(OLD/'runs/qwen25-zsre-sphere/checkpoint/latest.json')
assert pointer['batch']==9
receipt=dict(instruction_id='USER-SH2-QWEN-ZSRE-SPHERE-OOM-RERUN-20261010-R1',
    accepted_turn='01a122f4-e240-7532-91fd-5295daaa902a',old_job_id='62087',
    old_source='7b5097aa447946e35de42229c22b0c0feabd11ae',
    log_path=str(logs),log_sha256=file_sha(logs),first_proven_exception=first,
    failure_stage='B10 second sparse_projection -> torch.linalg.eigh(C)',
    durable_prefix=9,W20_completed=False,old_checkpoint=pointer,old_checkpoint_action='KEEP_NO_LOAD_NO_RESUME',
    source_diagnosis='Previous layer P_soft and U(view retaining full eigvecs) live during next RHS eigh',
    repaired_source_change='del upd_matrix_proj, P_soft, U immediately after selected weight addition',
    FP32_dense18944_bytes=18944**2*4,avoidable_previous_Psoft_and_eigvecs_bytes=2*18944**2*4,
    math_hparams_dtype_solver_order_unchanged=True,gradient_target_fit_preserved=True,
    CPU_tests=dict(lifetime_old_negative_control=True,lifetime_released_before_next_projection=True,
        bitwise_projected_selected_weights=True,disabled_projection_branch=True,formula_unchanged=True,
        same_sphere_config_not_FE_author=True,tests=4),
    GPU_qualification='NOT_RUN_USER_DISABLED',GPU_OOM_resolution='NOT_YET_OBSERVED',
    CF62079_action='KEEP',no_other_jobs_changed=True,monitoring_active=False,
    broadcast='NO_BROADCAST_NOT_REQUIRED',observed_utc=datetime.now(timezone.utc).isoformat())
prep=LOCAL/'registration-r1/preparation.json'
if prep.exists():receipt['preparation']=read(prep)
sub=LOCAL/'submission-complete.json'
if sub.exists():
    s=read(sub);receipt['actual_submission']=s
    receipt['submission_receipt_sha256']=file_sha(sub)
    write(OUT/'table-row.json',dict(model='Qwen2.5-7B',dataset='zsre',method='SPHERE',
        old_job_id='62087',job_id=s['job_id'],job_name=s['job_name'],state=s['initial_snapshot']['JobState'],
        reason=s['initial_snapshot']['Reason'],source=s['source'],config_sha256=s['config_sha256'],
        dependency=s['dependency'],metrics=None,W20_observed=False,qualification='NOT_RUN_USER_DISABLED'))
write(OUT/'receipt.json',receipt)
