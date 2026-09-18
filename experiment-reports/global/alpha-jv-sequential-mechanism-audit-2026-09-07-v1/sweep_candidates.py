"""Emit a proposal-only JV sweep catalog, never submits or edits a runtime."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = []


def add(name, phase, lam, total_time, steps, purpose, status='PROPOSED_NOT_SUBMITTED'):
    h = total_time / steps
    assert abs(steps*h-total_time) < 1e-12
    assert not any((r['lambda_response'],r['T'],r['N']) == (lam,total_time,steps) for r in rows)
    rows.append(dict(candidate_id=name,phase=phase,models='LLAMA_AND_QWEN',arm='JV_NATIVE',
        lambda_response=lam,T=total_time,N=steps,h=h,normalization='SOURCE_EXACT_N0',
        main_JVP_per_B100=5*steps,main_JVP_per_1000_chain=50*steps,
        compute_z_policy='PINNED_OFFICIAL_UNCHANGED_ONE_PER_REQUEST_PER_BATCH',
        baseline_P_L2_history_policy='UNCHANGED',purpose=purpose,status=status))


add('JV-BASE','reference',.1,2.,4,'현재 설정; 새 development에선 같은 entry로 기준선 생성')
for label, lam in [('001',.01),('00316',10**-1.5),('0316',10**-.5),('1',1.)]:
    add('JV-LAM-'+label,'lambda_operating_point',lam,2.,4,'T/N/N0 고정; 정상 state의 gain-cost tradeoff')
for steps in (2,8,16):
    add('JV-RES-N'+str(steps),'fixed_T_resolution',.1,2.,steps,
        '같은 T에서 Euler resolution 비교; N 증가를 exposure 증가로 혼동하지 않음',
        'CONDITIONAL_REFINEMENT_NOT_SUBMITTED' if steps==16 else 'PROPOSED_NOT_SUBMITTED')
for total_time, steps in ((1.,2),(4.,8)):
    add('JV-HOR-T'+str(int(total_time)),'fixed_h_horizon',.1,total_time,steps,
        'h=.5 고정; horizon/strength 효과를 수치 resolution과 분리')

with (ROOT/'sweep-candidates.csv').open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
policy=dict(status='CANDIDATES_ONLY_EXECUTION_HOLD',unique_configurations=len(rows),
    model_configuration_pairs=2*len(rows),new_gpu_model_slurm_actions=0,
    cpu_completed_lambda_grid=[10**(-3+k/4) for k in range(13)],
    cpu_completed_state_count=80,cpu_completed_shadow_count=1040,
    gpu_lambda_screen=[.01,10**-1.5,.1,10**-.5,1.],
    conditional_dense_refinement='lambda_dev * 10**(k/8), k=-2,-1,0,1,2; lambda_dev未選定',
    frozen_compute_z_hparams=['v_num_grad_steps','v_lr','v_loss_layer','v_weight_decay','kl_factor',
        'clamp_norm_factor','fact_token','layer_module_tmp','ln_f_module','lm_head_module',
        'target_loss','context_templates','target_layer8','tokenization/padding/position',
        'stock_early_stop_loss_0.05'],
    frozen_other_baseline=['P assets/index','native_L2','committed_history_policy','layers4..8','FULL_FP32'],
    normalization_ablation_candidates=[
        dict(id='N0',status='PRIMARY_UNCHANGED'),
        dict(id='NRMS_ENTRY',status='SEPARATE_SCIENCE_DIAGNOSIS_NOT_IMPLEMENTED_OR_SUBMITTED',
             definition='same active set; s_batch^2=mean(s_i^2); e_i=R_i/(sqrt(B_active)*s_batch); same JVP weighting'),
        dict(id='NNUM_OBSERVED_NOISE',status='CONDITIONAL_MEASURED_NOISE_REQUIRED_NOT_SUBMITTED',
             definition='define floor from independent entry readout noise; no arbitrary constant/result-selected row deletion')],
    configuration_requirements=['one immutable runtime config shared by trajectory/telemetry/locks',
        'assert T==N*h before model run; T-only JSON edit is not sufficient in current source',
        'lambda_response is NOT AlphaEdit L2 or compute-z weight_decay/kl_factor',
        'same-step h multiplies physical update/energy once; never enters NNLS objective',
        'source-exact current configuration reproduction before expanded settings'],
    interpretation=['fixed-state shadow is not actual trajectory',
        'same T refinement is not horizon extension',
        'normalization variants are different scientific ablations, not technical repair',
        'existing1000 is retrospective; development/audit identities must be newly sealed',
        'no automatic20-chain/full-lifelong submission; execution budget remains unset'])
(ROOT/'sweep-candidate-policy.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'configs':len(rows),'model_config_pairs':2*len(rows),'submission':0}))
