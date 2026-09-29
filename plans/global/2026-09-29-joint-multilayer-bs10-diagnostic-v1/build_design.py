"""BS10 다층 공동 편집 진단의 입력/명세를 생성한다. 모델 실행기는 아니다."""
from __future__ import annotations
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'local/datasets/counterfact-fixed-10k-v1'
NAME = 'joint-multilayer-bs10-diagnostic-v1'
CHECKPOINTS = [10, 50, 90]
LAYERS = [4, 5, 6, 7, 8]
ARMS = ['NATIVE', 'JOINT_STEP', 'JOINT_CUM']
N_EDIT, BATCH_SIZE = 500, 10
N_BATCH = N_EDIT // BATCH_SIZE
EVALUATE = [0, 1, 5, 10, 25, 50]
SAVE = [25, 50]
MODEL_REV = '8afb486c1db24fe5011ec46dfbe5b5dccdb575c2'
WEIGHT_BYTES = 4096 * 14336 * 4


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def csv_out(name, rows):
    with (OUT/name).open('w', encoding='utf-8', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    dataset=DATA/'counterfact.json'
    lock_path=DATA/'source-sample.lock.json'
    assert sha(dataset)=='3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1'
    assert sha(lock_path)=='a8d22c230611b6c14740d1d00658a53856bbde26d060189d5ddf30f2ffdbac92'
    records={r['case_id']:r for r in json.loads(dataset.read_text())}
    lock=json.loads(lock_path.read_text())
    source=sorted(lock['records'], key=lambda r:r['ordinal'])
    assert [r['ordinal'] for r in source]==list(range(10000))
    norm=lambda s:' '.join(s.casefold().split())
    subject=lambda r:norm(records[r['case_id']]['requested_rewrite']['subject'])
    prior_subjects={subject(r) for r in source[:9000]}
    used=set(); eligible=[]
    for r in source[9000:]:
        q=records[r['case_id']]
        if subject(r) in prior_subjects or subject(r) in used:continue
        if len(q['paraphrase_prompts'])<2 or len(q['neighborhood_prompts'])<10:continue
        if q['requested_rewrite']['target_new']['str']==q['requested_rewrite']['target_true']['str']:continue
        eligible.append(r); used.add(subject(r))
    stream=eligible[:N_EDIT]
    assert len(stream)==N_EDIT
    stream_rows=[]
    for i,r in enumerate(stream):
        stream_rows.append(dict(edit_number=i+1,batch=i//BATCH_SIZE+1,position_in_batch=i%BATCH_SIZE+1,
            case_id=r['case_id'],source_ordinal=r['ordinal'],subject_relation_group=r['subject_relation_group'],
            request_sha256=r['request_sha256'],raw_record_sha256=r['raw_record_sha256']))
    csv_out('continuation-ids.csv',stream_rows)
    batches=[dict(batch=b,case_ids=' '.join(str(r['case_id']) for r in stream_rows if r['batch']==b),
        first_edit=(b-1)*BATCH_SIZE+1,last_edit=b*BATCH_SIZE,batch_size=BATCH_SIZE) for b in range(1,N_BATCH+1)]
    csv_out('batches.csv',batches)

    panels=[]; used={subject(r) for r in stream}
    def ordered(pool, role):
        return sorted(pool,key=lambda r:hashlib.sha256(f"{NAME}:{role}:{r['case_id']}".encode()).hexdigest())
    def add(r,role,checkpoint='all',stratum=''):
        panels.append(dict(role=role,checkpoint=checkpoint,stratum=stratum,case_id=r['case_id'],
            source_ordinal=r['ordinal'],subject_relation_group=r['subject_relation_group'],
            request_sha256=r['request_sha256'],raw_record_sha256=r['raw_record_sha256']))
    for role,count in [('base_control',16),('base_observer',64)]:
        selected=[]
        for r in ordered(eligible[N_EDIT:],role):
            if subject(r) in used:continue
            selected.append(r);used.add(subject(r))
            if len(selected)==count:break
        assert len(selected)==count
        for r in selected:add(r,role)
    for ck in CHECKPOINTS:
        t=ck*100
        latest={r['subject_relation_group']:r for r in source[:t]}
        selected_subjects=set(used)
        for stratum,lo,hi in [('oldest_quarter',0,t//4),('newest_quarter',3*t//4,t)]:
            pool=[r for r in latest.values() if lo<=r['ordinal']<hi]
            for role,count in [('history_control',8),('history_observer',24)]:
                selected=[]
                for r in ordered(pool,role+':'+stratum):
                    control=int(hashlib.sha256(f'{NAME}:partition:{subject(r)}'.encode()).hexdigest(),16)%4==0
                    if control!=(role=='history_control') or subject(r) in selected_subjects:continue
                    selected.append(r);selected_subjects.add(subject(r))
                    if len(selected)==count:break
                assert len(selected)==count
                for r in selected:add(r,role,f'B{ck:03d}',stratum)
    controls={subject(r) for r in panels if r['role'].endswith('control')}
    observers={subject(r) for r in panels if r['role'].endswith('observer')}
    assert not controls&observers
    assert not {subject(r) for r in stream}&(controls|observers)
    csv_out('panel-ids.csv',panels)

    previous=list(csv.DictReader((ROOT/'plans/global/2026-09-24-historical-update-timeaxis-v1/checkpoint-bindings.csv').open()))
    bindings=[]
    for ck in CHECKPOINTS:
        r=next(r for r in previous if r['family']=='BASE_ALPHAEDIT' and int(r['batch'])==ck)
        bindings.append(dict(checkpoint=f'B{ck:03d}',seen_edits=ck*100,reported_host=r['reported_host'],
            path=r['reported_archive_path'],recorded_sha256=r['recorded_file_sha256'],
            recorded_bytes=int(r['file_bytes']),exists_on_current_host=Path(r['reported_archive_path']).is_file(),
            payload_rehashed_this_design=False))
    csv_out('checkpoint-bindings.csv',bindings)
    by_checkpoint={r['checkpoint']:r for r in bindings}
    cells=[]; snapshots=[]
    for ck in CHECKPOINTS:
        for arm in ARMS:
            trajectory=f'B{ck:03d}-{arm}'
            for b in batches:
                index=b['batch']; cell_id=f'{trajectory}-T{index:03d}'
                cells.append(dict(cell_id=cell_id,trajectory=trajectory,checkpoint=f'B{ck:03d}',arm=arm,
                    batch=index,batch_size=BATCH_SIZE,case_ids=b['case_ids'],physical_layers='4 5 6 7 8',
                    parent=f'{trajectory}-T{index-1:03d}' if index>1 else f'BASE_ALPHAEDIT:B{ck:03d}',
                    detailed_evaluation=index in EVALUATE,save_weights=index in SAVE,status='DESIGN_ONLY'))
                if index in SAVE:
                    snapshots.append(dict(snapshot_id=cell_id,trajectory=trajectory,checkpoint=f'B{ck:03d}',arm=arm,
                        offered_batches=index,offered_edits=index*BATCH_SIZE,committed_batches='RUNTIME_RECORDED',
                        physical_layers='4 5 6 7 8',tensor_count=5,shape_each='4096x14336',dtype='float32',
                        tensor_payload_bytes=5*WEIGHT_BYTES,relative_path=f'weights/{trajectory}/T{index:03d}.pt',
                        parent_checkpoint_sha256=by_checkpoint[f'B{ck:03d}']['recorded_sha256'],status='PLANNED_NOT_SAVED'))
    assert len(cells)==450 and len(snapshots)==18
    for trajectory in {r['trajectory'] for r in cells}:
        chain=[r for r in cells if r['trajectory']==trajectory]
        assert [r['case_ids'] for r in chain]==[r['case_ids'] for r in batches]
        assert [r['batch'] for r in chain if r['save_weights']]==SAVE
    csv_out('batch-cells.csv',cells)
    csv_out('weight-snapshots.csv',snapshots)

    contract=dict(name=NAME,status='DESIGN_ONLY_NOT_A_GPU_RUNNER',checkpoint_family='BASE_ALPHAEDIT',
        checkpoints=[f'B{b:03d}' for b in CHECKPOINTS],model_revision=MODEL_REV,layers=LAYERS,
        batch_size=BATCH_SIZE,batches_per_trajectory=N_BATCH,edits_per_trajectory=N_EDIT,arms=ARMS,
        source=dict(dataset=str(dataset),sha256=sha(dataset),source_lock_sha256=sha(lock_path),
            ordered_root=lock.get('ordered_root'),eligible_fresh_unique_subjects=len(eligible),
            policy='first 500 eligible fresh-subject records after ordinal 8999; preserve source order',
            same_order_and_batch_membership_in_all_9_trajectories=True),
        arm_semantics=dict(NATIVE='native five-layer AlphaEdit continuation, no new functional guard',
            JOINT_STEP='joint writer; fixed original/valid-target references; reset functional bounds to batch-entry risk plus epsilon',
            JOINT_CUM='same joint writer; base bound fixed at trajectory entry, history bounds fixed at initial entry or acceptance'),
        geometry=dict(native_L2=10,native_blue=False,
            key_side='A_l = solve(P_l @ (K_l @ K_l.T + M_l) + 10 I, P_l @ K_l)',
            basis='Q_l = orth(A_l), thin SVD with relative singular cutoff 1e-6',
            max_rank_per_layer=10,update='DeltaW_l = U_l @ Q_l.T',
            shared_full_forward=True,forward_keys_live=True,basis_fixed_within_batch=True,
            basis_rebuilt_from_own_state_each_batch=True,free_scalar_gates=False,
            history='accepted final endpoint: all five native post-write batch key Grams appended once'),
        objective=dict(energy='0.5 * sum_l ||U_l||_F^2 / ||W0_l||_F^2',
            trust_region='sum_l ||U_l||_F^2 / ||W0_l||_F^2 <= 1e-4',
            edit_guard='each of 10 requests: mean target NLL across six native contexts <= 1.0 nat/token; canonical mean target NLL <= canonical mean original-answer NLL',
            epsilon_base_kl=0.05,epsilon_history_nll=0.1,units='nats per target token',
            numerical_feasibility_tolerance=1e-5,
            bound_calibration='predeclared diagnostic settings; no outcome-driven retuning or budget expansion'),
        joint_solver=dict(optimizer='inequality augmented Lagrangian with descent line search and fixed total trust region',
            primal_coordinates='V_l = U_l / ||W0_l||_F',
            primal_direction='negative full joint gradient normalized by max(global L2 norm, 1e-12)',
            initial_trial_step=1e-3,backtracking_factor=0.5,armijo_constant=1e-4,
            trial_projection='project full concatenated V to radius 0.01; use projected displacement in Armijo test',
            line_search_failure='record and end current primal round; continue fixed dual/round budget',
            rounds=5,max_primal_steps_per_round=8,max_primal_steps_per_batch=40,
            max_backtracking_trials_per_step=6,
            beta_schedule=[1,2,4,8,16],initial_multipliers=0,
            multiplier_state='warm-start base/history multipliers in both joint arms; current-edit multipliers reset',
            working_history='all 16 old control items plus up to 32 accepted new items with least constraint slack',
            full_history_guard='all old control items and every previously accepted new canonical request, after each round',
            outcome='smallest-energy actually feasible candidate; no feasible candidate => atomic reject all 10',
            rejection='W/M/history anchors roll back; advance to next offered batch without replacement; retain best dual estimates separately',
            repeats='no retry with another layer or relaxed bound',
            microbatch='allowed for full-loss gradient accumulation; never 10 sequential singleton commits'),
        native=dict(v_num_grad_steps=25,max_adam_updates_per_request=24,v_lr=.1,
            v_weight_decay=.5,clamp_norm_factor=.75,kl_factor=.0625,target_cache=None,
            finite_failure='keep native update and history; record failed acquisition',
            technical_nan='stop that trajectory; preserve last finite state within snapshot cap'),
        runtime=dict(dtype='float32',attention='eager',tf32_matmul=False,tf32_cudnn=True,
            contexts='sealed original native contexts',observer_exposed_to_solver=False),
        panels=dict(base_control=16,base_observer=64,old_history_control_per_checkpoint=16,
            old_history_observer_per_checkpoint=48,
            new_history_control='all accepted new canonical records; gradients use active subset, acceptance scans all',
            new_history_observer='two held-out paraphrases per offered edit; no PS gradients or selection',
            preservation_scope='functional constraints cover control pool, not all archived 1k/5k/9k edits'),
        measurements=dict(detailed_batches=EVALUATE,detailed_offered_edits=[b*10 for b in EVALUATE],
            per_batch='current acquisition + acceptance/failure + control risks/slack/duals + per-layer updates/key-response traces + solver costs',
            milestone='fixed observer before last write and after; all offered continuation canonical/PS; current neighborhood',
            reference_axes=['original to entry','entry to current','pre-batch to post-batch'],
            progress_axes=['offered requests','accepted requests','actually acquired and retained requests'],
            quality_caveat='teacher-forced pair/strict and free-generation metrics remain separate'),
        weight_snapshots=dict(offered_batches=SAVE,offered_edits=[b*10 for b in SAVE],snapshots=18,
            tensors_per_snapshot=5,actual_full_weights=True,shape=[4096,14336],dtype='float32',
            tensor_bytes_each=WEIGHT_BYTES,total_tensor_bytes=18*5*WEIGHT_BYTES,
            total_tensor_gib=18*5*WEIGHT_BYTES/2**30,saved_this_design=0,
            reconstruction='pinned original model plus all five saved MLP down_proj weights',
            metadata='source checkpoint SHA, weight SHA, offered/accepted IDs, references/anchors/duals, config/code/runtime/context identities',
            validation='independent CPU copy, atomic save, tensor reload equality, full-model output parity, all other weights unchanged',
            technical_early_stop='last finite snapshot replaces next scheduled snapshot; at most two per trajectory',
            exact_editor_resume=False),
        budgets=dict(trajectories=9,scientific_batch_attempts=450,total_offered_request_exposures=4500,
            unique_continuation_requests=500,native_batch_calls=150,native_target_fits=1500,
            joint_batch_solves=300,max_joint_primal_proposals=12000,max_joint_backtracking_trial_forwards=72000,
            max_joint_full_history_guard_passes=1500,
            max_technical_replay_batch_calls=9,technical_replay_native=3,technical_replay_joint=6,
            gpu_hours_estimated=False,automatic_expansion=False),
        damage_interpretation='BS10 and 500-edit continuations create a stress condition; larger damage than BS1 is not guaranteed',
        comparison_limit='STEP grants epsilon again each batch, CUM does not: unequal total allowance is intentional; lower damage alone is not routing superiority',
        damage_markers=dict(base_observer_kl_increase_from_entry=0.05,old_history_entry_success_loss_percentage_points=5,
            usage='descriptive onset only; never stopping or tuning criteria'),
        result_status='NO_MODEL_FORWARD_EDIT_OR_WEIGHT_SNAPSHOT_EXECUTED')
    dump('contract.json',contract)
    source_path=ROOT/'audits/global/2026-09-22-alphaedit-original-failure-audit/official/AlphaEdit/AlphaEdit_main.py'
    model=Path('/mnt/raid5/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots')/MODEL_REV
    dump('asset-check.json',dict(status='LOCAL_METADATA_CHECK_ONLY',dataset_sha256=sha(dataset),
        source_lock_sha256=sha(lock_path),native_source=dict(path=str(source_path),sha256=sha(source_path)),
        checkpoints=bindings,model_snapshot=dict(path=str(model),exists=model.is_dir(),
            four_shards_present=all((model/f'model-{i:05d}-of-00004.safetensors').is_file() for i in range(1,5)),payload_rehashed=False),
        unresolved=['server2의 BASE W/M 실제 payload 결속','원실행 P/context/tokenizer/runtime 결속',
            'BS10 joint U writer와 누적/재설정 cap solver 구현','실제 shared-weight forward 및 restore/materialization 검증',
            'weight 저장 구현과 재로드 parity']))
    artifact_names=['continuation-ids.csv','batches.csv','panel-ids.csv','checkpoint-bindings.csv',
        'batch-cells.csv','weight-snapshots.csv','contract.json','asset-check.json']
    if (OUT/'design-ko.md').is_file():artifact_names.append('design-ko.md')
    dump('design-checks.json',dict(status='PASS_DESIGN_STRUCTURE_ONLY',trajectory_counts=dict(Counter(r['trajectory'] for r in cells)),
        unique_subjects=len({subject(r) for r in stream}),same_batches_in_all_trajectories=True,
        control_observer_subject_disjoint=True,continuation_panel_subject_disjoint=True,
        source_hashes_verified=True,batch_size=BATCH_SIZE,scientific_batch_attempts=len(cells),
        logical_weight_snapshots=len(snapshots),actual_saved_weights=0,
        file_sha256={n:sha(OUT/n) for n in artifact_names}))
    print(json.dumps(dict(trajectories=9,batch_size=BATCH_SIZE,batches_each=N_BATCH,edits_each=N_EDIT,
        scientific_batch_attempts=len(cells),snapshot_count=len(snapshots),snapshot_tensor_gib=18*5*WEIGHT_BYTES/2**30,
        status=contract['status']),ensure_ascii=False))


if __name__=='__main__':
    main()
