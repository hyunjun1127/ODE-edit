"""저장된 AlphaEdit BASE checkpoint를 사용하는 BS1 진단 명세만 생성한다."""
from __future__ import annotations
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'local/datasets/counterfact-fixed-10k-v1'
NAMESPACE = 'alpha-base-checkpoint-bs1-diagnostic-v1'
BATCHES = [10, 50, 90]
LAYERS = [4, 5, 6, 7, 8]
N_EDIT = 100
MEASURE = [0, 1, 10, 25, 50, 100]
SAVE_WEIGHT = [50, 100]
WEIGHT_SHAPE = [4096, 14336]
WEIGHT_BYTES = WEIGHT_SHAPE[0] * WEIGHT_SHAPE[1] * 4


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def write_csv(name, rows):
    with (OUT/name).open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    dp,lp=DATA/'counterfact.json', DATA/'source-sample.lock.json'
    assert sha(dp)=='3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1'
    assert sha(lp)=='a8d22c230611b6c14740d1d00658a53856bbde26d060189d5ddf30f2ffdbac92'
    records={r['case_id']:r for r in json.loads(dp.read_text())}
    lock=json.loads(lp.read_text())
    rows=sorted(lock['records'],key=lambda r:r['ordinal'])
    assert [r['ordinal'] for r in rows]==list(range(10000))
    norm=lambda x:' '.join(x.casefold().split())
    subj=lambda r:norm(records[r['case_id']]['requested_rewrite']['subject'])
    order=lambda pool,role:sorted(pool,key=lambda r:hashlib.sha256(f"{NAMESPACE}:{role}:{r['case_id']}".encode()).hexdigest())
    used_before={subj(r) for r in rows[:9000]}
    fresh=[r for r in rows[9000:] if subj(r) not in used_before]
    stream=[]; selected_subjects=set()
    for r in fresh:
        q=records[r['case_id']]
        if subj(r) in selected_subjects or len(q['paraphrase_prompts'])<2 or len(q['neighborhood_prompts'])<10:
            continue
        if q['requested_rewrite']['target_new']['str']==q['requested_rewrite']['target_true']['str']:
            continue
        stream.append(r); selected_subjects.add(subj(r))
        if len(stream)==N_EDIT: break
    assert len(stream)==N_EDIT
    stream_rows=[dict(step=i+1,case_id=r['case_id'],source_ordinal=r['ordinal'],
        subject_relation_group=r['subject_relation_group'],request_sha256=r['request_sha256'],
        raw_record_sha256=r['raw_record_sha256']) for i,r in enumerate(stream)]
    write_csv('continuation-ids.csv',stream_rows)

    panel=[]
    def add(r,role,checkpoint='all',stratum=''):
        panel.append(dict(role=role,checkpoint=checkpoint,stratum=stratum,case_id=r['case_id'],
            source_ordinal=r['ordinal'],subject_relation_group=r['subject_relation_group'],
            request_sha256=r['request_sha256'],raw_record_sha256=r['raw_record_sha256']))
    used=set(selected_subjects)
    for role,n in [('base_sensor',16),('base_observer',32)]:
        chosen=[]
        for r in order(fresh,role):
            if subj(r) in used:continue
            chosen.append(r); used.add(subj(r))
            if len(chosen)==n:break
        assert len(chosen)==n
        for r in chosen:add(r,role)
    for batch in BATCHES:
        t=batch*100
        latest={r['subject_relation_group']:r for r in rows[:t]}
        step_used=set(used)
        for stratum,lo,hi in [('oldest_quarter',0,t//4),('newest_quarter',3*t//4,t)]:
            pool=[r for r in latest.values() if lo<=r['ordinal']<hi]
            for role,n in [('history_sensor',4),('history_observer',8)]:
                chosen=[]
                for r in order(pool,role+':'+stratum):
                    sensor=int(hashlib.sha256(f'{NAMESPACE}:partition:{subj(r)}'.encode()).hexdigest(),16)%3==0
                    if sensor!=(role=='history_sensor') or subj(r) in step_used:continue
                    chosen.append(r); step_used.add(subj(r))
                    if len(chosen)==n:break
                assert len(chosen)==n
                for r in chosen:add(r,role,f'B{batch:03d}',stratum)
    sensors={r['case_id'] for r in panel if r['role'].endswith('sensor')}
    observers={r['case_id'] for r in panel if r['role'].endswith('observer')}
    assert not sensors&observers
    assert not {r['case_id'] for r in stream}&(sensors|observers)
    write_csv('panel-ids.csv',panel)

    cells=[]
    for batch in BATCHES:
        for layer in LAYERS:
            branch=f'B{batch:03d}-L{layer}'
            for i,r in enumerate(stream,1):
                cells.append(dict(cell_id=f'{branch}-S{i:03d}',branch=branch,checkpoint=f'B{batch:03d}',
                    step=i,case_id=r['case_id'],physical_layer=layer,batch_size=1,
                    parent_cell=f'{branch}-S{i-1:03d}' if i>1 else f'BASE_ALPHAEDIT:B{batch:03d}',
                    history_commit='native selected-layer append once',
                    save_actual_weight=i in SAVE_WEIGHT,status='DESIGN_ONLY'))
    assert len(cells)==1500 and len({r['cell_id'] for r in cells})==1500
    for batch in BATCHES:
        for layer in LAYERS:
            branch_cells=[r for r in cells if r['branch']==f'B{batch:03d}-L{layer}']
            assert [r['case_id'] for r in branch_cells]==[r['case_id'] for r in stream]
            assert [r['step'] for r in branch_cells if r['save_actual_weight']]==SAVE_WEIGHT
    write_csv('fit-cells.csv',cells)
    write_csv('stream-routing.csv',[{k:r[k] for k in ['branch','checkpoint','step','case_id','physical_layer']} for r in cells])

    historical=list(csv.DictReader((ROOT/'plans/global/2026-09-24-historical-update-timeaxis-v1/checkpoint-bindings.csv').open()))
    bindings=[]
    for batch in BATCHES:
        r=next(r for r in historical if r['family']=='BASE_ALPHAEDIT' and int(r['batch'])==batch)
        p=Path(r['reported_archive_path'])
        bindings.append(dict(checkpoint=f'B{batch:03d}',seen_edit_count=batch*100,reported_host=r['reported_host'],
            path=str(p),recorded_sha256=r['recorded_file_sha256'],recorded_bytes=int(r['file_bytes']),
            exists_on_current_host=p.is_file(),payload_rehashed_this_design=False))
    write_csv('checkpoint-bindings.csv',bindings)
    by_checkpoint={r['checkpoint']:r for r in bindings}
    snapshots=[]
    for cell in cells:
        if not cell['save_actual_weight']:continue
        snapshots.append(dict(snapshot_id=cell['cell_id'],branch=cell['branch'],
            checkpoint=cell['checkpoint'],step=cell['step'],last_case_id=cell['case_id'],
            parameter=f"model.layers.{cell['physical_layer']}.mlp.down_proj.weight",
            dtype='float32',shape='4096x14336',tensor_payload_bytes=WEIGHT_BYTES,
            relative_weight_path=f"weights/{cell['branch']}/S{cell['step']:03d}.pt",
            parent_checkpoint_sha256=by_checkpoint[cell['checkpoint']]['recorded_sha256'],
            status='PLANNED_NOT_SAVED'))
    assert len(snapshots)==30
    write_csv('weight-snapshots.csv',snapshots)
    model=Path('/mnt/raid5/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2')
    source=ROOT/'audits/global/2026-09-22-alphaedit-original-failure-audit/official/AlphaEdit/AlphaEdit_main.py'
    dump('asset-check.json',dict(status='LOCAL_METADATA_CHECK_ONLY',checked_date='2026-09-29',
        dataset=dict(path=str(dp),sha256=sha(dp)),sample_lock=dict(path=str(lp),sha256=sha(lp)),
        model_snapshot=dict(path=str(model),exists=model.is_dir(),four_shards_present=all((model/f'model-{i:05d}-of-00004.safetensors').is_file() for i in range(1,5)),payload_rehashed=False),
        checkpoints=bindings,native_source=dict(path=str(source),sha256=sha(source)),
        unresolved=['보관 host에서 checkpoint W/M/metadata 실제 결속','원실행 P/context/tokenizer/dependencies 결속',
                    'L2=10 singleton BS1 continuation/observer 및 실제 weight 저장 구현',
                    'restore, weight 재로드 및 zero-hook parity']))
    contract=dict(name=NAMESPACE,revision='stored_BASE_checkpoints_BS1_continuations',status='DESIGN_ONLY_NOT_A_GPU_RUNNER',
        model_revision='8afb486c1db24fe5011ec46dfbe5b5dccdb575c2',checkpoint_family='BASE_ALPHAEDIT',
        checkpoints=[f'B{b:03d}' for b in BATCHES],layers=LAYERS,batch_size=1,
        continuation=dict(length=100,same_ordered_requests_in_all_15_branches=True,
            routing='physical layer fixed within each branch',source='first 100 eligible fresh-subject records after ordinal 8999',
            no_w0_edit_replay=True,no_balanced_random_router=True,
            finite_functional_failure='keep update/history and record acquisition failure',technical_failure='stop; no silent retry or reroute'),
        writer=dict(method='native AlphaEdit singleton',blue=False,L2=10,alpha=1,
            v_num_grad_steps=25,max_adam_updates_per_fit=24,v_lr=.1,v_weight_decay=.5,
            clamp_norm_factor=.75,kl_factor=.0625,cache_template=None,
            target_layer='selected physical layer',dtype='float32',attention='eager',
            tf32_matmul=False,tf32_cudnn=True,contexts='sealed native contexts',
            history='restore all five source Grams; append selected-layer Gram once; other Grams remain unchanged',
            cross_layer_handoff='not performed in these fixed-layer branches'),
        measurements=dict(detailed_steps=MEASURE,per_step='native traces + current/PS + sensor entry-to-after',
            observer='fixed base/history and all continuation edits seen at milestones',
            observer_marginal='also evaluate before the milestone write, not only after',
            input_capture='same teacher-forced prefix; all relevant valid token positions'),
        panel_counts=dict(Counter(r['role'] for r in panel)),
        weight_snapshots=dict(steps=SAVE_WEIGHT,count=len(snapshots),saved_this_design=0,
            purpose='exact model reconstruction for post-hoc analysis; not editor-state resume',
            tensor='actual full selected-layer weight after commit; not factors or delta only',
            shape=WEIGHT_SHAPE,dtype='float32',tensor_bytes_each=WEIGHT_BYTES,
            total_tensor_bytes=len(snapshots)*WEIGHT_BYTES,total_tensor_gib=len(snapshots)*WEIGHT_BYTES/2**30,
            size_excludes='serialization headers, metadata and already archived parents',
            reconstruction=['load pinned original model','apply all five BASE checkpoint weights',
                            'overwrite selected-layer weight with saved tensor'],
            required_parents='retain original model payload and all three BASE checkpoint payloads',
            metadata=['branch/checkpoint/step','parameter name/shape/dtype','parent model revision and checkpoint SHA',
                      'saved file SHA','last case ID and ordered applied edit IDs','writer configuration',
                      'code hash','native context and tokenizer identity'],
            save_protocol='independent CPU tensor copy; temporary file then atomic rename',
            validation=['reload and check tensor bitwise equality','reconstructed-model output parity',
                        'all nonselected weights unchanged from parent; otherwise stop'],
            editor_state_saved=False,
            scope_limit='unsaved intermediate models require replay; exact editor resume also requires history/RNG'),
        risk_scores=['A_frozen_new_response','B_frozen_accumulated_error_alignment','C_inherited_live_key_alignment','F_small_sensor_output_change'],
        budgets=dict(branches=15,scientific_fits=1500,max_technical_duplicate_fits=2,max_total_fits=1502,
            max_target_loss_evaluations=37550,max_adam_updates=36048,max_patch_events=3,
            extra_patch_passes_per_event=3,automatic_expansion=False),
        causal_patch=dict(checkpoints=[f'B{b:03d}' for b in BATCHES],branch_layer=4,step=100,patch_layer=8,
            formula='-(W8_checkpoint - W8_original) @ (K8_step100 - K8_branch_entry)',
            arms=['native','zero_hook','targeted_patch','norm_matched_random'],
            negative_control='L8-only branch: K8 remains fixed so this key-drift patch is zero',refit=False),
        exclusions=['W0-to-checkpoint replay','policy training','residual splitting','strength grid','full SVD',
                    'full Hessian/K-FAC','R512 replacement','new paraphrase supervision'],
        result_status='NO_MODEL_FORWARD_OR_EDIT_EXECUTED')
    dump('contract.json',contract)
    dump('design-checks.json',dict(status='PASS_DESIGN_STRUCTURE_ONLY',fit_cells=len(cells),branches=15,
        cases_per_branch=dict(Counter(r['branch'] for r in cells)),same_stream_in_all_branches=True,
        unseen_subjects_through_first_9000=True,unique_continuation_subjects=len(selected_subjects),
        sensor_observer_disjoint=True,source_dataset_hash_verified=True,
        planned_weight_snapshots=len(snapshots),snapshots_saved_this_design=0,
        weight_steps_per_branch=SAVE_WEIGHT,weight_tensor_payload_bytes=len(snapshots)*WEIGHT_BYTES,
        file_sha256={name:sha(OUT/name) for name in ['continuation-ids.csv','checkpoint-bindings.csv','stream-routing.csv','panel-ids.csv','fit-cells.csv','weight-snapshots.csv','asset-check.json','contract.json','design-ko.md']}))
    print(json.dumps(dict(branches=15,edits_per_branch=100,scientific_fits=1500,
        planned_weight_snapshots=len(snapshots),weight_tensor_gib=len(snapshots)*WEIGHT_BYTES/2**30,
        checkpoints=contract['checkpoints'],continuation_first_ids=[r['case_id'] for r in stream[:8]],status=contract['status']),ensure_ascii=False))


if __name__=='__main__':
    main()
