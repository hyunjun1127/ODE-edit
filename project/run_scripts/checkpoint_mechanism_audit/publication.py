"""CPU-only terminal collection for this failed C01 audit, not a GPU runner.

Requires actual terminal gate receipts and an explicitly collected scheduler
receipt. Does not retry failed parity or make missing measurements zero.
"""
import argparse
import csv
from pathlib import Path
from .common import *
from .model_gate import verified_reuse


def bound(path):
    p=Path(path)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p))


def collect(output,scheduler):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    results=ATTEMPT/'results';old=results/'model-gate-r1';gate=results/'model-gate-tech-r1'
    t=read(gate/'terminal.json')
    assert t['status']=='FAILED' and t['failure_type']=='NUMERICAL_CONTRACT_UNRESOLVED'
    assert t['W0_restore'] and t['nonselected_pointer_versions'] and not t['checkpoint_saved']
    for m in t['members']:
        assert bound(m['path'])==m
    verified_reuse(ATTEMPT/'receipts/gate-technical-r1-reuse.json')
    e=read(gate/'evaluation-parity.json');r=read(gate/'reconstruction-parity.json')
    assert e['archive']['NS']['passed'] is False and r['M1_gram']['passed'] is False
    assert read(old/'C00.json')['status']=='PASS'
    jobs=read(scheduler)['jobs'];assert {j['job'] for j in jobs}=={'51071','51116'}
    assert all(j['state'] in ('FAILED','COMPLETED') and j['allocated_gpus']==1 for j in jobs)
    ready=read(ATTEMPT/'inputs/READY.json')
    assert ready['status']=='VERIFIED_SOURCE_KEEP' and ready['source_map_sha256']==sha256(ATTEMPT/'inputs/source-map.json')
    # Current immutable source/CP/model stat evidence, no new multi-GB rehash.
    source=read(ATTEMPT/'inputs/source-ready.json')
    for m in source['members']:
        assert Path(m['path']).stat().st_size==m['bytes'] and sha256(m['path'])==m['sha256']
    for m in read(results/'geometry/checkpoint-input-verification.json'):
        now=stat_identity(m['path'])
        assert m['full_sha256'] and all(now[k]==m[k] for k in ('bytes','mtime_ns','inode','device'))
    model_assets=read(ATTEMPT/'inputs/model-asset-verification.json')
    for m in model_assets['members']:
        now=stat_identity(m['path'])
        assert m['full_sha256'] and all(now[k]==m[k] for k in ('bytes','mtime_ns','inode','device'))
    cells=list(csv.DictReader((REPO/'plans/global/2026-09-20-server2-checkpoint-mechanism-audit-cells-v1.csv').open()))
    pass_evidence={
        'A00':ATTEMPT/'inputs/READY.json',
        'A01':results/'archival/archival-receipt.json',
        'B00':results/'geometry/geometry-receipt.json',
        'C00':gate/'C00.json'}
    statuses=[]
    for c in cells:
        key=c['cell_id']
        if key in pass_evidence:
            statuses.append(dict(cell_id=key,status='PASS',reason='Verified independent component receipt; not C01 promotion',evidence_path=str(pass_evidence[key]),evidence_sha256=sha256(pass_evidence[key])))
        elif key=='C01':
            statuses.append(dict(cell_id=key,status='FAILED',reason='NS row NLL/margin and four M1 Gram elements exceed unchanged contract; other B1 subchecks retained',evidence_path=str(gate/'terminal.json'),evidence_sha256=sha256(gate/'terminal.json')))
        elif key=='Z00':statuses.append(dict(cell_id=key,status='PASS',reason='Terminal collection report only; dependent measurements remain BLOCKED'))
        else:statuses.append(dict(cell_id=key,status='BLOCKED',reason='Dependency C01 NUMERICAL_CONTRACT_UNRESOLVED; not submitted; no numerical fallback',dependency='C01'))
    write_json(out/'cell-statuses.json',dict(cells=statuses))
    parity=dict(status=t['status'],contract=bound(CONTRACT_PATH),original_gate_source='3f65d1705ff69fe1d53c8b5e30313265889d5c2a',
        continuation_source='1eae5defd3dfcdcec28ba613e7b7951bc0c4c8a7',C00=read(gate/'C00.json'),evaluation=e,reconstruction=r,
        projector_identity_rca=read(ATTEMPT/'receipts/projector-identity-rca-r1.json'),W0_restore=True,
        nonselected_guard='all parameter pointer/version; not full bytes',threshold_changed=False,original_failure=bound(old/'failure.json'),terminal=bound(gate/'terminal.json'))
    archive=read(results/'archival/archival-receipt.json');geometry=read(results/'geometry/geometry-receipt.json')
    timing=dict(allocated_gpu_seconds=sum(j['elapsed_seconds']*j['allocated_gpus'] for j in jobs),jobs=jobs,
        model_original_wall_seconds=read(old/'failure.json')['elapsed_seconds'],model_continuation_wall_seconds=t['elapsed_seconds'],
        archival_cpu_seconds=archive['wall_seconds'],geometry_cpu_seconds=geometry['seconds'],
        checkpoint_hash_seconds_in_geometry=geometry['hash_seconds'],model_fullhash_seconds=model_assets['seconds'],
        peak_gpu_bytes_continuation=t['peak_gpu_bytes'],original_peak_gpu_bytes='NOT_RECORDED',
        compute_and_nested_timers_not_added_to_allocation=True,all_model_backward_calls=0,
        uninstrumented_forward_token_work='NOT_RECORDED',teacher_or_new_target_optimization=0,operator_extension_seconds='NOT_RUN')
    source_bindings=[dict(kind='immutable_native_source',**m) for m in source['members']]
    for name,path in [('input_map',ATTEMPT/'inputs/source-map.json'),('input_ready',ATTEMPT/'inputs/READY.json'),
                      ('contract',CONTRACT_PATH),('gate_execution_lock',ATTEMPT/'execution-gate-r1/execution.lock.json'),
                      ('continuation_execution_lock',ATTEMPT/'execution-gate-tech-r1/execution.lock.json'),
                      ('model_fullhash_receipt',ATTEMPT/'inputs/model-asset-verification.json'),
                      ('checkpoint_fullhash_receipt',results/'geometry/checkpoint-input-verification.json'),
                      ('reuse_manifest',ATTEMPT/'receipts/gate-technical-r1-reuse.json'),('scheduler_receipt',Path(scheduler))]:
        source_bindings.append(dict(kind=name,**bound(path)))
    artifact_index=[]
    local_names={'request_registry.csv','functional_long.parquet','paired_transitions.csv','at_write_outcomes.csv','mechanism_panel.csv'}
    for name in CONTRACT['required_outputs']:
        if name in local_names:
            p=results/'archival'/name;artifact_index.append(dict(required_output=name,status='MEASURED_LOCAL_RETAINED',**bound(p)))
        elif name=='checkpoint_geometry.csv':
            p=results/'geometry'/name;artifact_index.append(dict(required_output=name,status='MEASURED_LOCAL_RETAINED',**bound(p)))
        elif name in ('fixed_probe_history.csv','native_write_modes.csv','reconstruction.csv','counterfactuals.csv','activation_margin.csv'):
            artifact_index.append(dict(required_output=name,status='BLOCKED_C01',path='status row in report; no fabricated numeric measurement'))
        else:artifact_index.append(dict(required_output=name,status='GENERATED_PUBLICATION',path=name))
    # B1 dense subchecks are actual evidence, but the seven-cell N001..N091
    # factor/SVD decomposition required for reconstruction.csv was NOT_RUN.
    hyp={h:dict(status='UNRESOLVED',basis='C01 실패로 사전 지정 operator/demand/activation 의존 분석 미실행. 구현 또는 작은 B1 재현 오차를 기전 증거로 대체하지 않음.',provisional=False) for h in ('H2','H3','H4')}
    hyp['H1']=dict(status='SUPPORTED',basis='이 단일 stream의 저장된 at-write/같은 cohort 전이에서 RS·PS와 NS의 시간적 양상이 다름. B100 all-case 조건부 loss는55/9993,423/19403,11547/72505이며 conflict-free 및 strict 전이를 별도 보고. 누적 history의 인과효과나 일반 capacity 판정 아님.',provisional=False)
    gate_text=f'''최초 job51071은 P tensor hash header([d,d] 대 [1,d,d]) 오류로 FAILED1:0,230 allocated GPU-sec였다. P 원 bytes/fileSHA는 일치했다. Technical continuation51116은 이 metadata 검사만 교정하고 완료된 C00·W0/B1 평가를 SHA-bound 재사용했다. 새 평가 forward로 NS 값을 다시 시도하지 않았다. Continuation scheduler COMPLETED0:0이지만 numerical terminal은 **FAILED / NUMERICAL_CONTRACT_UNRESOLVED**다.

- C00: B1/B2/geometry32 K·bare K·h0 및 반복성 PASS.
- B1 성공 수: RS100/100,PS190/200,NS867/1000; 모든 success bit가 archive와 일치. W0 first100은5/100,20/200,886/1000이며10k W0로 확대하지 않는다.
- NS row237 최대 NLL 차이={e['archive']['NS']['max_nll_absolute_error']:.17g}, margin 차이={e['archive']['NS']['max_margin_absolute_error']:.17g} > 고정1e-4. 모든 repeat row spread=0.
- M1 Gram:205,520,896원소 중4원소가1e-6+1e-5|ref| 초과, 최대 허용치 비율={r['M1_gram']['max_tolerance_ratio']:.9g}. 전체 최대 절대오차와 최대 비율 위치는 다르며 원JSON에 둘 다 있다.
- 실제 delta 상대오차={r['actual_delta']['max_relative_error']:.9g}, 요청별 response 최대={r['actual_response']['max_relative_error']:.9g} <1e-3. FP32 dense solve의 FP64 검산 최대 RHS 잔차={r['dense_solve']['residual']['max_relative']:.9g} <1e-5. 재solve 차이0; 실제 block affine와 key invariant PASS.
- W0 복원 및 nonselected parameter pointer/version guard PASS. Nonselected 전체 byte hash는 NOT_CLAIMED.

이 수치는 B1 부분 재현 근거이지 C01 전체 PASS 또는 H2–H4 검증이 아니다. CPU/GPU GEMM reduction 및 원 실행 hardware 차이는 가능한 설명일 뿐 원인 확정이나 tolerance 변경 근거가 아니다. C01 의존25개 cell은 BLOCKED이며 두 lane을 채우려 중복/무효 분석을 실행하지 않았다. 두 번째 GPU 사용량0. 첫 gate 실패를 성능 실패나 single-layer capacity 불가능성으로 해석하지 않는다. Numerical threshold 변경은 새 승인 없이 하지 않는다.'''
    ledger=[dict(stage='original_gate',job='51071',scheduler_state=jobs[0]['state'],allocated_gpu_seconds=jobs[0]['elapsed_seconds'],runner_seconds=read(old/'failure.json')['elapsed_seconds'],numerical_status='FAILED_TECHNICAL_P_HEADER; NS parity FAILED'),
            dict(stage='technical_continuation',job='51116',scheduler_state=jobs[1]['state'],allocated_gpu_seconds=jobs[1]['elapsed_seconds'],runner_seconds=t['elapsed_seconds'],numerical_status='FAILED_NUMERICAL_CONTRACT')]
    context=dict(scientific_promotion=False,hypotheses=hyp,gate_summary_ko=gate_text,cost_ledger=ledger,timing=timing,
        numerical_parity=parity,source_bindings=source_bindings,artifact_index=artifact_index,
        actual_runtime=read(gate/'runtime.json'),save_checkpoints=False,red_audit='owner audit + independent CPU reducers; no independent model-red PASS claimed',
        completion='TERMINAL_REPORT_WITH_NUMERICAL_BLOCKS',source_map_sha256=ready['source_map_sha256'])
    write_json(out/'report-context.json',context)
    write_json(out/'collection-receipt.json',dict(status='PASS_TERMINAL_COLLECTION_NOT_SCIENCE',scheduler=bound(scheduler),
        context=bound(out/'report-context.json'),cells=bound(out/'cell-statuses.json'),input_component_count=ready['members'],
        checkpoint_and_model_fullhash_receipts_reused_after_stat_check=True,other_task_access=False))
    print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--scheduler-receipt',required=True)
    a=p.parse_args();collect(a.output,a.scheduler_receipt)
