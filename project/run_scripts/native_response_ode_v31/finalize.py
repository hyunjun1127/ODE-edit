"""Compose immutable primary/diagnostic packages into a GH REVIEW_READY handoff."""
import csv
import json
import io
from pathlib import Path
from .analysis import sha,md_table,CELL_NAMES
from .provenance import save,git
from project.run_scripts.ordered_response_barrier_ode.preflight import canonical_hash


def sealed_metadata(path,value):
    """Resume metadata-only publication only when existing bytes are exact."""
    path=Path(path)
    if path.exists() or path.is_symlink():
        expected=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False,default=str)+'\n').encode()
        if path.is_symlink() or not path.is_file() or path.read_bytes()!=expected:
            raise ValueError('EXISTING_METADATA_IDENTITY_BOUNDARY')
        return
    save(path,value)


def allocation_ledger(sacct_text,expected):
    """Charge max allocation/step lifetime once, never sum overlapping steps."""
    rows=list(csv.DictReader(io.StringIO(sacct_text),delimiter='|'))
    ledger=[];totals=[0]*4
    terminal=('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED')
    for job,cells in expected.items():
        for cell in cells:
            name=f'{job}_{cell}'
            members=[r for r in rows if r['JobID']==name or r['JobID'].startswith(name+'.')]
            parents=[r for r in members if r['JobID']==name]
            if len(parents)!=1:raise ValueError('ALLOCATION_ACCOUNTING_COMPLETENESS')
            if any(not r['State'].startswith(terminal) for r in members):raise ValueError('ALLOCATION_NOT_TERMINAL')
            if 'gres/gpu=1' not in parents[0]['AllocTRES']:raise ValueError('ALLOCATION_GPU_COUNT')
            charged=max(int(r['ElapsedRaw']) for r in members)
            totals[cell]+=charged
            ledger.append(dict(job=name,child_job_id=parents[0]['JobIDRaw'],cell=cell,
                state=parents[0]['State'],allocation_elapsed_seconds=int(parents[0]['ElapsedRaw']),
                charged_seconds=charged,step_rows=members))
    if max(totals)>7200 or sum(totals)>28800:raise ValueError('GPU_BUDGET_BOUNDARY')
    return dict(attempts=ledger,cumulative_seconds_by_cell=totals,total_gpu_hours=sum(totals)/3600,
        rule='max parent/batch/extern elapsed per allocation, never sum concurrent step lifetimes',
        cap=4,max_total_seconds=28800,max_cell_seconds=7200,
        cancelled_unallocated_array_cells='zero GPU seconds; not scientific endpoints',
        source_sacct_sha256=__import__('hashlib').sha256(sacct_text.encode()).hexdigest())


def verify_package(root):
    manifest=json.loads((root/'manifest.json').read_text())
    for member in manifest['members']:
        path=root/member['path']
        if path.is_symlink() or path.stat().st_size!=member['bytes'] or sha(path)!=member['sha256']:
            raise ValueError('COMPONENT_PACKAGE_IDENTITY')
    if canonical_hash(manifest['members'])!=manifest['members_root']:raise ValueError('COMPONENT_MEMBER_ROOT')
    receipt=json.loads((root/'rooted-receipt.json').read_text())
    identity=receipt.pop('identity')
    if canonical_hash(receipt)!=identity:raise ValueError('COMPONENT_RECEIPT_IDENTITY')
    return dict(path=str(root.absolute()),report_sha256=receipt['report_sha256'],members_root=manifest['members_root'],
        manifest_sha256=sha(root/'manifest.json'),receipt_sha256=sha(root/'rooted-receipt.json'),receipt_identity=identity)


def build(root,ledger,review):
    root=Path(root);ledger=json.loads(Path(ledger).read_text());review=json.loads(Path(review).read_text())
    if (root/'manifest.json').exists():raise ValueError('CREATE_ONCE_FINAL_PACKAGE')
    if any(s>7200 for s in ledger['cumulative_seconds_by_cell']) or sum(ledger['cumulative_seconds_by_cell'])>28800:
        raise ValueError('GPU_BUDGET_BOUNDARY')
    primary=verify_package(root/'primary');diagnostics=verify_package(root/'diagnostics')
    source=json.loads((root/'primary/source.lock.json').read_text())
    rows=list(csv.DictReader((root/'primary/pilot_main_table.csv').open()))
    turn=list(csv.DictReader((root/'primary/turning_summary.csv').open()))
    table=[];contrasts=[]
    for cell in range(4):
        panel={r['arm']:r for r in rows if int(r['cell'])==cell}
        for arm in ('O_NATIVE','ORBFH_HIST','JV_NATIVE','ORB_RAY_N'):
            r=panel[arm]
            table.append(dict(cell=CELL_NAMES[cell],arm=arm,RS=f"{r['RS_n']}/{r['RS_d']}",PS=f"{r['PS_n']}/{r['PS_d']}",
                NS=f"{r['NS_n']}/{r['NS_d']}",NLL=float(r['rewrite_new_nll_mean']),
                old_loss=f"{r['new_failure_n']}/{r['entry_success_d']}",V_ratio=float(r['V_ratio']),
                QN_net=float(r['native_net_normalized']),core_seconds=float(r['write_wall_seconds'])))
        j,o,r=panel['JV_NATIVE'],panel['O_NATIVE'],panel['ORB_RAY_N']
        contrasts.append(dict(cell=CELL_NAMES[cell],JV_ray_native_action_ratio=float(j['native_net_normalized'])/float(r['native_net_normalized']),
            JV_ray_rewrite_nll_delta=float(j['rewrite_new_nll_mean'])-float(r['rewrite_new_nll_mean']),
            JV_ray_rephrase_nll_delta=float(j['rephrase_new_nll_mean'])-float(r['rephrase_new_nll_mean']),
            JV_Official_core_time_ratio=float(j['write_wall_seconds'])/float(o['write_wall_seconds']),
            native_Rturn_mean=float(turn[cell]['Rturn_mean'])))
    sealed_metadata(root/'gpu-hour-ledger.json',ledger)
    save(root/'REVIEW_READY.json',{**review,'primary':primary,'diagnostics':diagnostics,
        'primary_execution_head':source['head'],'primary_execution_tree':source['tree'],
        'final_analysis_head':git(Path.cwd(),'rev-parse','HEAD'),'final_analysis_tree':git(Path.cwd(),'rev-parse','HEAD^{tree}'),
        'gh_independent_review_status':'PENDING','scientific_promotion':False})
    save(root/'decision_summary.json',dict(status='PILOT_COMPLETE_REVIEW_READY',
        same_state_physical_turning='OBSERVED_NOT_SCALAR_ONLY',
        realized_path_change='OBSERVED_NODE_READOUT_AND_ENDPOINT_ACTION_DIFFERENCES',
        matched_quality_old_retention_benefit='NOT_DEMONSTRATED_PRIMARY_OLD_LOSS_ZERO_ALL_ARMS',
        universal_or_lifelong_claim=False,scientific_promotion=False,
        primary_endpoint_denominator='16/16 arms;160/160 request endpoints',
        total_gpu_hours=sum(ledger['cumulative_seconds_by_cell'])/3600))
    text=f'''# Native-response ODE v3.1 — B10 short-history mechanism pilot 최종 사실 보고서

상태: **REVIEW_READY / GH 독립 코드·산출물 검토 대기**. scientific_promotion=false.

본 실험은 Official D10A(B10) 한 번으로 만든 공통 warm entry에서 D10B(B10)를 독립 편집한 짧은 history pilot이다. B9 replay 또는 lifelong 재현이 아니다. Server4 진행 retention rerun의 source/process/cache/checkpoint/result를 변경하거나 그 결과를 사용하지 않았다.

## 1. 핵심 성능·action·비용

RS/PS는 target-new NLL < target-true NLL, NS는 neighborhood target-true NLL < target-new NLL이다. 모든 tie는 실패다. RS 분모10, PS20, NS100/arm/cell이며 old loss 분모는 D10A warm-entry 성공 요청만이다. NLL은 token 평균이다. `PRE_EDIT_WARM`·target-new/true 분포·strict coverage·request-paired 상세표는 [primary 보고서](primary/factual-report-ko.md)에 별도 기록했다.

{md_table(table,['cell','arm','RS','PS','NS','NLL','old_loss','V_ratio','QN_net','core_seconds'])}

## 2. Mechanism 판정: 서로 다른 질문을 합치지 않음

1. 같은 state의 joint-vs-ray native Rturn은 32/32 finite comparison에서 양수다. 단순 ray 감속만으로 동일 physical velocity가 된다는 가설은 이 기록과 맞지 않는다.
2. 실제 JV/ray trajectory의 node readout, V/V0 및 materialized endpoint native action도 다르다. 그러나 cross-arm dense endpoint distance와 Frobenius angle은 저장되지 않았으므로 그 수치는 만들지 않았다.
3. JV는 ray보다 네 cell 모두 native endpoint action이 크다. 더 작은 latent residual을 더 적은 action/해로움과 혼동하지 않는다. Primary D10A 신규 망각은 모든 arm 0/10이다. 이 표만으로 retention 우월성을 입증할 수 없다.
4. Primary JV의 rephrase target-new 평균 NLL은 네 cell 모두 ray보다 높다. 일부 PS/NS 이득과 평균 NLL 결과를 함께 보고하며 일반적인 성능 우월성을 주장하지 않는다.

{md_table(contrasts,['cell','native_Rturn_mean','JV_ray_native_action_ratio','JV_ray_rewrite_nll_delta','JV_ray_rephrase_nll_delta','JV_Official_core_time_ratio'])}

## 3. 수학·fidelity와 경계

Native metric/qN_ref/normalization은 entry에서 고정하고 매 node native full-residual direction과 JVP를 current joint state에서 재관측했다. NNLS는 nonnegative FP64 active-set reference, physical forward/overlay는 FP32이며 MEMIT stock ephemeral FP64 solve를 보존했다. G0 B1 두 fixture×4 cells=8/8 PASS; CPU algebra와 source/transaction 검증은 component receipts에 결속했다.

JV와 ORB_RAY_N은 T2/N4/h.5, historical ORBFH는 원래 ordered T1/N4/h.25이다. T 숫자를 equal exposure로 해석하지 않는다. Infinitesimal native-action dissipation 관계는 locality/retention/finite-step monotonicity 보장이 아니다.

## 4. Refinement·H10 audit

[후속 진단 보고서](diagnostics/diagnostic-factual-ko.md)는 D2 cold DEV B1 fixed-T N2/4/8의 실제 block endpoint/activation 거리와 H10의 independent warm 평가를 분리한다. 모든 미실행·실패 상태는 해당 run_registry/audit table에 기록한다. N 선택, interpolation, budget expansion 또는 H10 결과로 primary 변경은 없다.

## 5. 계산량·예산·한계

총 GPU 점유시간은 기술 시도/G0/model load/diagnostic/CPU 관측 중 GPU reservation까지 포함하여 **{sum(ledger['cumulative_seconds_by_cell'])/3600:.6f} GPUh**다. cell별 누적 seconds={ledger['cumulative_seconds_by_cell']}; 각각7200s, 총28800s 이내다. 동시성 override는 cap4이며 총 예산 확대가 아니다. job별 상세는 gpu-hour-ledger.json이다.

Arm core wall은 evaluator callback을 제외하되 dictionary/solve/shadow/transaction을 포함한다. Setup/target/model load는 별도다. 실제 model.forward/JVP 수를 제공하며 FLOPs로 환산하지 않았다. Peak GPU memory는 arm마다 reset한 값이 아니라 process 누적 high-water다. N0 clipping·gain tuning·새 reference bank·controller old-prompt access=0. Intermediate matched-progress endpoint 평가 및 setup backward ledger 누락은 primary/missing_fields.csv에 명시했다. 추가 telemetry-only rerun은 하지 않았다.

## 6. Provenance와 GH 독립 검토

Baseline report ref=f2dcfd4ab6fcb2917ae0a29cbc384cf95a82eb3c이며 historical execution HEAD와 구분한다. LM-ORBFH B9/LM-JAC B6/LA-JAC B5의 exact historical state는 입력에 없어 HISTORICAL_STATE_UNAVAILABLE, replay0이다.

Execution source={source['head']} / tree={source['tree']}. Raw source/fixture/node/endpoint 경로, 실행 대 분석 branch/worktree, code/test 파일과 locks, component SHA/member roots는 REVIEW_READY.json에 있다. 초기 local commits는 공유 clone의 GH author 설정을 상속했으며 이는 GH 코드 검토 서명이 아니다. 최종 SH1 worktree는 별도 server-head identity/access gate를 적용했고 공유 GH 설정·기존 commit identity는 변경하지 않았다.

Git 포함은 reusable code/tests와 raw-free tables/reports/locks/manifests/receipts뿐이다. Raw prompts/logs/model/weights/tensors/cache/checkpoint/dataset/credentials는 제외했다. GH 독립 검토 전 scientific_promotion=false를 유지한다.
'''
    path=root/'factual-report-ko.md'
    with path.open('x') as handle:handle.write(text)
    path.chmod(0o600)
    members=[dict(path=str(p.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size)
        for p in sorted(root.rglob('*')) if p.is_file()]
    save(root/'manifest.json',dict(members=members,members_root=canonical_hash(members),scientific_promotion=False))
    receipt=dict(status='REVIEW_READY',manifest_sha256=sha(root/'manifest.json'),members_root=canonical_hash(members),
        report_sha256=sha(path),primary_receipt_identity=primary['receipt_identity'],diagnostic_receipt_identity=diagnostics['receipt_identity'],
        scientific_promotion=False,GH_independent_review='PENDING')
    receipt['identity']=canonical_hash(receipt);save(root/'rooted-receipt.json',receipt)
    return receipt
