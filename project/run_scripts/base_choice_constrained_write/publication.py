"""CPU-only report/figures; requires already reduced, completed B1 results."""
import argparse
import ast
import csv
import io
import json
from pathlib import Path
import subprocess
from .provenance import ROOT,sha,create_bytes,create_json,now
from .review import csvout,member,read

def table(rows,columns):
    def cell(x):
        if isinstance(x,float):return f'{x:.6g}'
        return str(x).replace('|','&#124;').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(k for k in columns)+' |','| '+' | '.join('---' for _ in columns)+' |']+
        ['| '+' | '.join(cell(r.get(k,'NA')) for k in columns)+' |' for r in rows])

def conformance(source,summary):
    spec=[
        ('B1 only / sequential fail closed','config.py','require_scope','execution-entry.json; two commits', 'SOURCE_CONFIRMED_AND_RUNTIME_SCOPE'),
        ('fresh native W0/M0 once, hparams fixed','model.py','native','native/native-binding.json; terminal.native_actual_executions','SOURCE_AND_STORED_RECEIPT'),
        ('W0 raw argmax, configured EOS, 16 tokens','choice.py','raw_greedy','capsule-manifest.json; technical/base-R512/generation-TF.json','ACTUAL_BOUNDED_PARITY_AND_FULL_TF_BINDING'),
        ('all positions/full vocab; top8 diagnostic only','choice.py','scan','controller/native-reference.json; round*/reference.json','STORED_COMPLETE_POSITION_COUNTS'),
        ('two-row fresh all-token scalar gradient factors','choice.py','pair_factor','technical/pair-*-AD.json; controller/round*/factors/rows.json','ACTUAL_REFERENCE4_AD_CHECK_PLUS_RUNTIME'),
        ('actual center raw gradient RHS, positive sum','factors.py','build','round*/qp-problem.pt; factor row b/mu/raw_center_inner','CPU_FORMULA_REPLAY_AVAILABLE'),
        ('full Gram, no row subsampling','factors.py','gram','technical/actual-factor-Gram.json; round*/qp-problem.pt','ACTUAL_REFERENCE4_GRAM_AND_FULL_LOCAL_CPU_AUDIT'),
        ('minimum-norm QP / full-bank GSS, no ridge','qp.py','solve','round*/qp-solution.json; ordering-audit.json','KKT_AND_SAME_LOCAL_PROBLEM_CPU_AUDIT'),
        ('all512 exposed / retain old pairs / max2 rounds','controller.py','run','controller/selection.json; round/factor receipts','STORED_COUNTERS_AND_SOURCE'),
        ('FP32 from immutable WN; one current guard','checks.py','current_guard','current-invariant.json or exact selected=native hash','ACTUAL_RECORDED_PATH_ONLY'),
        ('official P/N, Dev after both selection seals','runner.py','run','endpoint-seals.json; arm/current.json observer receipt','SOURCE_AND_RUNTIME_NONMUTATION'),
        ('wholebatch native history exactly once / atomic CP','transaction.py','commit','arms/*/checkpoint.pt; commit.json; reload-forward.json','CPU_W_M_HASH_AND_RUNTIME_PHYSICAL_RELOAD'),
    ]
    rows=[]
    for rule,file,function,evidence,level in spec:
        p=source/'project/run_scripts/base_choice_constrained_write'/file
        tree=ast.parse(p.read_text());node=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name==function)
        rows.append(dict(requirement=rule,file=file,function=function,line=node.lineno,source_SHA256=sha(p),
            evidence=evidence,verification_level=level,limit='not global optimality / not B2 continuation'))
    return rows

def publish(report,output,repo):
    s=read(report/'summary.json');terminal=read(output/'terminal.json');lock=read(output.parent/'execution.lock.json')
    source=Path(lock['execution']['source_root']);conf=conformance(source,s);csvout(report/'design-conformance.csv',conf)
    # Exact code-generated figures; English axis labels avoid unavailable Korean fonts.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':130})
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    x=np.arange(3)
    for arm,offset,color in [('W0',-.25,'#777777'),('N4',0,'#3568a8'),('BPCW512',.25,'#cc6b2a')]:
        y=[next(r['percent'] for r in s['final'] if r['arm']==arm and r['metric']==k) for k in ['RS','PS','NS']]
        axes[0].bar(x+offset,y,width=.24,label=arm,color=color)
    axes[0].set_xticks(x,['RS (100)','PS (200)','NS (1000)']);axes[0].set_ylabel('Canonical success (%)');axes[0].set_ylim(0,100);axes[0].legend()
    values=[next(r['seconds'] for r in s['compute'] if r['component']==k) for k in ['N4_standalone_editing','BPCW512_standalone_editing']]
    axes[1].bar(['N4','BPCW512'],values,color=['#3568a8','#cc6b2a']);axes[1].set_ylabel('Standalone editing seconds')
    axes[1].set_title(f'Ratio {s["editing_ratio"]:.3f}x; setup/observer excluded')
    png1=io.BytesIO();png2=io.BytesIO()
    for buffer in (png1,png2):fig.savefig(buffer,format='png',metadata={'Software':'BPCW CPU publication.py'})
    if png1.getvalue()!=png2.getvalue():raise ValueError('PNG_BYTE_REPRODUCTION')
    create_bytes(report/'b1-canonical-cost.png',png1.getvalue());plt.close(fig)
    selection=s['selection'];geometry=read(output/'edit-null-space.json');allowed=read(output/'allowed-range.json')
    reference_meta=read(ROOT/'reference-inputs-v2/manifest.json')
    gates=table(s['gates'],['gate','name','passed','evidence'])
    counts=table(s['final'],['arm','metric','numerator','denominator','percent','delta_N4_pp','ties'])
    paired=table(s['paired'],['metric','denominator','lost','gained','net_gain','mean_new_NLL_delta','maximum_new_NLL_increase'])
    refs=table(s['references'],['arm','split','documents','positions','retained_sequences','token_flips','base_EOS_sequences','base_censored_sequences','candidate_TF_EOS_choices','mean_d'])
    strict=table(s['strict'],['arm','denominator','rewrite_strict','two_P_strict','R_two_P_strict','R_two_P_NLL_joint'])
    generation=table(s['generation'],['arm','denominator','target_prefix_match','stopped_on_original_EOS','reached_max32','target_over32_censored'])
    retention=table(s['retention'],['arm','W0_correct_N','retained','lost','retention_percent'])
    cost=table(s['compute'],['component','seconds','aggregation'])
    allocation=table([dict(job=r['JobID'],state=r['State'],exit=r['ExitCode'],start=r['Start'],end=r['End'],
        allocated_GPU_seconds=s['accounting']['allocated_GPU_seconds'],GPU_hours=s['accounting']['allocated_GPU_hours']) for r in s['accounting']['rows']],
        ['job','state','exit','start','end','allocated_GPU_seconds','GPU_hours'])
    cf=table(conf,['requirement','file','function','line','evidence','verification_level'])
    roundpath=report/'round-ledger.csv'
    roundtable=table(list(csv.DictReader(roundpath.open())),['round','rows','candidate_scanned','all_choices','token_flips','ideal_norm','actual_norm','CPU_same_problem_order_audit']) if roundpath.exists() else 'Local QP/round 없음: NOT_APPLICABLE.'
    technical=table(list(csv.DictReader((report/'technical-actual.csv').open())),['check','status','value','ceiling','scope'])
    reporttext=f'''# BPCW512 v2 — cold B1 상세 사실 보고

상태: **{s['status']} / B1 한 batch만 완료 / WAITING_USER_APPROVAL_FOR_SEQUENTIAL**.
N4/BPCW512는 Server4 동일 GPU/runtime의 W0/zeroM4에서 새로 계산한 native100 한 번을 공유했다.
사용자 별도 승인 전 B2–B10은 제출·예약·실행하지 않았다. 여섯 gate가 모두 통과하더라도 sequential 권한은 발생하지 않는다.
모델 revision `{lock['model_revision']}`, method seed20260916, FP32/eager 및 matmul/cuDNN TF32off,
physical L4 down_proj [4096,14336] 하나다. Native L2=1/lr=.1/decay=.5/clamp=.75/KL=.0625,
최대25 loss·24 Adam과 원 early-stop 수식을 유지했다. Canonical evaluator MB16과 choice MB1을 구분한다.
공통 context의 실제 token/RNG/P4 binding은 실행 lock 및 runtime-load.json에 결속했다. 같은 source라는 사실을 다른 hardware bitwise 동등성으로 확대하지 않는다.

## 1. 범위와 원분모

고정 O0 first100(100 unique requests), R100/P200/N1000. RS/PS는 new NLL < true NLL,
NS는 true NLL < new NLL이며 tie=failure다. 모든 요청과 fallback을 분모에 유지했다.
아래 값은 저장된 원 new/true NLL 쌍에서 별도 CPU reducer로 재집계했으며 실제 endpoint/state 및 case/prompt/target/order와 결속했다.

{counts}

![B1 canonical counts and cost](b1-canonical-cost.png)

R/P/NS의 반대 방향 변화를 상쇄한 단일 성공 점수는 만들지 않았다.
N4는 이번 새 native endpoint이며, 과거 EN/cold7 또는 own-trajectory shadow를 독립 대조로 대신 쓰지 않았다.

## 2. 요청별 paired 변화, strict와 W0-correct N

아래는 N4→BPCW512의 같은 문항 전이다. 동일 총점이 동일 성공 ID 집합을 뜻하지 않는다.
전체 identity별 전이는 paired-RS/PS/NS.csv, NLL 평균/분위수/tail은 NLL-distributions.csv에 보존했다.

{paired}

TF-strict와 pair-NLL 성공은 별도다. two-P는 두 paraphrase 모두 성공해야 하며 joint는 rewrite도 포함한다.

{strict}

Greedy32는 canonical TF와 별도로 관측했다. `target_over32_censored`와 실제 max32 도달은 다른 항목이다.

{generation}

W0-correct N은 같은 actual W0 관측에서 성공한 neighborhood만 조건부 분모로 사용한다.

{retention}

## 3. Reference512와 Dev128의 실제 choice

R512=S64+Reserve320+사전봉인 추가 train128, Dev128은 별도 observer다. 입력SHA `{reference_meta['inputs_sha256']}`.
추가 입력 선정의 legacy overlap 검사에 있던 미래 CounterFact/공식 P/N 경로를 모델 실행 전에 제거했다.
v1/v2 입력 bytes는 같았으나 정책·receipt는 분리 보존했다. 기존384의 역사적 sampler provenance를 지우지 않았다.
Report256는 중복 제외용 기존 입력 fingerprint만 사용했으며 답변 capsule/모델평가를 열지 않았다.

W0 deterministic raw argmax(max16), 모델 설정 EOS, 동일 W0-generated prefix의 모든 보호 위치/full vocabulary를 사용한다.
Top8은 provenance일 뿐 competitor 제한이 아니다. W0 답변을 사실 정답으로 주장하지 않는다.
실제 생성 EOS는 보호 label에 포함하며, 길이16 censor이면 가짜 EOS를 추가하지 않는다.
이번 W0 640개는 모두 censored/EOS0이므로 actual EOS 종료 branch를 관측했다고 쓰지 않는다. candidate_TF_EOS는 고정 prefix에서 EOS를 예측한 횟수이지 자유생성 종료 측정이 아니다.
아래 main metric은 fixed-prefix teacher forcing이며 자유생성 전체 유지 보장이 아니다. fixed8 greedy는 사전봉인 별도 observer다.

{refs}

확률 감쇠 d=logp_W0(base token)−logp_current(base token)는 진단량이다. d threshold/평균KL을 목적이나 선택 기준으로 추가하지 않았다.
Dev flip은 독립 observer이며 R512 조건 충족과 구별한다.
Reference와 current의 의미상 직접 fact 충돌을 전수 판정하는 semantic audit는 미측정이다.
입력·중복·split 검사는 의미상 양립성 증명이 아니며, 이번 reference 미충족을 특정 원인의 증거로 해석하지 않는다.

## 4. 실제 보정 경로와 최소거리 문제

선택 상태 `{selection['status']}`, nonlinear round {len(selection['rounds'])}, pair-scalar backward {selection['pair_scalar_gradients']},
candidate full-bank scan {selection['candidate_reference_scans']}, final current guard {selection['current_guard_count']}.
actual correction norm={selection['correction_actual_norm']:.12g}, ideal norm={selection['ideal_norm']:.12g}.
native update norm={s['native_counts']['actual_delta_norm']:.12g}; actual correction/native ratio={selection['correction_actual_norm']/s['native_counts']['actual_delta_norm']:.12g}.
selected SHA `{selection['selected_sha256']}`; native SHA `{selection['native_sha256']}`.

교정공간은 Q=V(I−JJ†)Vᵀ, J=VᵀK_E이며 D=DQ, DK_E=0이다.
K_E는 current native/canonical old+new의 전체 valid token에서 얻었다. 임의 CA/output-span 제한을 추가하지 않았다.
P_raw는 native 그대로 사용하며 P_star는 봉인된 native allowed-range provenance로 결속한다.
space status `{s['technical']['space_status']}`, allowed rank={geometry['allowed_dimension']}, blocked rank={geometry['blocked_dimension']}, q={geometry['dimension']}, actual distinct K columns={geometry['key_shape'][1]}.
rank/spectrum/ambiguity와 실제 검사는 geometry-summary.json 및 원 기술 receipt를 참조한다.
원 geometry helper의 `kind=EN-F`는 null-space 생성자의 내부 라벨이며 이번 optimizer가 EN-F라는 뜻이 아니다. 선택은 BPCW minimum-norm QP다.

각 local row는 해당 reference의 worst protected position/competitor다. Round2는 이전 pair를 유지하고 새 worst pair를 추가하며,
전체 exposed gradient를 실제 FP32 center에서 다시 계산한다. RHS는 −mu+〈raw gradient,W_center−W_N〉,
해는 **양의** dual 합 D=+Σalpha_i h_i다. Native W_N에서 FP32 materialize한다.
목적은 native로부터 Frobenius 최소 이동이며 평균KL·확률floor·새 normcap/ridge는 없다.
GSS는 full local QP의 working-set 순서이지 reference/gradient bank 축소가 아니다.
저장된 같은 local 문제의 full/most-violation CPU audit은 모델 재실험이나 nonlinear 전역 최적성 증명이 아니다.

{roundtable}

전체 factor row의 원 RHS 산술·이전 pair 유지·실제 호출 수는 mechanism-audit.json/factor-row-ledger.csv로 독립 확인했다.
QP-independent-arithmetic.csv는 저장 G/b/alpha의 raw-unit slack·dual·complementarity를 다시 계산한다.
이는 동일 solver의 row-scaled KKT certificate와 구별하며 전체 factor/D의 독립 재구성으로 확대하지 않는다.

Native 모든 choice가 안전하면 D=0으로 gradient/QP를 생략한다. 처음 reference-safe 후보에 current guard를 한 번 적용하며 실패하면 native fallback이다.
Reference 위반이 있는 fallback은 reference PASS가 아니다. Budget 종료는 전체 비선형 infeasibility 증명이 아니다.
Past는 B1 이전 요청이 없으므로 N/A0이다. 후보 중 history append0, 각 arm committed endpoint에서 native history1이다.

## 5. 설계 → 실행 코드 → 저장 증거

실행 `{s['source']}` / tree `{s['source_tree']}`. 표의 source 확인과 실제 검증 수준을 구별한다.
전체 file SHA는 design-conformance.csv에 있으며 line은 frozen source 기준이다.

{cf}

## 6. 여섯 B1 gate

{gates}

현재 gate 결과는 `{s['status']}`이다. 비용만 실패하면 B1_BUDGET_FAIL로 구분한다.
이 표는 한 cold B100의 관측이며 일반적 비열화/장기성능/우월성의 증명이 아니다. 효능 해석·후속정책 선택은 GH 영역이다.

## 7. 기술 수리와 검증 한계

최초 source dd21a22/job50291은 CPU/source 감사에서 QP near-singular 반례가 확인되어 중단했다.
이는 실제 모델 QP 실패 관측이 아니라 실행 전에 발견한 구현 오류다. 첫 job은 capsule 준비만 수행했고 nativefit0/endpoint0, 267 allocated GPU-sec였다.
완결 W0 capsule151개를 input/model/source identity로 확인해 새 attempt에서 재사용했다. 원로그·부분자료·source는 보존했다.
수정은 false infeasible 분류, zero-direction FD의 거짓 PASS 집계, actual factor/Gram 조건 누락에 한정했다.
QP 상수·수식·arm·quality ceiling을 바꾸지 않았다. 불확실한 작은 고유방향은 typed technical uncertainty이며 정상 fallback으로 숨기지 않는다.

CPU35 checks와 bounded 별도 solver 구현/감사를 수행했다. 전체 실제 모델 검증 상태는 `{s['technical']['status']}`이며
reference4/current prefix의 direct AD/FD/cache·physical/Gram 검사 범위다. 광범위 모델 미분 검증 또는 모든 reference의 direct-gradient parity로 확대하지 않는다.
0방향·미해결 rank는 FD PASS와 구별한다. B1 CP는 CPU weights_only/schema/finite/hash 및 실제 물리 weight reload와 제한된 forward parity를 확인한다.

{technical}

FD-fixed-grid.csv는 두 방향의 사전 고정12scale 전체를 포함한다. 큰 scale의 불일치를 숨기지 않았으며,
통과는 사전 규약의 인접2scale 기준이다. fixed competitor pair 미분이며 tie에서 변화하는 max-margin 미분과 다르다.
**B2 GPU continuation은 실행하지 않았다.** CPU 재적재/receipt를 새 sequential continuation PASS로 쓰지 않는다.
저장된 native WN+ideal correction의 FP32 materialization과 BPCW checkpoint의 byte equality는 CPU-endpoint-reconstruction.json으로 별도 확인했다.
이는 모든 factor에서 selected D를 독립 재계산하거나 모델 전체 forward를 replay한 검증은 아니다.

## 8. 비용·메모리·보존

{cost}

실제 shared native fit은 1회다. standalone 비교에는 같은 native 비용을 각 arm에 포함하지만 실제 연구 총비용에는 중복 청구하지 않는다.
표의 standalone view/전체wall/nested component를 서로 합산하지 않는다. Setup W0 capsule/cache와 통합기술/observer는 2× gate의 editing과 분리했다.
반면 BPCW current K/Q·native anchor·controller·전체 commit(atomic 저장·재적재 포함)은 editing에 포함했다. History/I-O는 commit 내부 timer이므로 다시 합산하지 않는다. 순수 writer는 native inclusive에서 NOT_SEPARATED다.
GPU allocated seconds는 accounting.json 원job행 기준이며 extern/batch 중복합산0; utilization으로 부르지 않는다.
peak GPU allocated={s['peak_gpu_allocated']} bytes, peak host={s['peak_host_KiB']} KiB.
실제 완료 attempt의 parent GPU allocation은 아래와 같다. 첫 취소267초와 별도이며 이미 합산한 값이 아니다.

{allocation}

Forward/token/교사강제/두-logit backward의 실제 계수는 runtime-work-counters.json에 보존했다. 이 계수에는 명시된 기술·관측 경로가 포함될 수 있어 phase별 timer와 중복 합산하지 않는다.
두 W/M/RNG/context/ledger/registry atomic B1 checkpoint를 보존한다. 원자료 actual bytes와 SHA는 raw-inventory.json, CPU tensor 검산은 checkpoint-inventory.csv에 있다.
전체 pretrained/Jacobian/512개 dense W-gradient를 저장하지 않았다. Gradient는 pair activation/projected-key factors와 bounded 기술4개 dense 증거로 구분했다.

## 9. 기존 EN 정리와 권한 경계

새 제출 전에 EN50071_0/_2/_3을 exact ID로 취소하고 terminal/resource release를 확인했다. 완료된 EN-F50071_1은 재취소하지 않았다.
사용자 지시에 따라 EN checkpoint52개(29,480,834,368 logical bytes)를 원위치 영구삭제했다.
공용 model/tokenizer/dataset/P/stats/context/reference/teacher/native target·key·공용 capsule 및 평가·생성·로그·source·receipt는 보존했다.
삭제 파일의 별도 복구사본은 확인하지 않았으며 기존 EN exact checkpoint resume은 불가하다.
새 BPCW 두 checkpoint는 삭제 대상이 아니다. 공유filesystem free delta를 독점 해제량으로 주장하지 않았다.

## 10. 재현·종료

원 scientific output: `{output}`. Lock SHA `{s['lock']['sha256']}`.
아래는 이미 완료된 원자료의 CPU 재집계 명령이다. 새 GPU 실행 명령이 아니다.

```bash
python -B -m project.run_scripts.base_choice_constrained_write.review --output {output} --accounting <saved-accounting.json> --report <new-empty-review-directory>
python -B -m project.run_scripts.base_choice_constrained_write.publication --output {output} --report <review-directory> --repo <analysis-worktree>
```

Report256 미개봉, 새로운 arm/order/후속job0. `max_batches=1`, `sequential_authorized=false` 유지.
최종 상태 **WAITING_USER_APPROVAL_FOR_SEQUENTIAL**, monitoring_active=false, automatic_resume=false.
'''
    create_bytes(report/'diagnostic-report-ko.md',reporttext.encode())
    create_bytes(report/'report-ko.md',b'# BPCW512 B1 canonical report\n\n[Detailed Korean factual report](diagnostic-report-ko.md)\n\nSequential is not authorized.\n')
    create_json(report/'b1-gate.json',dict(status=s['status'],gates=s['gates'],
        endpoints={a:terminal['commits'][a]['identity']['W'] for a in ['N4','BPCW512']},
        sequential_authorized=False,expansion_submitted=False,next='WAITING_USER_APPROVAL_FOR_SEQUENTIAL'))
    create_json(report/'execution-manifest.json',dict(instruction='ODEEDIT-S06-BPCW512-COLD-B1-SH4-V2',
        execution_source=s['source'],tree=s['source_tree'],archive=lock['execution']['archive'],lock=s['lock'],
        job=s['accounting']['job'],raw_output=str(output),model_revision=lock['model_revision'],
        reference_capsule_manifest=member(output/'capsule-manifest.json'),runtime_identity=member(output/'runtime-identity.json') if (output/'runtime-identity.json').exists() else 'execution-entry lock and terminal.identity',
        sample_order=lock['sample_order'],prior_source_replaced=False,teacher='new W0 deterministic answer capsules; not old distribution teacher'))
    create_json(report/'preflight.json',dict(local_execution_lock=s['lock'],
        source_resource_admission=member(output.parent/'resource-admission.json'),held_inspection=member(output.parent/'held-inspection.json'),
        prior_CPU35=member(ROOT/'CPU-preflight-r1/receipt.json'),integrated_actual=member(output/'technical/result.json'),
        limits_source=member(source/'project/run_scripts/base_choice_constrained_write/config.py'),
        actual_B1_only=True,task_concurrent_GPU=1,project_cap=2,old_EN_cleanup=member(ROOT/'en-cleanup/removal-receipt.json')))
    create_json(report/'reference-scope-ledger.json',dict(status='SEMANTIC_DIRECT_FACT_CONFLICT_NOT_ESTABLISHED',
        input_split_and_duplicate_checks='BOUND_TO_LOCKED_BUILDER',exhaustive_semantic_adjudication=False,
        confirmed_conflict_count=None,reference_reselected_after_result=False,
        limit='No inference from token-flip failure to semantic conflict; W0 answers are behavioral anchors, not external truth.'))
    csvout(report/'batch-metrics.csv',[dict(batch=1,phase='entry_W0' if r['arm']=='W0' else ('native_selected' if r['arm']=='N4' else 'selected'),
        q=geometry['dimension'],blocked_rank=geometry['blocked_dimension'],**r) for r in s['final']])
    create_bytes(report/'reference-summary.csv',(report/'reference-Dev.csv').read_bytes())
    ledger=[];orders=[]
    for directory in sorted((output/'controller').glob('round*')):
        if not directory.is_dir():continue
        qp_path=directory/'qp-solution.json';infeasible=directory/'qp-local-infeasible.json'
        sol=read(qp_path) if qp_path.exists() else (read(infeasible) if infeasible.exists() else {})
        ledger.append(dict(round=directory.name,problem=member(directory/'qp-problem.pt'),
            pair_rows=member(directory/'factors/rows.json'),state=read(directory/'round.json') if (directory/'round.json').exists() else 'see selection fallback',
            local_status=sol.get('status'),row_ids=sol.get('row_ids'),working_set_history=sol.get('working_set_history'),
            diagnostics=sol.get('diagnostics'),guard=member(directory/'current-invariant.json') if (directory/'current-invariant.json').exists() else 'NOT_INVOKED'))
        audit=read(directory/'ordering-audit.json')
        orders.append(dict(round=directory.name,status=audit['status'],comparisons=audit.get('comparisons'),
            order_summaries={k:dict(status=v.get('status'),elapsed_seconds=v.get('elapsed_seconds'),diagnostics=v.get('diagnostics')) for k,v in audit.get('results',{}).items()},
            local_full_audit=member(directory/'ordering-audit.json')))
    ledger.append(dict(final_selection=selection))
    create_bytes(report/'controller-ledger.jsonl',(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in ledger)).encode())
    create_json(report/'qp-order-audit.json',dict(rounds=orders,status='NOT_APPLICABLE_NO_QP' if not orders else 'STORED_SAME_PROBLEM_CPU_AUDIT',new_GPU=0))
    create_bytes(report/'artifact-index.json',(report/'raw-inventory.json').read_bytes())
    create_json(report/'terminal.json',dict(status=s['status'],batch_count=1,unique_requests=100,arms=['N4','BPCW512'],
        raw_terminal=member(output/'terminal.json'),source=s['source'],job=s['accounting']['job'],
        checkpoints=[member(output/f'arms/{a}/checkpoint.pt') for a in ['N4','BPCW512']],
        sequential_authorized=False,monitoring_active=False,automatic_resume=False,next='WAITING_USER_APPROVAL_FOR_SEQUENTIAL'))
    create_json(report/'analysis-lineage.json',dict(time=now(),analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        sources=[member(Path(__file__).parent/n) for n in ['review.py','publication.py','test_review.py']],
        executed_source=s['source'],source_tree=s['source_tree'],new_GPU=0,model_replay=False))
    # GFM pipe/column and local-link check. Actual renderer availability is separately recorded, never assumed.
    lines=reporttext.splitlines();width=None;table_rows=0
    for line in lines:
        if line.startswith('|'):
            cols=len(line.split('|'))-2
            if width is None:width=cols
            if cols!=width:raise ValueError('MARKDOWN_TABLE_COLUMN_MISMATCH')
            table_rows+=1
        else:width=None
    import importlib.util
    renderers={m:bool(importlib.util.find_spec(m)) for m in ['markdown','markdown_it','mistune']}
    create_json(report/'publication-checks.json',dict(GFM_column_rows_checked=table_rows,internal_pipes_escaped=True,
        local_image_exists=(report/'b1-canonical-cost.png').is_file(),actual_HTML_renderer=renderers,
        actual_HTML_render='NOT_RUN_NO_INSTALLED_RENDERER' if not any(renderers.values()) else 'AVAILABLE_NOT_AUTOMATICALLY_CLAIMED',
        code_generated_PNG=True,PNG_second_render_byte_equal=True,raw_prompt_or_tensor_in_report=False,counts_from_independent_reducer=True))
    members=[member(p) for p in sorted(report.iterdir()) if p.is_file()]
    create_json(report/'manifest.json',dict(members=members,analysis_only=True,raw_modified=False))
    create_json(report/'rooted-receipt.json',dict(status=s['status'],manifest_SHA256=sha(report/'manifest.json'),
        report_SHA256=sha(report/'diagnostic-report-ko.md'),source=s['source'],lock=s['lock'],
        sequential_authorized=False,next='WAITING_USER_APPROVAL_FOR_SEQUENTIAL'))
    print(json.dumps(member(report/'diagnostic-report-ko.md'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True);a=p.parse_args();publish(a.report,a.output,a.repo)
