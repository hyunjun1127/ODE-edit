"""Create compact factual tables/report/figures from independent CPU outputs."""
from .review import *
import ast
from collections import Counter,defaultdict

def rows(name):return list(csv.DictReader((PACKAGE/name).open()))
def md(headers,data):
    def cell(x):return str(x).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+['| '+' | '.join(map(cell,r))+' |' for r in data])
def scalar(x):return not isinstance(x,(dict,list))

def evidence():
    costs=[];modes=[];factors=[];source=[]
    for r in rows('writer-events.csv'):
        costs.append(dict(scope='writer events',entry=r['entry'],branch=r['branch'],kind=r['kind'],seconds=r['seconds'],reused=r['reused'],nested='within branch program_seconds; do not add parent'))
    geo=read(ROOT/'execution/attempt-r3/geometry/cost-breakdown.json')
    for r in geo['events']:costs.append(dict(scope='geometry',kind=r['kind'],seconds=r['seconds'],reused=False,nested='runtime_seconds nested inside capture',entry=r.get('state','')))
    for e in ENTRIES:
        cc=read(WRITERS/f'W{e:03d}/components/cost.json')
        for r in cc['events']:costs.append(dict(scope='components',entry=e,kind=r['kind'],seconds=r['seconds'],reused=False,nested='within component seconds_inclusive'))
        for b in BRANCHES:
            obj=read(WRITERS/f'W{e:03d}/{b}/writer-modes.json')
            for l,v in obj.items():
                # Independent arithmetic of saved response/demand per-request vectors.
                assert np.isclose(sum(v['actual_response_energy_per_request']),v['actual_response_energy'],rtol=1e-9)
                assert np.isclose(sum(v['target_demand_energy_per_request']),v['target_demand_energy'],rtol=1e-9)
                modes.append(dict(entry=e,branch=b,layer=l,**{k:z for k,z in v.items() if scalar(z)},
                    negative_target_gain_requests=sum(z<0 for z in v['actual_gain_along_target_per_request']),
                    P_idempotence_relative=v['stored_P_numerical']['idempotence_relative_frobenius']))
                f=read(WRITERS/f'W{e:03d}/{b}/write/L{l}-factor-receipt.json')
                assert f['divisor']==9-int(l)
                factors.append(dict(entry=e,branch=b,**{k:z for k,z in f.items() if scalar(z)}))
    table('compute-components.csv',costs);table('writer-mechanism.csv',modes);table('native-factor-checks.csv',factors)
    specs=[
      ('original baseline/source/precision','runtime.py','__init__','PASS_SOURCE_AND_RUNTIME_BINDING','G0 READY; input binding; blue=False,L4..8,L2=10; TF32 matmul off/cuDNN on'),
      ('W/M/context/RNG branch restore','runtime.py','restore','PASS_RUNTIME_RECEIPTS_LIMITED','entry-terminal and observer hashes; no independent full W/M GPU continuation'),
      ('native100 once per entry / shared z only','writer_runner.py','run_writers','PASS_COUNTER_AND_ORDER','400 total target requests; r4 new300/reuse100; branch downstream fresh'),
      ('native original function instrumented not replaced','native_writer.py','write','SOURCE_PATH_AND_RECEIPTS','original_function_called; five divisors/final L8; native source retained'),
      ('same entry NATIVE vs SHAM numerical control','writer_runner.py','verify_sham','OBSERVATION_ONLY_USER_DIRECTED','all4 SHAM nonexact differences retained; not equivalence PASS'),
      ('N/P observer-only','writer_runner.py','__call__','PASS_SOURCE_AND_NONMUTATION','fixed branches; no outcome-selected subset; N512 never basis'),
      ('all512 history/no success filtering','interventions.py','_bank','PASS_SOURCE_AND_IDENTITIES','512 columns/stat weights; functional overwrite mask separate'),
      ('temporary M_eff / native persistent history','interventions.py','history_operand','PASS_SOURCE_AND_RECEIPTS','ephemeral operand; history120 appends once/layer; no M refresh'),
      ('geometry raw/center/unit/context/mean','geometry_runner.py','run_geometry','PASS_ARITHMETIC_WITH_LIMIT','33040 spectrum rows recomputed; 4 CPU tensor samples; no fresh model'),
      ('calibration512 vs assessment3488','geometry_runner.py','run_geometry','PASS_CASE_SET_CHECK','18 families x512+4x872 disjoint; no aggregate3488 Gram claimed'),
      ('upper negative control','geometry_runner.py','_upper_control','PASS_SAVED_ACTUAL','W80/W100 K6/K7 exact identity; not new forward'),
      ('component all valid tokens/full physical','component_runner.py','full_hook_parity','PASS_BOUNDED_ACTUAL_ONLY','8 small-panel full physical exact; sum route nonzero recorded'),
      ('component vs weight intervention separated','component_runner.py','run_components','PASS_SOURCE_ENDPOINT_SEALS','mean/centered inference hooks not weight methods'),
      ('rank1 direction/norm-control definition','interventions.py','weight_ablation','DEFINED_ALL8_QUALITY_NOT_MATCHED','calibration-only direction; quality matching NOT_ESTABLISHED'),
      ('rank1/2/4 key interchange','interventions.py','calibration_key_basis','DEFINED_ALL24','calibration-only; inference not writer operand intervention'),
      ('K/R same receiving W/P/M/lambda','component_runner.py','run_components','PASS_8_MATRICES','32 actual endpoints; same receiving/P/M hashes; no suffix write'),
      ('observer postseal / W/M/RNG nonmutation','component_runner.py','_seal_observe','PASS_RUNTIME_HASHES','136 explicit before/after hashes, RNG restore; no model call in review'),
      ('full vocabulary TF/NLL definition','observer.py','evaluate_pairs','PASS_RAW_REDUCER','128256 vocab metadata; canonical ties=failure; TF lowest-ID tie separately'),
      ('N512 W0-correct one-token selection','observer.py','evaluate_neighborhood512','PRIOR_SELECTION_BINDING_PLUS_CURRENT_RAW','W0 correctness selection evidence reused, current TF independently reduced'),
      ('H512 overwrite/active/superseded','observer.py','history_masks','PASS_SOURCE_MASK_ARITHMETIC','statistics512; functional masks only; target identity not success'),
      ('writer whitening algebra','writer_diagnostics.py','summarize','DIAGNOSTIC_APPROXIMATION_ONLY','stored P not exact projector; symmetrized native H, not ideal-P proof'),
      ('no new full-state CP','writer_runner.py','run_writers','PASS_DECLARED_AND_OUTPUT_INVENTORY','approved z/K/R/delta diagnostics retained; exact resume not established'),
      ('autonomous queue / followup exclusions','runner.py','main','PASS_SUBMISSION_AND_TERMINAL','94 families; 7 followups not submitted; no periodic monitor in review'),
    ]
    src=ROOT/'execution-source-r4/project/run_scripts/alpha_key_concentration_causal'
    for req,file,func,status,note in specs:
        p=src/file;tree=ast.parse(p.read_text());matches=[n.lineno for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==func]
        source.append(dict(requirement=req,frozen_source='a21cffa08d4cf86ed258acabdb786fba778f4dfb',file=str(p),function=func,
            lines=';'.join(map(str,matches)) or 'MODULE_SCOPE_FUNCTION_NAME_NOT_FOUND',sha256=sha(p),status=status,evidence_limit=note))
    table('source-conformance.csv',source)
    # Exact source identity binding; copying old giant archive/model is unnecessary.
    authority=[]
    for p in [REPO/'messages/head/2026-09-24-sh4-all-jobs-detailed-review.md',ROOT/'inputs/design/design-ko.md',ROOT/'inputs/design/contract.json',ROOT/'inputs/design/cells.csv',REPO/'PROTOCOL.md']:
        authority.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),read_level='FULL_READ_THIS_RECALL' if '2026-09-24-sh4' in p.name else 'EXACT_PRIOR_FULL_READ_REBOUND'))
    write_json(PACKAGE/'authority-binding.json',authority)

def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    data=rows('first-actual-table.csv')
    for b in BRANCHES:
        axes[0].plot(ENTRIES,[float(next(r['percent'] for r in data if r['entry']==str(e) and r['branch']==b and r['metric']=='NS')) for e in ENTRIES],marker='o',label=b)
    axes[0].set(xlabel='Approved entry checkpoint W#',ylabel='Current N preference (%)',title='Same-entry actual writer endpoints')
    axes[0].legend(ncol=2,fontsize=8)
    g=rows('geometry-independent-spectrum-arithmetic.csv')
    states=[0,10,50,70,80,90,100]
    for l in (5,6):
        for variant,style in [('raw','-'),('centered','--')]:
            vals=[float(next(r['PR'] for r in g if r['phase']=='E1' and r['state']==str(s) and r['layer']==str(l) and r['panel']=='early' and r['space']=='raw' and r['variant']==variant)) for s in states]
            axes[1].plot(states,vals,style,marker='.',label=f'L{l} {variant}')
    axes[1].set(xlabel='Approved checkpoint W#',ylabel='Participation ratio',title='Early1000 actual-mean keys')
    axes[1].legend(fontsize=8)
    fig.savefig(PACKAGE/'actual-endpoints-and-geometry.png',dpi=150,metadata={'Software':'CPU review deterministic plot'})
    plt.close(fig)

def report():
    job=rows('job-inventory.csv');first=rows('first-actual-table.csv');jointrows=rows('first-joint-table.csv')
    metrics=rows('all-endpoint-metrics.csv');pairedrows=rows('paired-transitions.csv');geo=rows('geometry-independent-spectrum-arithmetic.csv')
    def metric(e,b,k,panel='Current',stage='history'):
        return next(r for r in metrics if r['entry']==str(e) and r['branch']==b and r['metric']==k and r['panel']==panel and r['stage']==stage)
    def counts(e,b):return '/'.join(metric(e,b,k)['numerator'] for k in ('RS','PS','NS'))
    inventory=read(PACKAGE/'output-inventory-receipt.json');account=read(PACKAGE/'accounting-receipt.json')
    chunks=['''# Server4 전체 job 목록·완료 상세 사실 리뷰

Instruction/nonce: `ODEEDIT-GH-SH4-ALL-JOBS-DETAILED-REVIEW-20260924-R1`.

이번 recall은 **CPU-only 리뷰**다. 신규 GPU/model/forward/evaluator/Slurm write·수리·재제출·삭제는 0이다.
Scheduler 상태와 과학 완료/수치 동등성을 분리했다. 현재 main이 아니라 실제 frozen runtime을 검토했다.

## 1. 범위·전체 상태

2026-09-24 00:52:05 KST의 owner `janghj` queue snapshot은 비어 있었다. Server4의 2026-09-20 이후 accounting census를 제출기록과 대조하고, 실행되지 않아 node census에서 빠지는 정확 dependency job50983/52528/52529도 보완했다. 총17개 parent다. 현재 미완료 RUNNING/PENDING은 이 snapshot에서 0이며 이후 상태를 주기 조회하지 않았다.

직전 완료리뷰 이후 신규 과학 실행은 alpha-key다. SLMF 완료51058와 앞선 실패, EN-adaptive51260 실패는 기존 상세 검산/보고를 재사용하고 이번 accounting을 결속했다. 더 오래된 게시 자료97개는 [역사 보고서 SHA index](historical-report-index.csv)로 보존했다. 이것은 과거 모든 raw를 다시 계산했다는 뜻이 아니다. 타 사용자/타 서버 job은 제외했다.

''',md(['Job','이름','상태','exit:signal','GPU-sec','검토'],[[r['job_id'],r['job_name'],r['state'],r['exit_signal'],r['allocated_gpu_seconds'],r['scope']] for r in job]),'''

정확 source/tree/archive/lock/argv/dependency/start/end/resource는 [job inventory](job-inventory.csv), 조회 경계는 [accounting receipt](accounting-receipt.json)에 있다. Parent만 비용에 사용했으며 batch/extern을 더하지 않았다. CPU reducer COMPLETED는 선행 과학 실패를 지우지 않는다.

## 2. 실패·수리 lineage

| Attempt/job | 최초 원인·단계 | 수리/재사용과 실제 후속 |
| --- | --- | --- |
| r1 gate52527, 44 GPU-sec | `technical.py:200 WRITER_UNEXPECTED_BOS: 0`; 첫 G1 model forward 전 | Fast tokenizer의 `add_bos_token=False` attribute를 BOS 제거로 오인한 새 검사. 원 native tokenization은 변경하지 않음 |
| r1 geometry52528/writers52529 | 선행 gate 실패, 실제 allocation0 | 정확 dependency 취소. reducer52530은 실패 상태를 집계한 CPU 완료 |
| r2 freeze | 제어 변수 이름 충돌, CPU freeze 실패 | GPU 등록0; 별도 실패기록. 과학 재시행으로 세지 않음 |
| r3 gate52563 | native token/backend/lookup binding으로 수정 | G0/G1 PASS; 2요청×6context actual parity. native100/SHAM은 당시 NOT_RUN |
| r3 writers52565, 1494 GPU-sec | `verify_sham`의 `SHAM_NUMERICAL_CONTROL_NOT_ESTABLISHED`; W50 NATIVE/SHAM 실행·평가 뒤 exact0 비교에서 종료 | FP32 M−stampGram+stampGram 경로 roundoff. OOM/IO/성능탈락 아님 |
| r3 geometry52564/reducer52566 | geometry46 family 완료; reducer는 writer 부분완료 포함 | 성공한 geometry는 재실행하지 않음. reducer afterany가 과학 gate를 우회한 실행은 아님 |
| r4 writers52575/reducer52576 | 최신 사용자 “상대차 gate는 관찰만” 적용 | SHAM·hook verdict/차이는 보존, blocks_execution만 정책 변경. 원 interventions/runtime/native 수식 diff0. W50 NATIVE/SHAM/z100 재사용, 나머지 완결 |

실행 r1 `a95876f8e5c4cf59df9cd9d7f824d1ac99f8bc77`, r3 `f9fbd56f31b0c520763ec9026e660a76cb3074ff`, r4 `a21cffa08d4cf86ed258acabdb786fba778f4dfb`를 분리한다. R4 archive SHA `77b961e07c83cc20f2a8805de850fe33076fd673dcad76d536b9ef9eb289e9ae`, lock SHA `869f990e67b5f9697be7038b0e13de78b89a6fcdb7cd2b672217058ba1600b3b`는 기존 immutable seal과 결속했다. 이번에는 실행 source104개 member를 새로 해시했고 원 대형 archive/model/12CP 전량 재해시는 반복하지 않았다.

52565의 W50 SHAM L8 delta 상대차 약4.264e−6, Current N true-NLL 최대 차 약8.297e−5는 원 실패값이다. 이를 0이나 parity PASS로 수정하지 않았다. 기존 source/raw/cost와 실패 receipt는 보존돼 있다. 원 실패와 후속 cleanup을 혼동하지 않았고, 이번 리뷰에서 runtime 수리·rerun은 없었다.

## 3. Alpha-key 완료 범위와 gate

''',md(['Family','승인','완료 대응','검산 범위'],[['E0','기술','G0/G1 actual + G2 observer-policy 완료','CPU toy와 모델 검증 분리'],['E1',28,28,'7states×4cohorts; L4–L8'],['E2',18,18,'W80/W100×8masks+upper control'],['E3',4,4,'W50/70/80/90 native100'],['E4-H',20,20,'각 entry SHAM/H5/H6/H56/MASS56'],['E4-W',16,16,'두 upstream layer×components/matrix'],['E4-KR',8,8,'같은 receiving state의 2×2'],['SEQ/ORDER/FUTURE',7,'FOLLOWUP_NOT_SUBMITTED','범위 밖; negative result 아님']]),'''

[101개 설계행 대응](family-coverage.csv)에서 승인94와 후속7을 분리했다. 24개 writer branch, 136개 component/matrix/interchange/KR endpoint가 저장됐다. Family 하나를 job이나 단일 endpoint라고 세지 않는다. 정의 가능한 matrix8개와 rank1/2/4 basis24개가 실제 실행됐고 이 범위에서 NOT_APPLICABLE은 없었다.

G0는 입력/source/12CP 기존 fullSHA 검증을 결속했다. G1은 두 요청의 prefix/full/own/upper key 및 restore 비교이며 모든 실험에 대한 새 bitwise 증명은 아니다. G2는 W50 native100+SHAM+후속 component 완료이지만 상태는 `EXECUTION_COMPLETED_NUMERICAL_COMPARISONS_OBSERVER_ONLY`다. G3는 등록된 유한 queue/실행source/manifest다. 이전 agent 인계는 PENDING/initial 미관측이었으며, 지금 파일을 읽었다고 과거 실시간 initial 관측을 PASS로 소급하지 않는다.

## 4. 지표 정의와 최종 writer 표

Canonical RS/PS = new NLL < true NLL, NS = true NLL < new NLL, tie=failure. 분모는 각 entry마다 R100/P200/N1000이다. TF는 R/P new target, N true target을 teacher-force한 정확도다. Token-micro는 전체 target token 합, prompt-macro는 prompt별 token 정확도 평균, strict는 모든 target token 정답이다. Lowest-token-ID argmax와 unique-argmax strict를 따로 계산했다. 자유생성 accuracy가 아니다.

Raw case/prompt/index/target/token/order/endpoint/finite를 대조했으며 동일 weight SHA로 점수를 대입하지 않았다. [1,608개 전체 endpoint/panel 지표](all-endpoint-metrics.csv)에 true/new/desired NLL, margin, p01/p05/median, NLL p95/p99와 TF를 남겼다. Stage/현재문항/N512/H512/partial upstream endpoint를 서로 다른 분모로 유지한다.

''',md(['Entry→batch','NATIVE R/P/N','SHAM','H5','H6','H56','MASS56'],[[f'W{e}→B{e+1}',*[counts(e,b) for b in BRANCHES]] for e in ENTRIES]),'''

이들은 W0 first100 독립 실험이나 새로운 N4 lifelong chain이 아니다. 각각 보존 AlphaEdit W50/70/80/90에서 다음100 요청을 편집한 same-entry branch다. Entry 사이 문항/누적상태가 달라 위 행을 하나의 독립 paired 분모로 합치지 않는다.

### 4.1 TF와 joint는 preference와 다르다

''',md(['Native entry','R TF-strict/100','P TF-strict/200','N TF-strict/1000','R+twoP TF-joint/100','NLL joint/100'],[[e,*[metric(e,'NATIVE',k)['strict_count'] for k in ('RS','PS','NS')],next(r['TF_R_twoP_joint'] for r in jointrows if r['entry']==str(e) and r['branch']=='NATIVE'),next(r['pair_R_twoP_joint'] for r in jointrows if r['entry']==str(e) and r['branch']=='NATIVE')] for e in ENTRIES]),'''

모든 branch의 token-micro/prompt-macro/strict는 첫 표 CSV에 있고 요청 joint는 [joint CSV](current-joint.csv)에 있다. 총점이 같아도 성공집합 동일로 간주하지 않았다.

### 4.2 Native 대비 문항 전이

''',md(['Entry','Branch','NS lost','NS gained','Δpp','TF lost/gained','Δmean desired NLL'],[[r['entry'],r['branch'],r['lost'],r['gained'],r['delta_pp'],r['tf_lost']+'/'+r['tf_gained'],f"{float(r['desired_nll_delta_mean']):.7g}"] for r in pairedrows if r['panel']=='Current' and r['metric']=='NS']),'''

[paired transitions](paired-transitions.csv)는 R/P/N·H512의 lost/gained ID hash와 NLL 악화 tail을 포함한다. Current 비교는 case cluster 단위 bootstrap2000회/seed20260924의 기술적95% 구간을 함께 기록했다. 같은 case의 여러 prompt를 독립 표본으로 부풀리지 않았다. 한 entry/order의 기술통계이며 보편적 비열화·인과 우월성의 검정이 아니다.

### 4.3 N512와 history overwrite

N512는 W0 one-token true top1 정답·중복제거·R/P overlap 제외·hash-only로 사전 선택된 고정 observer다. W0 정답 판정은 원 panel provenance 재사용이며 이번 새 W0 평가가 아니다. Current N1000과 다른 모집단이다.

''',md(['Native entry','N512 preference/512','TF true/512','W0-correct gross lost','평균 true NLL'],[[e,metric(e,'NATIVE','NS','N512')['numerator'],metric(e,'NATIVE','NS','N512')['strict_count'],512-int(metric(e,'NATIVE','NS','N512')['strict_count']),f"{float(metric(e,'NATIVE','NS','N512')['true_nll_mean']):.6f}"] for e in ENTRIES]),'''

[N512 stage 전이](N512-stage-transitions.csv)는 entry→W4→W5→W6→W7→W8→history를 exact ID로 묶었다. [W0 retention](N512-w0-retention.csv)은 stage별 gross loss를 보존한다. 이 패널은 모두 W0-correct이므로 W0-failure recovery 분모0이며, native에서 새로 잃거나 회복한 문항은 stage paired lost/gained로 별도 기록했다.

H512 통계는 superseded까지 all512·weight1을 유지한다. Functional ACTIVE/SUPERSEDED만 entry-time latest-target mask로 나눴다. [마스크 수와 ledger](history-mask-summary.csv), [분리 지표](H512-active-superseded.csv)에 R512/P1024와 active 분모를 따로 두었다. Current overwrite/성공률로 통계bank를 바꾸지 않았다. 원 timestamp bank는 처음 도착한 요청의 native write 시점 key이며 현재 key와 구별된다.

## 5. E1/E2 geometry: 저장 수치의 독립 산술

PR = (Σλ)²/Σλ². Raw/centered/unit-raw/unit-centered, projected/raw, individual6context와 실제 native writer mean을 분리했다. 평균은 bare0.5 + generated각0.1이며 6개 단순평균이 아니다. Mean-energy, norm ESS, top1% norm-energy와 zero norm 정책도 따로 기록했다.

저장 고유값·요청별 norm에서 총33,040행의 PR/에너지/ESS를 재계산했다. 공개 [writer-mean table](geometry-independent-spectrum-arithmetic.csv)와 [context range](geometry-context-ranges.csv), 전체 context별 local CSV SHA는 [geometry audit](geometry-audit.json)에 있다. W0/W100·early·L5/L6 네 사전지정 tensor를 CPU weights_only/mmap으로 읽고 finite/shape 및 raw/centered Gram PR을 독립 재계산해 일치했다. 전체 key tensor의 eigendecomposition이나 model forward를 전부 다시 했다는 뜻은 아니다.

''',md(['early1000 state','L5 raw PR','L6 raw PR','L5 mean-energy','L6 mean-energy'],[[s,*[f"{float(next(r['PR'] for r in geo if r['phase']=='E1' and r['state']==str(s) and r['layer']==str(l) and r['panel']=='early' and r['space']=='raw' and r['variant']=='raw')):.6f}" for l in (5,6)],*[f"{float(next(r['mean_energy'] for r in geo if r['phase']=='E1' and r['state']==str(s) and r['layer']==str(l) and r['panel']=='early' and r['space']=='raw' and r['variant']=='raw')):.6f}" for l in (5,6)]] for s in (0,10,50,70,80,90,100)]),'''

이는 사전 지정 early cohort 예시이며 다른 cohort/centered 결과는 CSV에 모두 보존했다. 원인 귀속이나 효능 판정으로 바꾸지 않는다.

E2 각 mask는 calibration512와 assessment4×872=3488의 문항집합을 따로 검사했다. 18개 family 모두 네 assessment cohort와 calibration의 중복0/합4000을 확인했다. Geometry는 각872별 계산이며 pooled3488 PR로 바꾸지 않았다. Upper-control의 W80/W100 K6/K7은 저장 actual key SHA exact/maxdiff0이다. Gate/up 교차 특성은 저장 통계·source 범위이며 미보존 full feature tensor를 재구성하지 않았다.

## 6. E3/E4: native, history, component, K/R

Native z는 각 entry L8 target100회, 총400회다. R4에서는 W50의 기존100을 재사용하고 W70/80/90에서 새300회를 수행했다. 같은 entry의 post-z fork만 공유하며 각 branch의 K/residual/solve는 다시 계산했다. 마지막 L8 residual24회와 divisor5/4/3/2/1을 factor receipt에서 확인했다.

[writer mechanism](writer-mechanism.csv), [native factor](native-factor-checks.csv)는 Δ/K/R/target demand/실제 response/잔차·history coverage의 수치를 분리한다. 120개 layer-row의 request energy 합을 독립 산술로 검사했다. Stored P는 이상적 exact projector로 인증되지 않았고 whitening은 `SYMMETRIZED_NATIVE_H_APPROXIMATION_NOT_IDEAL_P`다. 따라서 이상식의 mode 분석은 diagnostic approximation이지 원 native solver를 대체한 계산이나 완전한 기전 증명이 아니다. C를 통한 R Cᵀ와 실제 Δ 차이도 보존한다.

E4-H는 temporary solve operand M_eff만 바꾸고 persistent history는 native timestamp key를 정확히1회 append한다. 24branch×5layer=120 append 기록, r4 신규110 + r3 재사용10이다. 후보/component/observer append는0. [history audit](history-exactly-once.csv)의 pre/post M/key hash는 실행 receipt 증거이며 이번 전체 M tensor 독립 재구성은 아니다.

E4-W의 no/mean/centered/full은 inference activation 대조다. Mean/centered hook을 실제 weight 방법으로 부르지 않는다. Rank1 parallel/perpendicular 및 norm-control은 별도 물리 weight 대조이며 quality-matched는 NOT_ESTABLISHED다. Full-hook physical parity8개는 작은 실제 패널에서 exact0, 별도 sum-route 차이는 nonzero로 기록됐다. 전체 관측에 대한 일반 bitwise 동등성 PASS가 아니다.

E4-KR 8개 행렬은 모두 같은 receiving W_b, P/M/lambda=10, residual divisor, 고정z에 결속했다. 32개 endpoint 뒤 suffix write/history append는 없다. 이것을 full native B100 최종 endpoint와 혼합하지 않는다. Interaction은 bb−ba−ab+aa의 관측 산술이지 인과 결론이 아니다.

''',md(['Entry','upstream→receiving','NS aa/ab/ba/bb','interaction count','NLL interaction'],[[r['entry'],r['upstream']+'→'+r['receiving'],'/'.join(r[k] for k in ('aa','ab','ba','bb')),r['interaction_count'],f"{float(r['interaction_desired_nll']):.8g}"] for r in rows('KR-interactions.csv') if r['metric']=='NS']),'''

R/P interaction과 모든 component/interchange 수치는 [KR table](KR-interactions.csv), [component paired](component-and-N512-paired.csv), 전체 endpoint CSV에 있다. 불리한 quality도 제거하지 않았고 N/P로 basis·rank·strength·branch를 고르지 않았다.

## 7. 설계 적합성·수치 정책·복원 한계

[source-conformance.csv](source-conformance.csv)에 23항목의 frozen file/function/line/SHA→stored evidence→판정·한계를 연결했다. 원 BASE_ALPHAEDIT blue=False/L4–L8/L2=10/P threshold.02, entry L8 z와 원 native write를 사용한다. L4-only/BLUE/optimized-z로 바꾸지 않았다. FP32/eager, matmul TF32=false, cuDNN TF32=true라는 원 조건을 보존했다.

Observer P/N/H는 endpoint seal 뒤 실행하며 selection feedback은 없다. 136 component observer의 W/M/context/cursor 전후 hash 및 RNG restore equality, writer192 stage observer의 nonmutation/append0 기록을 확인했다. Branch entry 복원은 각 entry-terminal과 source의 RAM restore로 확인했다. Output 존재만으로 복원을 추정하지 않았다.

수치 관찰 정책은 사용자의 명시 override다. NaN/shape/source/state/IO 무결성은 계속 hard failure였다. [numerical comparisons](numerical-comparisons.csv)는 SHAM4개·hook8개의 원 comparison verdict를 보존한다. `full_numerical_equivalence=NOT_ESTABLISHED`이며 관찰 정책을 actual parity PASS로 바꾸지 않는다.

신규 full W/M/RNG/optimizer resume checkpoint는0. 승인된 입력 CP12와 z/K/R/Δ/계수/timestamp/current diagnostic bank는 별도 예외다. 일부 branch weight/history 산술 재구성이 가능하더라도 원 post-z RNG 미보존 등으로 exact crash-resume/GPU continuation은 NOT_ESTABLISHED/NOT_TESTED다. 이번에는 GPU continuation을 수행하지 않았다.

## 8. 비용·자원·재사용

''',md(['구간','Parent GPU-sec','비고'],[['alpha r1 gate failure',44,'자원 사용; 과학 forward 전'],['alpha r3 gate',196,'공통 source/prefix/timestamp'],['alpha geometry',19069,'E1/E2 CPU진단 포함 할당'],['alpha failed writer',1494,'완료 W50 NATIVE/SHAM·z100 보존'],['alpha r4 writer',29802,'새z300 + reuse복원 + 남은branch/component'],['alpha 합계',50605,'14.056944 GPU-hour; utilization 아님'],['SLMF 이전실패',1113,'기존 상세리뷰 재사용'],['SLMF 완료B1',3629,'기존 상세리뷰 재사용'],['EN adaptive failure',6218,'기존 RCA 재사용']]),'''

Alpha CPU reducers는0/113/302초이며 GPU 비용에 넣지 않았다. 최근 구간 전체 parent GPU 합계61,565초다. [interval](allocation-intervals.csv)에서 geometry와 old/new writers의 중첩을 확인했으며 최대2GPU다. Cap은 allocation 구간 기준이고 실제 GPU utilization을 측정한 것은 아니다.

Geometry program19,066.973초, r4 writer program29,798.617초. Geometry GPU allocated peak31.111GiB/reserved33.211GiB, r4 writer38.235GiB/reserved44.381GiB; host MaxRSS 약34.03/34.87GiB다. 이는 프로그램 저장 peak이며 Slurm sampling MaxRSS와 혼동하지 않는다. 59GiB host 요청은 상한이고 실측과 분리했다.

Writer event에서 새300target263.096초, 재사용100의 원 fit87.158초, r4 native-direct-solve110회13.087초, reused solve10회1.234초를 분리했다. 관측 비용은 writer stage r4 12,961.323초/기존1,146.826초이며 component 관측/geometry/hash/restore는 별도 outer timer 안에 있다. 원 z400비용을 각 branch에서 다시 청구하지 않았다. [compute](compute-components.csv), [events](writer-events.csv)에 rejected/실패전 완료 작업도 보존했다.

Outer program/capture/observer와 내부 kernel/native/diagnostic timer는 중첩돼 합산하지 않는다. 순수 writer 전체·restore·I/O·GPU idle은 NOT_SEPARATED이며 독립 반복 timing이 없으므로 인과 speedup·안정적 배수는 미확립이다. 사용자의 wall hardcap은 지정되지 않았다.

## 9. 다른 server4 family: 기존 리뷰 재사용

| Family/job | 기존 검산 사실 | 이번 경계 |
| --- | --- | --- |
| SLMF51058 | 네 arm R100/P200/N1000 = 100/194/865; EN trial4수용, DEC2arm native fallback; B1→S3 gate FAIL | full numerical validation NOT_ESTABLISHED/FD waiver 유지; B1-only |
| SLMF50974/51055/51056/51057 | 앞선 T0/수리 실패1113GPU-sec; 50983실행전 취소0 | source/submission/새accounting 결속; raw 전량 재검산하지 않음 |
| EN-adaptive51260 | JSON np.bool serialization first cause, commit/history/eval0,6218GPU-sec | raw 실패 보존, noCP exactresume 불가; SH3 repair 소유권을 이번 리뷰로 재개하지 않음 |
| EN-reuse50410/50449 | preparation와 matched single-B1 완료 상세리뷰 게시 | 다른 R512/G256 목적·분모이며 alpha 대조에 섞지 않음 |
| EN-F50071_1 / SLZ49466 / BPCW 등 | 게시된 completed review·failure report SHA index 재사용 | live sibling/옛 전체raw/모델 평가 재개0 |

[SLMF 기존 상세보고](../single-layer-mechanism-first-20260919-v1/completed-b1-review-20260920-v1/report-ko.md), [EN-adaptive RCA](../en-adaptive-nullspace-2026-09-20-v1/failure-handoff-r1/report-ko.md), [EN reuse 리뷰](../en-execution-reuse-r512-g256-20260919-v1/completed-review-v1/diagnostic-report-ko.md)를 연결한다. 여러 실험의 성공률을 동일 조건/동일 paired 분모로 합치지 않았다.

SLMF 세부 lineage도 기존 근거 그대로다. 50974 첫 원인은 `T0_NATIVE_Z_HOOK_PARITY_FAILED`(서로 다른 Adam 궤적 gradient 상대차)였다. 51055는 hook 비교 뒤 `Runtime.reset()`으로 비운 oracle list를 다시 remove한 cleanup `list.remove(x): x not in list` 오류다. 51056의 JSON 직렬화 수리 뒤 51057에서 남은 첫 차단은 pair-factor AD/FD 미확립이었다. 4reference direct/cached gradient는 같았지만 3reference가 고정 인접2scale 조건을 충족하지 못했고, 사용자가 waiver한 뒤51058 B1이 진행됐다. 수리와 사용자 기준변경을 같은 종류로 쓰지 않는다. 이 이력은 [r1](../single-layer-mechanism-first-20260919-v1/repair-r1/repair-and-gate-policy-ko.md), [r2](../single-layer-mechanism-first-20260919-v1/repair-r2/cleanup-repair-ko.md), [r3](../single-layer-mechanism-first-20260919-v1/repair-r3/diagnostic-report-ko.md)에 연결된다.

51260 첫 원인은 frozen controller의 NumPy scalar 비교가 반환한 `np.bool`을 `runner.py:72→21 json.dump(allow_nan=False)`가 직렬화하지 못한 것이다. EN_EXACT 결과 저장 중 종료, 후속 cleanup 예외는 기록되지 않았다. T0553.238초, shared native100 outer194.840초, R512512문서/130235위치 gradient1709.714초 등은 prior receipt 재사용이며 중첩 합산하지 않는다. Partial objective 값은 endpoint/commit 성공이 아니고 process entry rollback은 NOT_VERIFIED다. EN_NUM/ADAPT/B2/B3 공식평가는 미실행이다.

## 10. 무결성·재현·최종 한계

''',f"완료/실패 output {inventory['members']:,}개 경로를 새 fullSHA로 봉인했다. inode 중복제거 {inventory['unique_inodes']:,}개, {inventory['unique_bytes']:,} bytes이며 논리 경로 합은 {inventory['logical_bytes']:,} bytes다. 링크 재사용분을 저장공간/실행량으로 중복 계상하지 않았다. [output inventory](output-inventory.csv)와 [receipt](output-inventory-receipt.json)에 있다. 원 입력 CP12/공용모델/teacher는 기존 fullSHA+현재 binding 재사용이고 이번 fullrehash로 쓰지 않았다.\n",'''

![Actual endpoint and geometry observations](actual-endpoints-and-geometry.png)

그림은 실제 저장 지점만 표시한다. 연결선은 읽기 보조이며 중간 미측정 성능이나 인과효과를 뜻하지 않는다. 리뷰 code는 새 namespace만 사용하고 production runtime은 변경하지 않았다. 독립 agent red는 사용하지 않았으며 **owner source audit + 별도 구현 CPU reducer**다.

CPU 회귀검사9개, GFM 표 열수·상대링크, 그림 bitwise 재생성과 육안 확인을 수행했다. HTML/GFM renderer가 설치되지 않아 실제 HTML 렌더와 browser screenshot은 NOT_RUN이며 PASS로 표시하지 않았다.

재현: repository에서 `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4`를 지정하고 `/data/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.server4_completed_jobs_review_20260924.review first` 및 `metrics`, `inventory`; `audit metadata|geometry|states`; `advanced run|tensor_sample`; `package evidence|plot|report`를 실행한다. Accounting은 새 조회하지 않고 원 snapshot을 재현한다. `unittest`는 새 review test만 실행한다.

알 수 없는 값은 없는 대로 남겼다: full W/M 사후 독립 재구성, GPU continuation, 일반 bitwise numerical equivalence, 미보존 featuretensor, pooled3488geometry, 순수 IO/utilization/인과 speedup은 미확립이다. 한정 actual gate·CPU 산술·frozen source 일치의 의미를 그 이상으로 확대하지 않는다. 과학적 원인 귀속/방법 승격/다음 실험 선택은 GH 별도 리뷰 범위다.

실행/분석 source와 검산 SHA는 `rooted-receipt.json`, `analysis-manifest.json`, `validation.json`에 연결한다. 최종 publication commit은 별도 인계에서 기록하여 자기참조 SHA를 만들지 않는다. 원자료 삭제/전송/모델 호출0, `NO_BROADCAST_NOT_REQUIRED`. 본 리뷰 게시 후 **TASK_COMPLETE_STOP**, monitoring_active=false, automatic_resume=false. 사용자 다음 recall 전 추가 실행·주기 조회 없음.
''']
    (PACKAGE/'report-ko.md').write_text('\n'.join(chunks))

if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
