"""Render completed CPU sweep package into factual Korean report + hashes.

The generator refuses absent arm audits; it never fills missing model results.
It does not submit, alter a runtime, or delete/overwrite a publication.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from . import review_nogate as r
from .sweep_control import ROOT,OLD,NEW_ARMS,TASK
from .sweep_review import REPORT,LABELS

def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))

def fmt(v):
    if v is None or v=='':return 'NOT_RECORDED'
    if isinstance(v,float):return f'{v:.7g}'
    return str(v)

def mdtable(columns,data):
    def cell(v):return fmt(v).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join('---' for _ in columns)+' |',
        *['| '+' | '.join(cell(v) for v in row)+' |' for row in data]])

def write(path,text):
    with Path(path).open('x') as f:f.write(text+'\n')

def scheduler_rows(status):
    entries=[x.split('|') for x in status['queries']['sacct']['stdout'].splitlines() if x.strip()]
    result=[]
    for item in status['states']:
        top=next(x for x in entries if x[0]==item['job_id'])
        assert top[2]=='janghj' and top[3]=='COMPLETED' and top[4]=='0:0',('SCHEDULER_NOT_SUCCESSFUL_TERMINAL',top)
        assert int(re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',top[9]).group(1))==1
        start=datetime.fromisoformat(top[6]);submit=datetime.fromisoformat(top[5])
        steps=[x for x in entries if x[0].startswith(item['job_id']+'.')]
        result.append(dict(arm=item['arm'],job_id=item['job_id'],state=top[3],exit=top[4],
            submit_host_KST=top[5],start_host_KST=top[6],end_host_KST=top[7],elapsed_seconds=int(top[8]),
            allocated_GPU_seconds=int(top[8]),queued_seconds=(start-submit).total_seconds(),
            allocated_TRES=top[9],MaxRSS=';'.join(x[0]+':'+x[10] for x in [top,*steps] if x[10]) or 'NOT_RECORDED'))
    return result

def build(worktree,package,scheduler):
    w,p=Path(worktree),Path(package);s=r.read(p/'sweep-summary.json')
    assert tuple(s['arms'])==LABELS and s['new_chains']==3 and s['new_batches']==30
    status=r.read(scheduler);jobs=status['states'];assert len(jobs)==3
    for x in jobs:
        assert x['terminal'] and x['terminal']['completed_requests']==1000 and not x['failure']
        assert x['committed_batches']==10
    final=rows(p/'first-final-table.csv');action=rows(p/'action-summary.csv');cost=rows(p/'cost-summary.csv')
    ledger=rows(p/'ledger-summary.csv');warnings=rows(p/'parity-warnings.csv');paired=rows(p/'cross-arm-paired.csv')
    bybatch=rows(p/'per-batch-actions.csv');policy=rows(p/'batch-policy.csv');base=rows(p/'baseline-first1000.csv')
    metric=lambda arm,m:next(x for x in final if x['arm']==arm and x['metric']==m)
    finaltable=mdtable(['Arm','RS n/1000 (%)','PS n/2000 (%)','NS n/10000 (%)','CAP1 대비 ΔR/ΔP/ΔN'],[
        [a,*[f"{metric(a,m)['numerator']}/{metric(a,m)['denominator']} ({float(metric(a,m)['percent']):.3f})" for m in ('RS','PS','NS')],
         '/'.join(metric(a,m)['delta_CAP1_count'] for m in ('RS','PS','NS'))] for a in LABELS])
    actiontable=mdtable(['Arm','C1/native % min–max','selected/RAW0 포함 % mean','nonzero selected % min–max','RAW/C1/C05/C025'],[
        [x['arm'],f"{float(x['C1_percent_min']):.6g}–{float(x['C1_percent_max']):.6g}",
         float(x['selected_with_RAW_percent_mean']),
         f"{x['nonzero_selected_percent_min']}–{x['nonzero_selected_percent_max']}",
         '/'.join(x[k] for k in ('RAW','C1','C05','C025'))] for x in action])
    strict=mdtable(['Arm','R TF-strict','P TF-strict /2000','two-P strict /1000','NS true NLL mean/p99'],[
        [a,metric(a,'RS')['desired_strict_n'],metric(a,'PS')['desired_strict_n'],metric(a,'PS')['all_prompt_strict_requests'],
         f"{float(metric(a,'NS')['true_nll_mean']):.7g}/{float(metric(a,'NS')['true_nll_p99']):.7g}"] for a in LABELS])
    pairtable=mdtable(['대조','metric','lost/gained','Δpp','request-cluster 95% CI'],[
        [x['comparison'],x['metric'],x['lost']+'/'+x['gained'],float(x['delta_pp']),f"[{float(x['CI_low']):.4g}, {float(x['CI_high']):.4g}]"] for x in paired])
    costtable=mdtable(['Arm','분류','allocated GPU-sec','program sec','native / map / Egrad / Dgrad / screen / history sec'],[
        [x['arm'],x['cost_origin'],x['allocated_GPU_seconds'],float(x['program_seconds']),
         ' / '.join(f"{float(x[k]):.2f}" for k in ('native_seconds','map_seconds','current_gradient_total','S64_gradient_total','screen_seconds','history_seconds'))] for x in cost])
    ledgerend=[x for x in ledger if x['batch']=='10']
    ledgertable=mdtable(['Arm','requested','accepted','distinct subject/relation','ACTIVE/SUPERSEDED/UNKNOWN','accepted overwrites'],[
        [x['arm'],x['requested'],x['accepted'],x['distinct_accepted_subject_relation'],'/'.join(x[k] for k in ('active','superseded','unknown')),x['accepted_target_overwrites']] for x in ledgerend])
    batchtable=mdtable(['Arm','B','선택','alpha_norm / alpha','C1/native %','selected/native %','ball hit','trust scale','RAW→selected E / D'],[
        [x['arm'],x['batch'],x['selected'],f"{float(x['alpha_norm']):.6g}/{float(x['alpha_used']):.6g}",
         float(x['C1_native_percent']),float(x['selected_native_percent']),x['ball_hit'],x['trust_retraction'],
         next(f"{float(y['selected_E'])-float(y['raw_E']):+.5g} / {float(y['D64_change']):+.5g}" for y in policy if y['arm']==x['arm'] and y['batch']==x['batch'])] for x in bybatch])
    baseline=mdtable(['Reference','metric','n/d','%','검증/비교 범위'],[
        [x['policy'],x['metric'],x['numerator']+'/'+x['denominator'],float(x['percent']),x.get('verification','')] for x in base])
    state=rows(p/'new-state-summary.csv')
    state_table=mdtable(['Arm','raw files / bytes','CP / bytes','인접links','inner/final history','GPU replay'],[
        [x['arm'],x['raw_files']+' / '+x['raw_bytes'],x['checkpoints']+' / '+x['checkpoint_bytes'],x['adjacent_links'],
         x['inner_history_appends']+' / '+x['final_history_appends'],x['GPU_continuation']] for x in state])
    schedule=scheduler_rows(status)
    for x in schedule:
        assert int(next(c['allocated_GPU_seconds'] for c in cost if c['arm']==x['arm']))==x['allocated_GPU_seconds']
    r.table(p/'scheduler-summary.csv',schedule)
    schedule_table=mdtable(['Arm/job','state/exit','start/end (KST)','queue sec','allocated GPU-sec','MaxRSS'],[
        [x['arm']+'/'+x['job_id'],x['state']+'/'+x['exit'],x['start_host_KST']+' / '+x['end_host_KST'],
         x['queued_seconds'],x['allocated_GPU_seconds'],x['MaxRSS']] for x in schedule])
    coverage=[
        ('fixed W0/M0, sample1000/order/model/P/context/RNG','SOURCE_AND_INPUT_LOCK_CONFIRMED','새3arm 동일10B100; CAP1 과거execution 별도'),
        ('cap-only mode/null/finite/legacy arithmetic','SOURCE_AND_CPU_FIXTURE_CONFIRMED','새GPU derivative correctness 검증 아님'),
        ('full terminal/current/accepted-old/first500/NLL/strict','STORED_ROWS_CPU_REDUCED','분모별 CSV; atwrite pool을 동일W0 평가로 부르지 않음'),
        ('all40 logical batches/all160 candidates','SOURCE_AND_STORED_CANDIDATE_RECORDS','CAP1 10batch 재사용+신규30batch; duplicate는 추가forward 아님'),
        ('M/RNG/ledger/selected W checkpoint','CPU_RELOAD_HASH_BRIDGE','신규30CP; CAP1 현재10CP 부재, 과거 JSON/tensor감사 재사용'),
        ('S64/Dev128','STORED_ROWS_CPU_REDUCED','S64 controller; Dev W5/W10 observer; independent Report 아님'),
        ('raw byte SHA vs tensor header hashes','CPU_FILE_AND_TENSOR_BRIDGE','full model/GPU continuation이나kernel parity 아님'),
        ('FD/direct-gradient/ULP/jitter/selfKL','SKIPPED_USER_DIRECTED','numerical_validation=NOT_ESTABLISHED; 사후GPU 재검증0'),
        ('pure writer / checkpoint I-O component','NOT_SEPARATED','native instrumentation 또는 미분리 overhead; 임의시간 추정0'),
        ('request Adam/m/v/local-teacher full payload','PARTIAL_OBSERVATION','native counter/target/anchor는 저장; fulloptimizer payload NOT_SAVED'),
        ('Report256/Audit128/MMLU68/FutureN','NOT_MEASURED','본scope 미실행; loss/selection에 사용0'),
        ('new baseline editing/teacher generation/remote raw','NOT_RUN','기존localasset·publication 재사용'),
        ('full10k/다른order/다른layer/후속method','NOT_RUN','추가cap/후속chain 자동제출0')]
    r.table(p/'coverage.csv',[dict(requirement=a,status=b,limitation=c) for a,b,c in coverage])
    coverage_table=mdtable(['요구','상태','한계'],coverage)
    warning_count=sum(x['recorded_status']=='WARNING' for x in warnings)
    maxwarn=max(float(x['max_abs_NLL_difference']) for x in warnings)
    text=f'''# EP-TW-1 alpha cap sweep — SH4 완료 사실 보고

Instruction `{TASK}`. 이 보고서는 실행 조건·저장 값·CPU 산술·오류/한계만 기록한다. 과학적 우열·인과 해석·후속 선택은 GH 소유다.

## 1. 범위와 완료 상태

CAP1/job47962는 완료 reference 재사용. CAP10/CAP100/NORM_ONLY 각각 fresh pretrained W0/coldM0에서 같은 첫1000을 B100×10으로 처리했다. 신규3chains/30batches/3000 arm-request observations, unique1000이다. 비교 전체는4정책/40 logical batches이며 새4000 unique 편집이 아니다. CAP1 재실행0. CAKE의 no-weight-storage와 monitoring pause는 이 task에 상속하지 않았고 CAKE에 대한 자원 admission 외 점검·변경0이다.

정확 신규jobs48148/48149/48150. Scheduler terminal은 별도 `{Path(scheduler).name}`에 보존했고 scientific terminal과10commit/arm을 독립 확인했다. 전체 완료 확인 전의 PENDING 기록을 initial PASS로 소급하지 않는다. 신규 source8c64366c2314f188e034e5f4403fe89f0eaad873/tree76ef5072b70134bf4bc1f9d3659a056041922bde, archive304be4f3de58ce2f9457f63324646525000191609ba64e1bb1e290253e59ea21. 분석 source와 publication HEAD는 manifest에 별도 pin한다.

## 2. Actual W10 원분모와 strict

{finaltable}

RS/PS는 new NLL<true NLL, NS는 true NLL<new NLL, ties=failure다. Current100·whole1000·first500을 합쳐 분모를 늘리지 않았다. At-write는 각자 다른 W1..10의 관측이며 W10 retention과 별도다.

{strict}

Ranking PS와 teacher-forced strict 및 두 rephrase 모두 strict는 다른 지표다. NLL mean/median/p90/p95/p99·signed desired margin·token counts는 first-final-table/batch-current-metrics/whole-prefix-metrics에 보존했다.

## 3. 실제 action과 후보 선택

{actiontable}

C1은 최대 후보이며 실제 선택과 다르다. RAW의 실행 보정은0이지만 native edit 자체는 commit된다. NORM_ONLY는 숫자 cap만 없앴으며 target ball·25% executable trust·E/strict finite screen은 유지했다. cap 간 후보 메뉴는 포함관계가 아니며 검사하지 않은 중간 action은 평가하지 않았다.

{batchtable}

모든 RAW/C1/C05/C025 후보의 E/D64/RAW대비 변화·strict lost IDs 수·finite/trust/feasible/선택/사유는 candidate-details.csv에 공개했다. Feasible minD·RAW우선 numerical tie·작은 actual correction norm·고정ID 순서를 저장 scalar와 exact strict ID 집합으로 독립 재계산했다. 같은 strict count를 같은 성공 ID로 간주하지 않았다. 추가 quality allowance/조기후보종료/후속튜닝0.

## 4. 설계 → 실제 동작과 검증 수준

실행은 자기 entry의 native fit/target100과 raw FP32 Vp를 만든 뒤 fixed A에서 E와 S64 D gradient를 각각1회 계산한다. C0=actual Vp이고 factorized native endpoint로 바꾸지 않았다. `alpha_norm=.25||Vp-Wentry||/(||dA||+1e-12)`, bounded는 `min(cap,alpha_norm)`, disabled는 alpha_norm. Target-ball 후 C만 trust-retract하며 후보는 Vp+beta(CA)다. 새 metadata 계산은 RNG/state를 쓰거나 바꾸지 않는다. Legacy CAP1 CPU C/alpha/candidate bytes 일치 및 고정 RNG fixture를 확인했으므로 reference 재사용을 유지했다.

반공간 d, q/gradient norms/coefficient/inner products와 ball/trust/actual selected norm은 per-batch-actions.csv에 있다. 새3arm은 저장 gE/gD/C/A/selectedCP로 CPU scalar와 실제 raw-to-selected norm을 다시 계산했다. CPU로 재도출한 d와 matmul은 원 GPU에 저장된 tensor라고 쓰지 않았고, CPU 반올림 차이도 기록했다. 원 native/fitter/model_adapter/ledger/selector source는 cap 외 의미를 바꾸지 않았다.

각batch 최종 selected endpoint에서 history1, innerappend0, 다음entry는 자기 W/M/RNG/ledger다. Observer는 Current P/N/accepted-old/Dev이고 controller에는 canonical Current E/strict 및 S64만 전달된다. Current 평균 E·strict 보호는 모든 요청 NLL·PS·old retention 보장이 아니다.

Numerical gate는 사용자 지시로 생략했다. `SKIPPED_USER_DIRECTED / numerical_validation=NOT_ESTABLISHED`를 유지하며 CPU source/산술/hash를 model-level derivative PASS로 바꾸지 않는다. 기존47942 saved episode E direct PASS도 그 범위뿐이다. 이번 새 GPU FD/ULP/jitter/selfKL/gradient 검사0.

Method observer/canonical evaluator 비교 {len(warnings)}패널 중 warning{warning_count}, 최대 NLL 차이 {maxwarn:.8g}. 추가 forward 없이 이미 계산된 rows를 비교했으며 기록을 제거하거나 numerical PASS로 승격하지 않았다.

## 5. 순차 유지·문항 전이·accepted ledger

{pairtable}

CI는 같은 fixed single order의 request-cluster bootstrap2000draws/seed20260915다. P2/N10 prompts를 독립 표본으로 늘리지 않았고 batch10을 독립 replicate로 보지 않는다. CI로 arm 탈락/선택0. B2이후 대조에는 서로 다른 trajectory가 누적되어 same-state 단일인자 인과효과가 아니다.

At-write→W10, first500W5→W10, cohort별lost/gained·conditional denominators·desired NLL harm p95/p99는 paired-transitions/cohort-retention에 있다. B10은 future exposure0으로 구분한다. 관측되지 않은 중간 최초 실패·회복 시점을 추정하지 않는다. Accepted-only와 전체requested1000의 분모는 별도다.

{ledgertable}

ACTIVE_TARGET/SUPERSEDED/UNKNOWN은 accepted exact subject/relation latest target 규약이다. Unaccepted 요청으로 과거 accepted label을 덮어쓰지 않는 코드 및 저장 ledger 규칙을 확인했다. Requested intent와 accepted-only를 섞지 않았다. Legitimate target overwrite와 canonical 실패를 동일하게 처리하지 않는다.

## 6. Generic와 baseline 비교 경계

S64는 매batch controller/selector panel, Dev128은 W5/W10 observer다. 고정 W0 full-vocabulary FP32 teacher192/24shards를 재사용했다. 자연256+BOS1, logits128:256의128positions, vocabulary합→position평균→document평균을 저장 rows와 identity로 확인했다. S64와 Dev ID는 disjoint이며 새 문서/token/teacher 선정0. S64 D 감소를 NS 성공 증가와 동일시하지 않는다. Generic-panel tables와 figure에 두 population을 분리했다.

{baseline}

표는 기존 동일 first1000의 actual W10 또는 W0 reference 재사용이다. Hparams/layers/seed/wrapper 환경 차이는 compatibility.csv에 원값을 유지했다. N4는 AlphaEdit-BLUE L4-only가 가장 가까우며 기존 독립 감사의 B001 native target·NLL 일치는 해당 최초 상태 범위로만 재사용한다. EP 자기 trajectory RAW는 별도 native sequential chain이 아니다. W50 suffix/full6000 또는 W100 full10k와 직접 혼합하지 않았다. Baseline paired는 로컬 저장 case/prompt/target identity가 일치한 문항으로만 계산하고 W0 aggregate를 per-case로 역복원하지 않았다.

## 7. 비용과 저장

{schedule_table}

{costtable}

신규 allocation 총 {s['new_allocated_GPU_seconds']} GPU-sec = {s['new_allocated_GPU_seconds']/3600:.6f} GPUh. 신규 추정6.411667GPUh는 선형 예상이었으며 실측/hardbudget이 아니었다. CAP1재사용7694초, teacher재사용98초, 기존실패47884/47942의473+69초는 신규지출에 중복합산하지 않았다. 286.5445 nativefit초는473초 안의 component다.

각native z/key/readout/solve는 native time 내부, S64 F/B와 teacher read는 generic total 내부이므로 단순합산0. Native RHS solve10/arm 외 fixed A 구성 solve10/arm은 별개다. 별도 purewriter/CP-I-O 전체 breakdown은 NOT_SEPARATED/NOT_RECORDED이며 임의 추정하지 않았다. Actual Adam/loss/earlystop/target counts와70current/640S64 backward 등 실제 단위는 cost-by-batch/target-counters를 참조한다. Cap마다 policy trajectory/earlystop/I-O/queue가 달라 비용비를 통제된 speedup으로 부르지 않는다.

{state_table}

신규 각arm W10 L4 weight, M4/history/context/RNG/ledger/model/P/order/nextordinal을 실제 보존한다. 30개 checkpoint fullSHA/weights_only/finite/shape/hash bridge와27개인접links를 확인했다. Full-model GPU continuation/replay는 NOT_TESTED다. CAP1의 과거10CP는 현재부재이며 남은native/route selectedW 복원에 대한 봉인 감사만 재사용한다. 현재 없는CP로 fullresume을 주장하지 않는다. 원자료 이동·삭제·원격전송0, raw tensors/prompts/teacher/fullstdout은 local-only다.

## 8. Coverage·미실행·산출물

{coverage_table}

PNG는 이 package의 CSV로만 Matplotlib 생성했으며 source/input/output SHA와 재현명령은 analysis-manifest/figure manifest에 있다. 원문 기하 예측은 CAP1 저장episode CPU 계산이고 이번 새 sequential model 실측과 구분한다. 결과가 유한하나 낮거나 RAW가 많아도 제외하지 않았다. 새 방법/추가cap/후속chain은 실행하지 않았으며 이 보고 후 task STOP한다.
'''
    write(p/'diagnostic-report-ko.md',text)
    r.save(p/'scheduler-terminal-receipt.json',status)
    source_files=[Path(__file__),Path(__file__).parent/'sweep_review.py',Path(__file__).parent/'plot_sweep.py',
        Path(__file__).parent/'review_nogate.py',Path(__file__).parent/'review_nogate/state_audit.py']
    files=[r.ref(x) for x in sorted(p.rglob('*')) if x.is_file()]
    manifest=dict(instruction_id=TASK,analysis_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip(),
        analysis_worktree=str(w),analysis_source=[r.ref(x) for x in source_files],
        execution=r.read(ROOT/'frozen-arms.json'),CAP1_execution='6d317bdb2660d7e9919bc3a9fb878564e9729e37',
        full_read=r.ref(ROOT/'full-read-receipt.json'),reuse_inputs=s['inputs'],package=files,
        scheduler=r.ref(scheduler),python=platform.python_version(),time=datetime.now(timezone.utc).isoformat(),
        numerical_validation='NOT_ESTABLISHED',validation_mode='SKIPPED_USER_DIRECTED',
        raw_policy='LOCAL_ONLY_NO_BROADCAST_NOT_REQUIRED',new_model_forwards_analysis=0,
        reproduction=['python -m project.run_scripts.bg_tw_reference.ep_tw.sweep_review arm --arm <arm> --allocated <actual seconds> --worktree <worktree> --output <new-dir>',
            'python -m project.run_scripts.bg_tw_reference.ep_tw.sweep_review combine --analysis-root <three-arm-root> --worktree <worktree> --output <new-package>',
            'python -m project.run_scripts.bg_tw_reference.ep_tw.plot_sweep --package <package> --output <new-figure-dir>'])
    mref=r.save(p/'analysis-manifest.json',manifest)
    return r.save(p/'rooted-receipt.json',dict(status='COMPLETED_SCIENCE_STORED_ARTIFACT_CPU_REVIEW',manifest=mref,
        report=r.ref(p/'diagnostic-report-ko.md'),new_chains=3,reused_chains=1,new_batches=30,
        unique_requests=1000,numerical_validation='NOT_ESTABLISHED',GPU_continuation='NOT_TESTED',
        automatic_resume=False,next='TASK_COMPLETE_STOP_AFTER_OWN_SCOPE_MAIN_PUBLICATION'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worktree',required=True);p.add_argument('--package',required=True);p.add_argument('--scheduler',required=True)
    a=p.parse_args();print(json.dumps(build(a.worktree,a.package,a.scheduler)))
