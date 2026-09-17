"""Create-once raw-free Korean publication from completed CPU evidence.

Does not read model payloads, invoke a scheduler, or modify runtime/raw files.
"""
import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
import shutil
import subprocess
from .bootstrap import ROOT,REVIEW,identity,save
from .metrics import ARMS,read,table

REPORT='experiment-reports/servers/server4/local-z-adaptive-allocation-seq1000-2026-09-16-v1/completed-review-20260917-v1'
EXEC='32a92ad6f3fff2f258d8778f3936d152e975ac1b'
def rows(p):return list(csv.DictReader(Path(p).open()))
def fmt(x):
    if x is None or x=='':return 'NA'
    if isinstance(x,bool):return str(x)
    if isinstance(x,float):return f'{x:.6g}'
    return str(x).replace('|','&#124;').replace('\n',' ')
def md(headers,records):
    return '\n'.join(['| '+' | '.join(map(fmt,headers))+' |','| '+' | '.join('---' for _ in headers)+' |']+['| '+' | '.join(map(fmt,r))+' |' for r in records])+'\n'
def write(p,text):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:f.write(text)
def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    with Path(src).open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b)
    assert identity(src)['sha256']==identity(dst)['sha256']

def publish(worktree):
    w=Path(worktree).resolve();out=w/REPORT;out.mkdir(parents=True,exist_ok=False)
    data=REVIEW/'analysis-r2';tensor=REVIEW/'tensor-audit';supp=REVIEW/'supplement-v1'
    s=read(data/'summary.json');t=read(tensor/'tensor-audit.json');u=read(supp/'summary.json')
    assert t['status']=='PASS' and s['batches']==70 and s['links']==63 and s['history']==110
    reuse=[]
    for root in (data,supp):
        for p in sorted(root.glob('*.csv')):copy(p,out/p.name);reuse.append(dict(source=identity(p),publication=p.name,mode='BYTE_EXACT_CPU_OUTPUT'))
    for name in ('checkpoint-tensors.csv','candidate-reconstruction.csv','native-fit-tensors.csv'):
        p=tensor/name;copy(p,out/name);reuse.append(dict(source=identity(p),publication=name,mode='BYTE_EXACT_CPU_OUTPUT'))
    for source,name in [(tensor/'raw-inventory.json','raw-inventory.json'),(supp/'input-source-inventory.json','input-manifest.json'),
        (REVIEW/'first-tables/first-final-table.csv','first-final-table.csv'),(REVIEW/'full-read-m0.json','full-read-m0.json')]:
        copy(source,out/name);reuse.append(dict(source=identity(source),publication=name,mode='BYTE_EXACT'))
    for p in sorted((REVIEW/'figures-v1').glob('*.png')):copy(p,out/'figures'/p.name);reuse.append(dict(source=identity(p),publication='figures/'+p.name,mode='CODE_GENERATED_BYTE_EXACT_COPY'))
    totals=rows(tensor/'target-counters.csv');target_summary=[]
    for arm in ARMS:
        for b in range(1,11):
            rr=[r for r in totals if r['arm']==arm and int(r['batch'])==b]
            target_summary.append(dict(arm=arm,batch=b,target_calls=len(rr),Adam=sum(int(r['Adam']) for r in rr),loss=sum(int(r['loss']) for r in rr),
                early_stop=sum(r['early_stop']=='True' for r in rr),zero_step=sum(r['zero_step']=='True' for r in rr),clamp_hits='NOT_RECORDED'))
    table(out/'target-counter-summary.csv',target_summary)
    costs=rows(data/'compute-components.csv');sched=rows(data/'scheduler-cost.csv');sel=rows(data/'selection.csv')
    cand=rows(data/'candidate-metrics.csv');cons=rows(data/'constraints.csv');pop=rows(data/'population-metrics.csv')
    paired=rows(data/'paired-policy-comparisons.csv');trans=rows(data/'paired-transitions.csv');strict=rows(supp/'strict-transitions.csv');dev=rows(data/'generic-observer.csv')
    cps=rows(tensor/'checkpoint-tensors.csv');inv=read(tensor/'raw-inventory.json');cold=read(ROOT/'technical/attempt-v1/cold-capsule.json')
    final={a:{r['metric']:r for r in pop if r['arm']==a and r['population']=='W10_ALL1000' and r['status']=='ALL'} for a in ARMS}
    finalrows=[]
    for a in ARMS:
        r=dict(arm=a,status='COMPLETED_1000_CPU_METRIC_STATE_CHECKED; OPERATIONAL_CAP_DEVIATION')
        for metric in ('RS','PS','NS'):
            m=final[a][metric];r.update({metric+'_count':int(m['numerator']),metric+'_denominator':int(m['denominator']),metric+'_percent':float(m['percent']),metric+'_vs_N4_pp':float(m['percent'])-float(final['N4'][metric]['percent'])})
        finalrows.append(r)
    table(out/'final-seven-arm.csv',finalrows)
    conformance=[
      ('Cold/input/seed','model.py',53,'W0; M4/M8=0; FP32/eager; TF32off; seed20260916','7 start+capsule+63links; same first1000 identity','SOURCE_CONFIRMED + STORED_EVIDENCE_CONSISTENT','Full pretrained tensor rehash/GPU reload not repeated'),
      ('P mapping','model.py',84,'physical4→asset0/local0; physical8→asset4/local0','P4 24a654cc…; P8 3a4c524e…; all state bindings','REUSED_IDENTITY + STORED_EVIDENCE_CONSISTENT','Large P payload prior fullSHA+current stable stat'),
      ('LOCAL','engine.py',13,'own N4 then fresh L8 for each a4; a8 variants share same fit','saved native fits, target captures, cache/input states; B2–10 candidate CPU hashes','SOURCE_CONFIRMED + STORED_EVIDENCE_CONSISTENT','B1 all candidate materialization not independently reconstructed from W0'),
      ('TERMINAL','model.py',26,'fixed entry Z8; writer4/8 key; residual Z8−H8 at current state','50 saved R exactly equal Z8−H8 on CPU','SOURCE_CONFIRMED + CPU_TENSOR_EXACT','No new neural forward or directsolve replay'),
      ('REFIT4','engine.py',33,'.75 actual L4 partial then fresh same-layer fit','20 fits/2000targets; M4 final append10','SOURCE_CONFIRMED + STORED_EVIDENCE_CONSISTENT','Not I2/I4 carry; no warm entry'),
      ('FP32 gate/all-token','policy.py',5,'0/1 exactcopy; .5/.75 U+g(V−U); physical parameter.copy_','materialization CPU fixtures; saved endpoints/hash reconstruction','SOURCE_CONFIRMED + CPU_TENSOR_EXACT','FP32 delta alone not claimed exact replay'),
      ('E/S_cur','model.py',189,'new token mean→rewrite context mean→100request mean; canonical strict set separately','190 E means and strict sets independently reduced','SOURCE_CONFIRMED + STORED_EVIDENCE_CONSISTENT','Not RS/PS/NS and not every request non-worsening'),
      ('H/S_past','model.py',223,'Past64 canonical rewrite new-NLL and exact strict IDs','190 candidate Past means/sets; B1 empty','SOURCE_CONFIRMED + STORED_EVIDENCE_CONSISTENT','Past64 is not all old requests'),
      ('D/teacher','model.py',103,'fixed W0 full vocab KL p0||pW; positions128; S64 mean','teacher reused; technical W0 D=0; all190 D means','SOURCE_CONFIRMED + REUSED_GPU_EVIDENCE','Review performs no teacher/model recomputation'),
      ('Selector','policy.py',37,'E plateau+1e-4; strict subset; Past H+1e-4; D tie1e-6','30 dynamic selections/constraints/shadows independently identical','CPU_ARITHMETIC_CONFIRMED','Fixed4arms do not apply dynamic screen'),
      ('Received Past','policy.py',26,'latest raw fact event; current overwrite excluded; SHA priority≤64','70 Past lists independently exact; arm-invariant input IDs','CPU_ARITHMETIC_CONFIRMED','Not EP accepted-only ledger'),
      ('Branch/cache/RNG','engine.py',13,'episode-local source/state/context/layer-bound fits; reset batch-entry RNG','63 state links; input-state M/P/context/RNG; stored order test exact','SOURCE_CONFIRMED + STORED_EVIDENCE_CONSISTENT','No full backbone bit audit'),
      ('Observer isolation','model.py',118,'nonselected version/pointer/hooks/grad/mode guards; observe state equality','190 scores and 70completed commits; no failure artifact','RUNTIME_GUARD_EVIDENCE','Pointer/version guard is not independent full-parameter byte equality'),
      ('Commit/history','runner.py',60,'selected endpoint; inner0; one append per eligible layer','70commits;110 append receipts; TD/N4 choice still M8 append','SOURCE_CONFIRMED + STORED_EVIDENCE_CONSISTENT','Finalizer keys not saved: Gram append not independently rederived'),
      ('Snapshots','runner.py',69,'B1/B5/B10 W4/W8/M4/M8/context/RNG/received/next','21CP;84 selected tensors CPU finite/shape/hash','CPU_WEIGHTS_ONLY_CONFIRMED','Independent GPU off/on continuation NOT_TESTED'),
      ('Evaluation/reuse','runner.py',51,'actual selected W; B5/B10 full population and exact current subset','W5/10 cardinality; current/first500/last500 rows identical to full','CPU_IDENTITY_CONFIRMED','Other candidate R/P/N NOT_RECORDED'),
      ('Technical preparation','technical.py',65,'teacher/local/terminal/order/history bounded checks','48679 TECHNICAL_READY; same-process restoration; original targets replay disclosed','REUSED_ACTUAL_GPU_CHECKS','Not every future endpoint or independent restart parity'),
      ('Cap1 admission','control.py',181,'technical afterok then seven-arm array%1','held inspect ArrayTaskThrottle=1; actual max2 /19428sec overlap','DEVIATION','Cause/actor/change time NOT_RECORDED; no scheduler mutation in review'),
      ('Nonfinite/errors','policy.py',39,'NaN/Inf technical error, not quality skip','190finite scores;70terminal commits; no failure.json','SOURCE_CONFIRMED + STORED_EVIDENCE_CONSISTENT','Not proof against unlogged external faults'),
    ]
    table(out/'design-conformance.csv',[dict(requirement=a,file=b,line=c,rule=d,evidence=e,status=f,limit=g) for a,b,c,d,e,f,g in conformance])
    coverage=[
      ('W10/full1000 and all Current', 'COMPLETE_CPU_REDUCED', '7×1000/2000/10000;140 entry/selected panels'),
      ('W5→W10 samefirst500; cohorts/atwrite', 'COMPLETE_CPU_REDUCED', 'Exact identities; all-request and active/superseded separate'),
      ('Candidate E/H/D/strict/selection', 'COMPLETE_CPU_REDUCED', '190declared=190unique;30dynamic;4fixedpolicies unscreened'),
      ('Candidate official RS/PS/NS', 'NOT_RECORDED', 'Only entry and selected canonical pairs; no new forward to fill'),
      ('Snapshots/state', 'PARTIAL_VERIFICATION', '21CP CPU tensors;63hash links; no GPU restart'),
      ('B2–B10 all candidate endpoint materialization', 'CPU_EXACT_HASH', '171weight candidates; starting actual B1CP'),
      ('B1 all candidate reconstruction', 'NOT_TESTED', 'No independent W0 selected tensor file; selected CP verified'),
      ('Final history K/Gram', 'NOT_RECORDED', 'Append source/receipt/M hash verified; K not saved'),
      ('Target/Adam/loss', 'SAVED_NATIVE_COUNTERS', '12000target observations; exact totals below'),
      ('Clamp hits/total F-B/copy ops', 'NOT_RECORDED', 'No inference of iteration hits or total FLOPs from quota'),
      ('S64/Dev128', 'COMPLETE_STORED_REDUCED', 'Controller vs observer split; one reused W0 teacher'),
      ('GPU allocation', 'COMPLETE_EXACT_JOB_AUDIT_WITH_DEVIATION', '8targets once; cap1 exceeded by recorded max2'),
      ('Pure writer/utilization', 'NOT_SEPARATED_OR_NOT_MEASURED', 'Inclusive timers not additive; allocation not utilization'),
      ('Report256/Audit/MMLU/FutureN', 'NOT_MEASURED_NOT_REQUESTED', 'No backfill; future content never selected online'),
      ('FD/ULP/KKT/backbone/off-on', 'NOT_TESTED_THIS_REVIEW', 'Local-z design does not require EP gradient diagnostics'),
      ('Historical arms', 'NOT_REAUDITED', 'No old N4/REFIT4 substitution; prior reports preserved'),
    ]
    table(out/'source-state-coverage.csv',[dict(item=a,status=b,scope=c) for a,b,c in coverage])
    source_link=lambda name,line:f'[{name}:{line}]({os.path.relpath(w/"project/run_scripts/local_z_adaptive_allocation"/name,out)}#L{line})'
    sections=[]
    def add(x):sections.append(x.strip()+'\n')
    add('# Local-z adaptive allocation: 신규 cold 7-arm 상세 사실 리뷰\n\n상태: **7/7 완료 산출물의 CPU 리뷰 완료 — 운영 cap1 이탈 별도 확인**. 신규 GPU/모델/evaluator 실행 0.\n\nInstruction `ODEEDIT-S06-LOCAL-Z-SEVENARM-DETAILED-REVIEW-SH4-V1`; 2026-09-17 SH4. 과학적 우월성·인과 기여율·후속 선택은 본 보고의 판정 범위가 아니다.')
    add('## 1. 먼저 확인한 사실과 검증 경계\n\n준비48679와 array48680_[0–6] 모두 scheduler COMPLETED/exit0:0이다. 원 NLL pair를 독립 재집계해 7개 actual W10의 RS1000/PS2000/NS10000을 확인했다. 70commit,63인접 W/M/context/RNG/received 연결,110history append 기록,21CP가 존재한다. Scheduler 완료와 아래 CPU 산출물 검산은 별개의 확인이다.\n\n원 agent의 2026-09-16T11:36:50.122707+00:00 PENDING/actual gate NOT_OBSERVED 기록은 불변이다. initial-gate 파일과 terminal을 이번 recall에서 사후 확인한 것이며 과거 GPU PASS로 소급하지 않는다. 준비 단계의 실제 GPU 검사 기록을 재사용하지만 이번 CPU 검산을 model-level 재실행·GPU off/on continuation PASS로 부르지 않는다.\n\n**운영 이탈:** 봉인된 제출은 cap1/array%1이지만 지정 job interval의 최대 동시 GPU 할당은2, cap1 초과19,428초(5.3967시간)였다. 원인/행위자/변경 시점은 보존 증거에 없다. 이번에는 정확 job들만 한 번 조회했고 job/정책을 수정하지 않았다. 수치 결과는 보존하되 cap 준수 또는 통제된 속도 비교가 확인됐다고 쓰지 않는다.')
    add(md(['Arm','Job','Scheduler','GPU초','GPUh','commit/요청'],[[r['arm'],r['job'],r['state'],r['GPU_seconds'],float(r['GPU_hours']),'준비 only' if r['arm']=='PREPARATION' else '10 / 1000'] for r in sched]))
    add('## 2. actual W10 / first1000 주표\n\nRS/PS는 new NLL < true NLL, NS는 true NLL < new NLL이며 ties는 실패다. TF-strict/argmax token은 별도다. 분모를 accepted-only로 바꾸지 않았고 동일1000을 7arm에 공유한 **7000 arm-request observations / unique1000**, 새70batch다. 과거 warm N4/REFIT4 재사용이 아니다.')
    add(md(['Arm','RS /1000 (%)','PS /2000 (%)','NS /10000 (%)','ΔRS pp','ΔPS pp','ΔNS pp'],[[r['arm'],f"{r['RS_count']} ({r['RS_percent']:.2f})",f"{r['PS_count']} ({r['PS_percent']:.2f})",f"{r['NS_count']} ({r['NS_percent']:.2f})",r['RS_vs_N4_pp'],r['PS_vs_N4_pp'],r['NS_vs_N4_pp']] for r in finalrows]))
    add('![최종7arm](figures/final-seven-arm.png)\n\n첫 전달 표 [first-final-table.csv](first-final-table.csv)는 그 시점의 STATE_AUDIT_PENDING 라벨까지 보존했다. 전체 검산 후 표는 [final-seven-arm.csv](final-seven-arm.csv)다. N4와TD는 단순 동점이 아니라 전10batch W4/W8/M4 hash 및 최종 원 NLL 행이 같았다. M8은 달랐다: N4는 미사용 zero, TD는 선택 N4일 때도 매batch append했다. 이를 terminal 제안의 선택 성공이나 추가 capacity의 효능으로 해석하지 않는다.')
    add('### TF-strict/token 정의를 분리한 최종표\n\nR/P는 desired=new, N은 desired=true token을 사용했다. Two-P strict는 2개 paraphrase 모두 strict인 요청 수/1000이지 PS 분모2000과 다르다. 아래 strict는 저장된 token-correct/count의 일관성 검산이며 새 forward는 아니다.')
    add(md(['Arm','R strict/1000','P strict/2000','two-P strict/1000','N true strict/10000','R token correct/total','P token correct/total'],[[a,final[a]['RS']['desired_TF_strict'],final[a]['PS']['desired_TF_strict'],final[a]['PS']['two_P_strict'],final[a]['NS']['desired_TF_strict'],f"{final[a]['RS']['desired_token_correct']}/{final[a]['RS']['desired_token_count']}",f"{final[a]['PS']['desired_token_correct']}/{final[a]['PS']['desired_token_count']}"] for a in ARMS]))
    add('## 3. 핵심 paired 비교: 증가와 손실을 함께\n\n아래는 왼쪽 treatment−reference의 같은 W10 요청/prompt/target 비교다. Lost는 reference 성공→treatment 실패, gained는 반대다. RS/PS/NS 원분모는 각각1000/2000/10000. 95% CI는 같은 request 안의 P2/N10을 묶어 1000request cluster를 2000회 bootstrap(seed20260917)한 기술적 구간이다. 10batch를 독립 반복실험으로 취급하지 않았고 fixed order/요청 간 공유 사실·학습 의존성이 남는다. CI를 arm 탈락·선택 또는 인과확증에 사용하지 않는다.')
    maincontrasts=[r for r in paired if r['treatment']=='LD' or r['treatment']=='T75' and r['reference']=='L75']
    add(md(['대조','지표','Δpp','lost','gained','95% request-cluster CI'],[[r['treatment']+'−'+r['reference'],r['metric'],float(r['delta_pp']),r['lost'],r['gained'],f"[{float(r['CI95_low']):.3f}, {float(r['CI95_high']):.3f}]"] for r in maincontrasts]))
    add('LD−N4: PS는 −25/2000(48 lost,23 gained), NS는 +284/10000(120 lost,404 gained)다. RS 총999는 같지만 lost1/gained1로 보존 문항은 다르다. T75−L75는 PS +23/2000과 NS −1066/10000, RS −3/1000이 함께 관측된다. 어느 하나로 전체 정책을 우월/열등하게 승격하지 않는다. 모든 arm−새N4 및 strict/two-P 전이는 [paired-policy-comparisons.csv](paired-policy-comparisons.csv), [strict-transitions.csv](strict-transitions.csv)에 있다.')
    add('### 요청 손실의 NLL 꼬리\n\n아래 p95/p99는 treatment−reference의 원 signed NLL 차이 분포다. 양수는 해당 label likelihood 악화이며 NS의 desired는 true, RS/PS desired는 new다. new/true를 혼합하지 않는다. 각 요청의 원 pairs는 local-only 원 artifact에 유지했다. 평균/중앙/p90/p95/p99와 margin 변화·양방향 전이는 CSV 전부에 남겼다.')
    add(md(['대조','지표','Δnew mean','Δnew p95','Δnew p99','Δtrue p95','Δtrue p99','desired NLL 악화행'],[[r['treatment']+'−'+r['reference'],r['metric'],float(r['new_nll_delta_mean']),float(r['new_nll_delta_p95']),float(r['new_nll_delta_p99']),float(r['true_nll_delta_p95']),float(r['true_nll_delta_p99']),r['desired_NLL_harmed']] for r in maincontrasts if r['reference'] in ('N4','L75')]))
    add('## 4. Current, at-write, retention, overwrite\n\n매batch current100은 서로 다른 모집단이다. pooled at-write1000은 10개 다른 endpoint에서 얻은 관측이며 W0나 final W10 평가가 아니다. Entry→at-write는 동일batch의 쓰기 전후, at-write→W10은 이후 trajectory 변화다. B10은 future exposure0이며 동일 endpoint 재사용 행의 대조이다. 중간 실패/회복 최초 시점은 관측하지 않은 구간에서 보간하지 않는다.')
    add('![current 추이](figures/current-batch-curves.png)\n\n전140개 entry/selected Current panel의 mean/median/p90/p95/p99 new/true NLL, signed margin, strict/token은 [batch-metrics.csv](batch-metrics.csv). cohort별 at-write→W10은 [cohorts.csv](cohorts.csv)에 있고 early B1–3 / middle B4–7 / late B8–10 등은 그 관측 행에서만 집계할 수 있다.')
    for contrast,title in [('W5_FIRST500_TO_W10_SAME500','W5 first500 → W10 같은500'),('POOLED_ATWRITE_TO_W10','각 요청 at-write → W10 전체1000')]:
        add('### '+title)
        rr=[r for r in trans if r['contrast']==contrast]
        add(md(['Arm','지표','before','after','lost','gained','before성공 분모','before실패 분모'],[[r['arm'],r['metric'],r['before'],r['after'],r['lost'],r['gained'],r['conditional_loss_denominator'],r['conditional_recovery_denominator']] for r in rr]))
    add('W5→W10 first500에서 LD의 PS는945로 같지만 lost4/gained4, N4는961→958이지만 lost8/gained5다. T75의 pooled at-write PS1942→1942도 lost13/gained13이다. 같은 총점은 같은 문항 유지가 아니다. NS에서는 neighborhood prompt의 true/new 상대 likelihood 비교가 변하므로 NLL true/new 양쪽을 함께 본다.\n\nRaw(subject,relation)의 최종 target 문자열 기준 ACTIVE999/SUPERSEDED1이며 ALL1000을 유지했다. 같은 target 재발행은 ACTIVE다. received ledger는 strict 성공 여부와 무관하게 모든 도착 event를 포함한다. 교체된1요청은 [population-metrics.csv](population-metrics.csv)의 별도 SUPERSEDED 분모와 [paired-transitions.csv](paired-transitions.csv)의 ACTIVE/SUPERSEDED 전이로 분리했다. superseded target의 손실을 모두 오류나 forgetting으로 단정하지 않는다. W0 관측은 공통 준비에서 이미 저장된 first1000만 한 번 재사용했다. W10 first500/last500은 같은 full1000 행을 exact subset으로 재사용했으므로 분모를 중복 합산하지 않는다.')
    add('### 모집단별 최종 수준\n\n다음 표는 상태·population을 명시한다. W5 first500과 W10 first500은 같은 요청의 다른 state이며 last500은 별도 cohort다.')
    add(md(['Arm','state/population','RS count/d','PS count/d','NS count/d'],[[a,label]+[f"{next(r for r in pop if r['arm']==a and r['population']==label and r['status']=='ALL' and r['metric']==tag)['numerator']}/{next(r for r in pop if r['arm']==a and r['population']==label and r['status']=='ALL' and r['metric']==tag)['denominator']}" for tag in ('RS','PS','NS')] for a in ARMS for label in ('W5_FIRST500','W10_FIRST500','W10_LAST500')]))
    add('## 5. 설계 → frozen 코드 → 실제 저장 증거\n\n실행 source는 `'+EXEC+'` / tree `75a96b2e2122d2be6b096af12c22df8a4365a8e1`이며 이번 분석 코드는 별개다. 아래 source 링크는 파일 탐색용이며 exact frozen SHA/함수 시작·끝 줄은 [source-functions.csv](source-functions.csv)에 봉인했다. 함수명이나 문서 선언만으로 실행 PASS를 대신하지 않는다.')
    add('공통 capsule은 Llama3-8B-Instruct revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, torch2.9.1+cu128 / transformers4.44.2, FP32/eager, matmul·cuDNN TF32off, seed20260916이다. Native L2=1, decay=.5, clamp=.75, lr=.1, max25loss/24Adam, KLfactor=.0625, loss layer31을 유지했다. Writer tokenizer는 add_bos_token=False, pad=eos/right padding이며 evaluator tokenizer는 별도 원 설정을 사용했다. canonical evaluator MB16은 재사용했다. Context text SHA aa169bc574115e932462517a4e9e8aa4e183017721c74b5a99e6e2a4b6fa7a09, actual token IDs SHA cfb6efea82ae828342f13fdd24c015dacf0275f93d47839043f907d6115cfc6f, common RNG SHA e458c35c6067bf08f8d196fc391febe29beb5177d5f5be7235c00033431c2a67을 모든 arm에 결속했다. Native editor SHA79da927aad5ab817fd008c5958768adcd00556989a8efbaa2c4bdc80d8fc842e는 원본 그대로다.')
    add(md(['요구사항','실제 함수/줄','설계/식','저장 관측','판정','한계'],[[a,source_link(b,c),d,e,f,g] for a,b,c,d,e,f,g in conformance]))
    add('### LOCAL/TERMINAL과 state의 실제 의미\n\nLOCAL은 자기 batch entry의 L4 native target/solve로 N4 endpoint를 만들고 actual FP32 a4 partial을 적용한 뒤, a4=.75와1 상태마다 L8 target을 각각 fresh 계산한다. a8=.5/1 사이에는 동일 a4의 fit만 공유한다. REFIT4는 L4 .75partial에서 같은 L4를 다시 fresh fit한다. cache는 batch 생성 내부에만 있고 state/layer/source/hparams/context/RNG를 결속한다.\n\nTERMINAL은 자기 entry에서 Z8을 요청별 한 번 계산하고 고정한다. writer4의 residual도 Z8−h8(entry), writer8은 Z8−h8(partial)이다. Z8−h4가 아니다. frozen native BLUE의 key/readout/repeat/directsolve/add AST를 추출하고 readout8을 명시했다. inverse·Cholesky·대칭화로 바꾸지 않았다. 저장50개 residual의 정확 CPU 차이가 이 연결을 지지하나 neural forward나 solve를 새로 재현한 검산은 아니다.\n\n물리 W4/W8 전체 parameter copy를 통해 모든 token에 update가 적용된다. gate는0/1 exactcopy, .5/.75는 CPU FP32 U+g(V−U)다. gate합1 제약은 없고 native RHS divisor와 bit-exact라고 가정하지 않았다. TD는 terminal 제안6개+own-entry commonN4의 controller이며 pure-terminal 정책이 아니다. 고정 L75/T75는 각각 (.75,1) endpoint를 그대로 commit하고 dynamic 품질 screen을 적용하지 않는다.')
    add('### E/H/D와 online 입력 분리\n\nE는 native rewrite training mask의 desired token평균→contexts평균→100request평균이다. canonical rewrite strict는 별도로 기록한다. H는 Past64 canonical rewrite의 mean desired NLL이다. D는 고정 W0 full-vocabulary teacher에 대한 KL(p0||pW), vocab sum→128 scored position평균→64문서평균이다. 자연256+BOS1, logit[128,256), scored input[129,257), vocab128256을 기존 C4 계약 그대로 재사용했다.\n\n선택 함수에는 candidate E/H/D와 current/Past strict ID만 들어간다. official P/N은 entry/selected observer, Dev128은 B5/B10 observer이며 후보·target·Past 선정에 쓰지 않는다. no-plateau/strict-only shadow는 이미 얻은 점수의 같은-state 재선택이며 commit하지 않았다. 기존 EP gradient route/alpha/quality screen을 local-z 정책으로 호출하지 않았다.')
    add('## 6. selector 전 후보와 실제 선택\n\nFeasible은 E(c)≤max(E(ownN4),.05)+1e-4 및 ownN4 current strict ID subset이다. Past가 있으면 H(c)≤H(ownN4)+1e-4와 Past strict subset도 필요하다. Feasible 중 Dmin+1e-6 이내 후보를 ownN4→actual L8delta0→작은 concat(ΔW4,ΔW8) Frobenius norm→ID 순으로 선택한다. 190declared 후보는190distinct materialization이고 실제 score reuse0이었다. dynamic150후보의 조건·재선택은 모두 저장selected와 일치했다. 같은 strict count만을 ID 보존으로 대체하지 않았다.')
    selection_summary=[]
    for a in ARMS:
        rr=[r for r in sel if r['arm']==a];selection_summary.append([a,'; '.join(f'{k}×{v}' for k,v in s['selection_counts'][a].items()),sum(r['shadow_no_plateau']!=r['selected'] for r in rr if r['shadow_no_plateau']!='NOT_APPLIED'),sum(r['shadow_strict_only']!=r['selected'] for r in rr if r['shadow_strict_only']!='NOT_APPLIED')])
    add(md(['Arm','실제선택 10회','no-plateau 선택차이','strict-only 선택차이'],selection_summary))
    add('![선택](figures/dynamic-selections.png)\n\nL4D는 B8만 .75L4를 선택했고 나머지9회N4, LD는 B4만N4와 나머지9회(.75,.5), TD는10회모두N4다. LD no-plateau shadow의 차이는 현재 trajectory에서의 점수 재선택일 뿐, 그 정책을10batch 실행한 결과가 아니다.')
    add(md(['동적arm','선언후보','feasible','부적격','Current mean','Current strict','Past mean','Past strict'],[[a,len(rr),sum(r['feasible']=='True' for r in rr),sum(r['feasible']=='False' for r in rr)]+[sum(reason in r['reasons'].split(';') for r in rr) for reason in ('CURRENT_MEAN','CURRENT_STRICT_IDS','PAST_MEAN','PAST_STRICT_IDS')] for a in ('L4D','LD','TD') for rr in [[r for r in cons if r['arm']==a]]]))
    add('탈락 사유는 중복 가능하므로 사유별 수를 총부적격 수로 합산하지 않는다. TD의60 terminal 제안 중7개는 조건을 통과했지만 실제선택은N4였다. 전190후보의 gate/E/H/D/strict집합 손실수/실제norm/feasibility/reason/선택을 [candidate-metrics.csv](candidate-metrics.csv), [constraints.csv](constraints.csv), [selection.csv](selection.csv)에 공개한다. 원 strict ID집합과 요청별 NLL은 local candidate JSON에 유지되며 reducer가 exact subset을 확인했다. NaN/Inf를 단순 부적격으로 버리는 toy 분기는 production choose에 없고, 유효한 finite 품질탈락과 기술오류를 구분한다.')
    add('### Worked examples: 실제 변수와 선택 근거')
    for a,b in [('LD',1),('LD',4),('TD',1),('L75',1),('T75',1)]:
        rr=[r for r in cand if r['arm']==a and int(r['batch'])==b];cc={r['candidate_id']:r for r in cons if r['arm']==a and int(r['batch'])==b}
        add(f'#### {a} B{b:03d}\n\n'+md(['후보','a4/a8','E','H','D64','strict current/Past','실제 concat norm','조건','선택'],[[r['candidate'],r['a4']+'/'+r['a8'],float(r['E']),None if not r['H'] else float(r['H']),float(r['D']),r['current_strict']+'/'+r['past_strict'],float(r['action_norm']),cc[r['candidate']]['reasons'] or ('feasible' if a in ('LD','TD') else 'fixed/no screen'),r['selected']] for r in rr]))
    add('LD B1은 공통 W0/M0에서 시작하고 Past가 비어 있다. ownN4 E≈.0034046이므로 plateau ceiling은.0501이며 선택 (.75,.5)의 E≈.0124145는 N4보다 크지만 규정 ceiling 이내다. selectedD≈.00125266은 ownN4D≈.00172091보다 작았다. 평균 E 또는 각 요청 NLL 무악화를 보장하는 설계가 아님을 이 사례가 보여준다. 선택 후 M4/M8 각1회 append하고 actual B1CP→B2entry hash가 연결된다.\n\nLD B4의 N4 선택은 score/constraint 표의 그대로다. 더 작은 D를 갖는 후보의 부적격 사유도 표에 남겼다. TD B1 역시 같은W0에서 출발하지만 terminal 제안이 아니라 commonN4를 선택했고 M8까지 append했다. L75/T75 B1은 같은gate(.75,1)이나 local versus terminal target/readout 경로가 다르다. B2 이후에는 각자의 누적 state까지 달라지므로 동일state 단일인자 인과효과로 확대하지 않는다.\n\n선택된 actual endpoint에서 mean E/strict/Past를 보호했다는 사실과 PS·NS·모든 old 요청 보존은 별개다. 후보별 canonical rewrite desired NLL의 p95/p99 및 ownN4 대비 악화 요청 수는 candidate CSV에 있다. 다른 후보의 official R/P/N true/new pairs는 저장되지 않아 same-state N4→selected R/P/N 전체를 추정하지 않았다.')
    add('### Past64/received ledger\n\nraw(subject,relation)별 최신 event 하나를 남긴 뒤 현batch가 overwrite할 fact를 제외하고, `LZ-ALLOC-v1|20260916|past|`+stable_event_id의 UTF-8 SHA256 우선순위로64개를 고른다. stable ID는 ordinal/case_id/subject/relation/target의 canonical ASCII JSON SHA다. 모든70list를 독립 재생성했고 B1은0, B2–10은64이며 각arm 입력목록은 같았다. target 성공여부·미래score·과거Historical128·공식P/N은 이 선정에 들어가지 않는다. received1000과 accepted/strict-success 수는 다른 개념이다.')
    add('## 7. S64 선택지표와 Dev128/NS 관측 분리')
    add(md(['Arm','W5 S64','W10 S64','W5 Dev128','W10 Dev128','W10 Dev naturalNLL','W10 NS%'],[[a]+[float(next(r for r in sel if r['arm']==a and int(r['batch'])==b)['D']) for b in (5,10)]+[float(next(r for r in dev if r['arm']==a and int(r['batch'])==b)['D']) for b in (5,10)]+[float(next(r for r in dev if r['arm']==a and int(r['batch'])==10)['natural_NLL_mean']),float(final[a]['NS']['percent'])] for a in ARMS]))
    add('![선택KL과paired](figures/selector-and-paired.png)\n\nS64는 동적 선택에 사용된 개발 패널이다. Dev128은 고정 별도 observer이지만 한 번의 fixed order 개발 실행이며 대규모 blind test가 아니다. KL 감소와 NS 성공률 증가는 정의부터 다르다. Report256/Audit/MMLU/FutureN 독립평가는 NOT_MEASURED, 기존정본 밖 추가 teacher/reference/평가는 수행하지 않았다.')
    add('## 8. checkpoint·history·native counter 실물 검산\n\n21CP의84개 W/M tensor를 CPU weights_only로 읽어 FP32/shape/finite/원 raw-byte SHA를 대조했다. W4/W8 shape[4096,14336], M4/M8 shape[1,14336,14336]. dtype/shape-header+bytes SHA를 별도로 남겨 두 convention을 혼동하지 않았다. checkpoint는 source/model/P/common/context/RNG/received/nextordinal까지 연결됐다.\n\nB1actual selectedCP에서 출발해 B2–B10 저장native weight와 FP32 gate로171후보 endpoint를 재구성했고 기록된actual weightSHA와 일치했다. B5/B10selectedCP와도 정확히 일치했다. B1 전체 후보는 독립W0weight파일이 없어 역산하지 않았으며 B1selectedCP검산만 했다. FP32 delta 차분만의 exact replay, 미저장 finalizer keys로부터 Gram history 재생성, 독립 GPU off/on restart는 주장하지 않는다.')
    add(md(['항목','관측'],[['CP',t['CP_count']],['CP tensors',t['checkpoint_tensor_count']],['재구성 candidate B2–10',t['reconstructed_candidate_count']],['원 target captures',t['target_calls']],['actual Adam',t['Adam']],['actual loss eval',t['loss']],['준비+과학 raw members',t['members']],['검토 inventory bytes',t['bytes']],['inventory GiB',t['bytes']/(1<<30)]]))
    add('각 CP의 tensor norm/hash/bytes와 M diagonal 비감소 검사는 [checkpoint-tensors.csv](checkpoint-tensors.csv), actual gate reconstruction norm의 CPU scalar 반올림 차이는 [candidate-reconstruction.csv](candidate-reconstruction.csv), native layer/readout/target inventory는 [native-fit-tensors.csv](native-fit-tensors.csv), terminal R/H8/Z8 norm은 [terminal-residual.csv](terminal-residual.csv)에 있다. source와 receipt가 지지하는110append는 L4총70 + L8총40이다. REFIT4는두fit이지만 M4append10이며 LD/TD의a8=0/N4선택도M8append한다.\n\n비선택 parameter는 runtime pointer/version/hooks/buffers/gradient/mode guard를 사용했다. 이는 모델전체 bytes를 독립 hash한 검증과 다르다. 모든190score가finite이고 failure.json/미완료prefix가 없다는 사실을 기록하되 미저장 내부 tensor까지 검증했다고 확대하지 않는다.')
    add('## 9. 실측 비용: 계획 quota와 분리\n\n과학7chain allocation합 '+str(s['science_GPU_seconds'])+'GPU초('+f"{s['science_GPU_seconds']/3600:.6f}"+'GPUh), 준비 '+str(s['preparation_GPU_seconds'])+'초('+f"{s['preparation_GPU_seconds']/3600:.6f}"+'GPUh)다. 합계 '+str(s['science_GPU_seconds']+s['preparation_GPU_seconds'])+'GPU초이며 새 review GPU는0이다. 원teacher47592의98초는 과거 재사용 비용으로 이번합에 다시 더하지 않았다. teacher는 재생성되지 않았다. 준비700fresh target+400saved target replay/13solve는 과학12000target/150solve와 구분한다. 준비source target replay는 write 연결검사용이며 과학분모에 포함하지 않았다.')
    actualcounter={a:{k:sum(int(r[k]) for r in target_summary if r['arm']==a) for k in ('target_calls','Adam','loss','early_stop','zero_step')} for a in ARMS}
    add(md(['Arm','target calls','actual Adam','loss eval','early-stop requests','0-step','solve','history','distinct후보'],[[a]+[actualcounter[a][k] for k in ('target_calls','Adam','loss','early_stop','zero_step')]+[next(r for r in costs if r['arm']==a)[k] for k in ('solve','history','candidate_distinct')] for a in ARMS]))
    add('12000target request-calls는 unique12000이 아니다. 최대288000Adam/300000loss/150solve/190후보는 계획이며 위 저장counter가 실측이다. early-stop은 native actualupdates<24이고 0Adam도 이후 writer action0을 뜻하지 않는다. iteration별 clamp-hit, 전체 model-forward/backward/FLOPs/copy operation 수는 NOT_RECORDED이며 최종 anchor/radius에서 추정하지 않았다. candidate E forward는100request groups/후보, S64는64doc forward/후보, canonical current는MB16의7groups/후보다. 단위를 혼합해 단일forward수로 합치지 않는다.')
    add(md(['Arm','program초','nativefit(local)포함초','proposal포함초','candidate scoring포함초','history포함초','entry/selected evaluator초','state I/O포함초','peak GPU allocated GiB','peak host process GiB'],[[r['arm'],float(r['program_seconds']),None if not r['local_native_fit_runtime'] else float(r['local_native_fit_runtime']),float(r['proposal_inclusive_inclusive']),float(r['candidate_scoring_inclusive']),float(r['history_finalization_inclusive']),f"{float(r['entry_observer_inclusive']):.3f}/{float(r['selected_observer_inclusive']):.3f}",float(r['state_IO_inclusive']),float(r['peak_GPU_allocated_bytes'])/(1<<30),float(r['peak_host_RSS_KiB'])/(1<<20)] for r in costs]))
    add('Timer들은 nested이며 위 열을 총비용으로 합산하지 않는다. proposal에는native target/solve/restore/hash/native-fit파일 I/O가 포함되고 candidate_scoring에는E/H/D 외 snapshot/hash/restore/JSON I/O가 포함된다. state_IO는selected-delta/CP/hash/CPUreload의 포함시간으로 순수disk latency가 아니다. terminal key/readout/solve는 receipt의combined seconds만 있어 PURE_WRITER=NOT_SEPARATED; compute-components의 local_* 및 native compute_z_seconds는 LOCAL receipt만의 값이며 T75의0을 target계산0으로 읽으면 안 된다. TERMINAL target시간과 실제Adam/loss는 [terminal-target-cost.csv](terminal-target-cost.csv), combined writer시간은 [terminal-residual.csv](terminal-residual.csv)로 별도 보완했다.\n\nS64 timer의 teacher streaming은 nested subcomponent다. LD/TD의 teacher-read와 scoring 시간이 커도 원인을 I/O contention 등으로 확정하지 않는다. 실제overlap이 있어 samehost라도 controlledspeedup비교가 아니다. GPU allocated/reserved와 hostprocesspeak, Slurm MaxRSS는 서로측정범위가 다르며 allocation은 utilization이 아니다. 실제 disk는 raw manifest에서 준비/arm·CP/source단위를 구분하고 fullmodel/teacher기존 bytes를 새산출물로 중복가산하지 않았다.')
    add('### cap1 위반 구간 — 과학 성능 gate와 별도\n\n제출 held receipt는 ArrayTaskThrottle=1, afterok48679였고 공개source도%1이다. 이후실제두할당의겹침은 아래와 같다. scheduler row만으로 throttle 변경자·사유를 추정하지 않으며 권한 밖 전체audit나새query를 하지 않았다.')
    overlap=rows(data/'allocation-intervals.csv')
    add(md(['Start KST','End KST','동시arm','GPU','초'],[[r['start_KST'],r['end_KST'],r['arms'],r['GPUs'],r['seconds']] for r in overlap if r['above_cap1']=='True']))
    add('## 10. 남은 한계·반대 결과·검증 coverage\n\n전7arm의 유효한 낮은PS/높은NS손실/많은N4선택을 삭제하지 않았다. 고정T75는선택screen이없는정책이며낮은NS를technicalfailure로표기하지않는다. LD의S64/Dev 감소와PS손실, TD의N4동일결과·추가비용을 함께 보존했다. 각gate·norm은실제행동크기이며 기능기여율 또는 capacity증명이 아니다.\n\n이번 데이터는한fixedorder·하나의seed20260916이며 per-batch policy차이가누적된다. historicalN4/REFIT4/warmW50 결과로이번새coldchain을대체하지않았다. 과거baseline을별도재감사하거나새first1000행을합성하지않았다. 기존 [CAKE/baseline 정본](../../cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v2/diagnostic-report-ko.md), [cap sweep 정본](../../ep-tw1-alpha-cap-sweep-2026-09-15-v1/completed-review-v2/diagnostic-report-ko.md)은다른seed/hparams/controller/평가범위의보존참고이며이번주표와혼합하지않는다.')
    add(md(['항목','확인수준','범위/미측정'],coverage))
    add('## 11. 재현·identity·publication\n\n원 source/실행/실패/pause/기존보고bytes는 read-only로 보존했다. 실행 archiveSHA `6378df3a237d5a6c489b87c90f6d99c0221b409857632852916f2f5bffa7f901`, lockSHA `093bb13dd6b42b8f2f8b8478248b067add1d94533547afa8ca9af416e6f14629`. 분석기준 main `31c02006affc2f94f64ff43539c8892206ca4b17`; 분석source HEAD/tree는 analysis-manifest에 별도 봉인한다.\n\n공통 C4 reference `f5791dd3c986261a252796bd5d7293ece46339d261609449dcfe674970bbfde0`, teacher manifest `f81b798f44ce626ac1e2e402ca7438363b1dd60f5681cd60f5681ec0ac17b9e92d096761a`는 아래 input-manifest의 exact 실제값을 정본으로 한다. C4sample seed20260915와method seed20260916은다르다. modelrevision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, datasetSHA `3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1`, wholeorder `5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729`, first1000root `40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd`.\n\n[REPRODUCE.md](REPRODUCE.md)의CPU명령은새reviewattempt에쓴다. scheduler-once는저장receipt를재사용하며재현시재조회하지않는다. PNG는저장CSV로직접생성하고재생성SHA를검산했다. 원raw/prompt/teacher/gradient/tensor/fullstdout은local-only, Git에는source·집계·path/SHA/size만게시한다. 자체분석및원runtime과독립구현한reducer검산이며별도reviewer agent를사용했다고주장하지않는다.\n\n**TASK_COMPLETE_STOP / automatic_resume=false / monitoring_active=false.** 추가실험·repair·selector변경·GPU검산·다른pausedtask재개없음.')
    # Exact identity inserted from the actual capsule, never hand-transcribed as authority.
    report='\n'.join(sections).replace('f81b798f44ce626ac1e2e402ca7438363b1dd60f5681cd60f5681ec0ac17b9e92d096761a',cold['teacher_manifest']['sha256'])
    write(out/'diagnostic-report-ko.md',report)
    write(out/'REPRODUCE.md','''# CPU 재현과 evidence 범위

Python `/data/janghj/EasyEdit/.venv/bin/python`; 실행 모듈이 아닌 아래 analysis만 호출한다. 원 output은 read-only다.

```bash
python -B -m unittest project.run_scripts.local_z_adaptive_allocation.analysis.test_analysis -v
python -B -m project.run_scripts.local_z_adaptive_allocation.analysis.reduce --attempt analysis-new-recall
python -B -m project.run_scripts.local_z_adaptive_allocation.analysis.tensors --attempt tensor-new-recall --reuse-inventory /data/janghj/ODE-edit/local/local-z-adaptive-allocation/20260916-v1/review-20260917-v1/tensor-audit/raw-inventory.json
python -B -m project.run_scripts.local_z_adaptive_allocation.analysis.plot --data REPORT_DIRECTORY --out NEW_FIGURE_DIRECTORY
```

실제 이번 metric 출력은 analysis-r2, tensor는 tensor-audit, supplemental은 supplement-v1이다. bootstrap의 scheduler 조회는 이미1회 완료했으므로 재현시 실행하지 않는다. `audit`/`publish`는 create-once 경로를 사용하므로 원폴더가 있으면 종료한다. 새허용reviewnamespace에서만 경로를 정해 재실행한다. 보고 구성은 publish.py이며 기존CSV를복사/집계하고raw수치를변경하지않는다.

Raw inventory 재사용은 fullSHA가 같다는 새확인이 아니라 과거이검토의fullSHA+현재size 재사용이다. 독립 fullSHA를 다시 원하는 경우 reuse-inventory를빼되불필요중복해시를기본요구하지않는다. 모델·teacher기존immutable자산은priorfullSHA+stat재결속이며새모델검증이아니다. figures는matplotlib코드로PNG재생성후byteSHA대조했다. 분석코드수정전실패analysis-v1은보존되며 runtime/science 재실행0.
''')
    save(out/'evidence-reuse-manifest.json',dict(members=reuse,large_assets='PRIOR_FULL_SHA_CURRENT_STABLE_STAT',old_pending_immutable=identity(ROOT/'resume-manifest.json'),
        current_CPU=dict(metrics=s,tensors=t,supplement=u),analysis_bug='analysis-v1 duplicate keyword reasons; corrected reducer only; original raw unchanged',new_GPU=0))
    print(json.dumps(dict(report=identity(out/'diagnostic-report-ko.md'),files=len(list(out.rglob('*')))),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worktree',required=True);a=p.parse_args();publish(a.worktree)
