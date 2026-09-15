"""Generate raw-free review tables/reports/figures from audited CPU outputs."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import statistics
from .begin import WT,LOCAL,TASK
from .cake import RAW,ATTEMPT,OLDREPORT,csvread,label
from project.run_scripts.bg_tw_reference.ep_tw import review_nogate as r
from project.run_scripts.bg_tw_reference.ep_tw.build_sweep_report import mdtable

BASE=WT/'experiment-reports/servers/server4'
CAKE=BASE/'cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v1'
CAP=BASE/'ep-tw1-alpha-cap-sweep-2026-09-15-v1/completed-review-v1'
OVER=BASE/'completed-experiments-review-2026-09-16-v1'
SOURCE=LOCAL/'analysis/source-audit-v2'

def write(p,text):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:f.write(text+'\n')

def copy(p,q):
    q=Path(q);q.parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('rb') as src,q.open('xb') as dst:shutil.copyfileobj(src,dst)

def schedule():
    s=r.read(LOCAL/'receipts/scheduler-terminal-once.json');rr=[x.split('|') for x in s['sacct']['stdout'].splitlines()]
    rows=[]
    for arm,job in [('CAKE','48101'),('CAP10','48148'),('CAP100','48149'),('NORM_ONLY','48150')]:
        x=next(x for x in rr if x[0]==job);assert x[3:5]==['COMPLETED','0:0']
        step=next(x for x in rr if x[0]==job+'.batch')
        rows.append(dict(arm=arm,job=job,state=x[3],exit=x[4],start=x[6],end=x[7],allocated_GPU_seconds=int(x[8]),
            queue_seconds=(datetime.fromisoformat(x[6])-datetime.fromisoformat(x[5])).total_seconds(),MaxRSS=step[10],GPU_count=1))
    edges=sorted([(x['start'],1) for x in rows]+[(x['end'],-1) for x in rows]);active=peak=0
    for _,delta in edges:active+=delta;peak=max(peak,active)
    assert peak==2
    return rows,peak

def stage():
    CAKE.mkdir(parents=True,exist_ok=False);OVER.mkdir(parents=True,exist_ok=False)
    for p in (LOCAL/'analysis/cake-r3').iterdir():
        if p.is_file():copy(p,CAKE/p.name)
    for p in SOURCE.iterdir():
        if p.is_file():copy(p,OVER/p.name)
    con=csvread(SOURCE/'design-conformance.csv')
    r.table(CAKE/'design-conformance.csv',[x for x in con if x['family']=='CAKE'])
    r.table(CAP/'design-conformance.csv',[x for x in con if x['family']=='EP_SWEEP'])
    for name in ('CAKE-log-target-counters.csv','CAKE-layer-weights.csv'):copy(SOURCE/name,CAKE/name)
    copy(WT/'experiment-reports/servers/server4/cake-native-lifelong-b100x100-2026-09-15-v1/compatibility.json',CAKE/'compatibility.json')
    copy(CAKE/'CAKE-B10-first1000.csv',CAP/'CAKE-B10-first1000.csv')
    # Existing scheduler snapshot reformatted only; no new scheduler query.
    old=r.read(LOCAL/'receipts/scheduler-terminal-once.json');states=[]
    for arm,job in [('CAP10','48148'),('CAP100','48149'),('NORM_ONLY','48150')]:
        root=Path('/data/janghj/ODE-edit/local/ep-tw1-alpha-cap-sweep/20260915-v1')/arm/'attempt-v1/scientific-v1'
        states.append(dict(arm=arm,job_id=job,terminal=r.read(root/'terminal.json'),failure=None,committed_batches=10))
    r.save(LOCAL/'receipts/sweep-scheduler-adapted.json',dict(time=old['time'],queries={'sacct':old['sacct']},states=states,source_receipt=r.ref(LOCAL/'receipts/scheduler-terminal-once.json'),no_new_query=True))
    rows,peak=schedule();r.table(OVER/'scheduler-cost.csv',rows)
    r.save(OVER/'allocation-overlap.json',dict(maximum_allocated_GPU_overlap=peak,method='HALF_OPEN_START_END_INTERVALS_FROM_EXACT_FOUR_JOB_SACCT',GPU_utilization='NOT_MEASURED',new_review_GPU_seconds=0))
    extra_sweep()
    cake_report()
    overview(rows)

def extra_sweep():
    actions=csvread(CAP/'per-batch-actions.csv');candidate=csvread(CAP/'candidate-details.csv');policy=csvread(CAP/'batch-policy.csv')
    groups=[];mechanism=[]
    for arm in ('CAP10','CAP100','NORM_ONLY'):
        aa=[x for x in actions if x['arm']==arm]
        groups.append(dict(arm=arm,cap_bound=sum(x['cap_active']=='True' for x in aa),norm_bound=sum(x['cap_active']=='False' for x in aa),
            halfspace_active=sum(x['projection_active']=='True' for x in aa),ball_hit_requests=sum(int(x['ball_hit']) for x in aa),
            retractions=sum(float(x['trust_retraction'])<1 for x in aa),post_ball_geC_positive=sum(float(x['ge_dot_C_CPU'])>0 for x in aa),
            CPU_selected_reconstruction_maxabs=max(float(x['selected_reconstruction_CPU_maxabs']) for x in aa),
            unequal_selected_elements=sum(int(x['selected_reconstruction_CPU_unequal']) for x in aa)))
        for x in aa:
            p=Path('/data/janghj/ODE-edit/local/ep-tw1-alpha-cap-sweep/20260915-v1')/arm/f"attempt-v1/scientific-v1/B{int(x['batch']):03d}/policy.json"
            d=r.read(p)['correction'];mechanism.append(dict(x,**{k:d.get(k,'NOT_RECORDED') for k in ('native_proposal_ball_max_excess','postprojection_ball_max_excess','mapped_direction_norm','mapped_correction_norm')},
              **{k:d['projection'].get(k,'NOT_RECORDED') for k in ('kkt_stationarity_norm','kkt_complementarity','first_order_violation_recorded_not_silently_reprojected')}))
    r.table(CAP/'mechanism-summary.csv',groups);r.table(CAP/'mechanism-observed-details.csv',mechanism)
    raw_reasons=[]
    for x in policy:
        if x['selected']!='RAW':continue
        cc=[c for c in candidate if c['arm']==x['arm'] and c['batch']==x['batch'] and c['candidate']!='RAW']
        raw_reasons.append(dict(arm=x['arm'],batch=x['batch'],correction_feasible=sum(c['feasible']=='True' for c in cc),
             reasons='; '.join(c['candidate']+':'+c['reason'] for c in cc),selection_reason=x['selection_reason']))
    r.table(CAP/'RAW-selection-reasons.csv',raw_reasons)
    lines=['## 9. 이번 recall의 설계-실행 상세 보충',
      '이번 review-only recall에서 원문·pause를 다시 결속했다. 2026-09-15 사용자 pause를 보존했고 이번에는 정확 네 job의 scheduler만 한 번 확인했다. CAKE도 별도 완료 리뷰했지만 10k 최종값과 아래1k를 직접 합치지 않는다.',
      '실행 lock의 `numerical_rationale.alpha_cap=1`은 CAP1에서 복사된 **구형 설명 metadata**다. 실제 runner가 읽은 `numerical_policy.alpha_cap_mode/alpha_cap` 및 저장 correction receipt는 각각10/100/disabled-null로 일치한다. 이 metadata 차이를 숨기거나 lock을 수정하지 않았다. `after_gate`의 과거 RUN_THROUGH 문자열도 이후 user pause/현재 review-only 권한을 대체하지 않는다.',
      mdtable(['Arm','cap-bound/norm-bound','halfspace active','ball hit','trust retractions','post-ball <gE,C> >0','CPU W diff max'],[
       [x['arm'],f"{x['cap_bound']}/{x['norm_bound']}",x['halfspace_active'],x['ball_hit_requests'],x['retractions'],x['post_ball_geC_positive'],x['CPU_selected_reconstruction_maxabs']] for x in groups]),
      '30개 batch의 q/gradient norm/cos/coefficient·KKT·direction/C inner product·pre/post ball·actual selected geometry는 mechanism-observed-details.csv에 전부 있다. 반공간의 1차 조건은 ball/FP32 materialization 이후 개별 요청/PS/old 보존 보장이 아니다. 양의 post-ball 내적도 제외하지 않았다. CPU 재구성 일치는 해당 저장 selected W4 범위이며 GPU derivative/전체 모델 replay가 아니다.',
      '### 실제 동작 사례 — 모든 batch 자료는 별도 CSV로 유지']
    for arm in ('CAP10','CAP100','NORM_ONLY'):
        aa=[x for x in actions if x['arm']==arm]
        wanted={1,int(next(x['batch'] for x in aa if x['selected']!='RAW')),int(next(x['batch'] for x in aa if x['selected']=='RAW'))}
        for b in sorted(wanted):
            a=next(x for x in aa if int(x['batch'])==b);p=next(x for x in policy if x['arm']==arm and int(x['batch'])==b)
            cc=[x for x in candidate if x['arm']==arm and int(x['batch'])==b]
            lines += [f"#### {arm} B{b:03d}",
              f"자기 batch-entry→native Vp의 action norm={float(a['native_action_norm']):.8g}. q={float(a['q_CPU']):.8g}, alpha_norm={float(a['alpha_norm']):.8g}, used alpha={float(a['alpha_used']):.8g}. Ball hit={a['ball_hit']}, trust scale={a['trust_retraction']}. 최대 C1/native={float(a['C1_native_percent']):.6g}%, 실제 selected={a['selected']} / {float(a['selected_native_percent']):.6g}%다.",
              mdtable(['후보','E−RAW','D−RAW','strict lost','feasible','선택','사유'],[[x['candidate'],x['E_minus_RAW'],x['D64_minus_RAW'],x['RAW_strict_lost'],x['feasible'],x['selected'],x['reason']] for x in cc]),
              f"선택 후 E {float(p['raw_E']):.9g}→{float(p['selected_E']):.9g}, D {float(p['raw_D64']):.9g}→{float(p['selected_D64']):.9g}. 선택된 실제 weight에서 finalizer1회, history_count={b}, next_ordinal={100*b}. 다음 batch가 있으면 저장 W/M/RNG/ledger 일치가 연결된다. RAW도 native write를 commit한다."]
    lines += ['### Source-backed 매핑',mdtable(['요구','함수 source:line','실물 근거','확인수준'],[[x['requirement'],Path(x['source']).name+':'+x['line'],x['artifact'],x['verification']] for x in csvread(CAP/'design-conformance.csv')]),
      '원 수치 검증은 SKIPPED_USER_DIRECTED/NOT_ESTABLISHED. 이번 source/CPU 산술·actual tensor 확인은 FD/direct/selfKL 재검증이 아니다. Strict lost ID의 숫자/문자열 정렬 차이는 reviewer만 exact set으로 보완했고, runtime의 [12179,18415,9763]은 동일3개 ID였다. 과학적 screen 오류로 오인하지 않았다.',
      'CAKE의 actual B10 first1000 참고값은 CAKE-B10-first1000.csv에만 별도 제공한다(RS989/1000, PS1703/2000, NS8118/10000). Layer/L2/decay/clamp/seed가 달라 단일요인 효과로 해석하지 않는다. CAKE W100/full10k를 이 표에 넣지 않는다.']
    write(CAP/'design-conformance-ko.md','\n\n'.join(lines))

def cake_report():
    s=r.read(CAKE/'summary.json');sa=r.read(SOURCE/'source-audit.json');final=csvread(CAKE/'first-final-table.csv')
    metrics=csvread(CAKE/'seen-prefix.csv');pairs=csvread(CAKE/'paired-transitions.csv');fam=csvread(CAKE/'family-final-comparison.csv');conf=csvread(CAKE/'design-conformance.csv')
    full=[x for x in metrics if x['population']=='ACTUAL_FULL_SEEN']
    lines=['# CAKE_NATIVE — fixed10k B100×100 완료 사실 리뷰',f'Instruction `{TASK}`. Job48101은 scheduler COMPLETED0:0이며 실제 terminal/100commit/99기록연결/10000요청을 별도 확인했다. 제출 당시 PENDING을 과거 initial PASS로 바꾸지 않는다. 새GPU/모델/평가0.',
      '## 1. actual W100/full10000 최종값',mdtable(['metric','n/d','%','desired strict','NLL new mean/p99','NLL true mean/p99'],[[x['metric'],x['numerator']+'/'+x['denominator'],x['percent'],x['desired_strict_n']+'/'+x['desired_strict_d'],x['new_nll_mean']+'/'+x['new_nll_p99'],x['true_nll_mean']+'/'+x['true_nll_p99']] for x in final]),
      'RS/PS newNLL<trueNLL, NS trueNLL<newNLL, tie=failure. 全分모10000/20000/100000, ties0. R strict9170/10000, P strict12266/20000, two-P strict4550/10000은 canonical 순위 성공과 별도다. Native target의 prefix-space/token 규약과 canonical observer는 원 source를 따른다.',
      '## 2. 동일10k family별 baseline 비교',
      'W0 pre-edit RS791/10000, PS1997/20000, NS89212/100000은 기존42673 publication에서 재사용한다. 아래14개 기존chain 결과는 봉인 정본 재사용이며 이번새실험이 아니다. BASE_*_NATIVE는 five-layer blue=False, *_BLUE(L4+L8)는 BLUE pair, singleton은 명시 layer다. Legacy ORIGINAL은 BLUE를 뜻하므로 그 단독표기를 쓰지 않는다.']
    for family in ('AlphaEdit','MEMIT'):
        rr=[x for x in fam if x['method']==family]
        lines += [f'### {family} family + CAKE reference',mdtable(['method','RS n/10000','PS n/20000','NS n/100000','allocated GPU-sec'],[[x['arm'],x['RS_numerator'],x['PS_numerator'],x['NS_numerator'],x['scheduler_seconds']] for x in rr])]
    lines += ['CAKE 대비 BASE_ALPHAEDIT_NATIVE 및 BLUE/L4-only 등의 paired loss/gain은 baseline-paired.csv에 local case/prompt/target identity가 일치한 범위만 집계했다. Aggregate와 paired 검증은 구별한다. 환경·hparam 차이가 있어 layer allocation만의 인과효과나 우열을 판정하지 않는다.',
      '## 3. 12개 actual fullseen과 순차 retention',mdtable(['batch/seen','RS','PS','NS'],[[f'{b}/{100*b}',*[next(x['numerator']+'/'+x['denominator'] for x in full if x['batch']==str(b) and x['metric']==m) for m in ('RS','PS','NS')]] for b in (1,5,10,20,30,40,50,60,70,80,90,100)]),
      '매 batch Current100과 all-seen RS는100시점; full R/P/N은12시점이다. 동일 endpoint current를 fullseen에서 재사용했음을 모든12시점 원 rows로 확인했다. 이 rows를 독립 추가평가/표본으로 합산하지 않는다. B10 actual first1000은989/1703/8118이며 W100의first1000과도 다른 상태다.',
      mdtable(['대조/population','metric','before→after','lost/gained','조건부 loss/recovery 분모','desired NLL harm p95/p99'],[[x['comparison']+'/'+x['population'],x['metric'],x['before_num']+'→'+x['after_num'],x['lost']+'/'+x['gained'],x['conditional_loss_den']+'/'+x['conditional_recovery_den'],x['desired_NLL_harm_p95']+'/'+x['desired_NLL_harm_p99']] for x in pairs]),
      'At-write pooled RS9962→W1009840은138 loss/16 recovery를 포함한다. 서로 다른 at-write W를 하나의W0/endpoint로 부르지 않는다. ACTIVE_TARGET/SUPERSEDED는 전체10000 requested event의 exact subject/relation 최신target문자열로 분리하며 same-target 재발행을 active로 둔다. Active9791/superseded209이며 CAKE에는 EP accepted-only ledger가 없다. Superseded 실패를 모두 순수 forgetting으로 치환하지 않는다.',
      'cohort-retention.csv는100개B100 cohort마다 ALL atwrite→W100/conditional분모와NLLtail, rewrite-trajectories.csv는실제저장된매batch RS 실패·회복 event를 담는다. 관측시점 사이의 정확한 실패시간은 추정하지 않는다. B100 cohort는future batch0이다. 초기100/500/1000·중간/후반 차이는그림과CSV의실제값으로제시한다.',
      '## 4. 원본 CAKE가 실제 어떻게 호출됐는가',
      '원본c8243e1/tree4f59249(MIT), 실행wrapper7884aeb/tree4dabdb1. 실제native복사본은 미사용 notebooks.util import1줄 제거와EOF LF 외동일AST이며 compute_z/compute_ks bytes도동일하다. 원본CLI 대신wrapper를쓴것은fixed sample·canonical평가·user no-CP 연결때문이다. 원본의compute_optimal_deltas를사용하지않고apply_Cake_to_model을100회직접호출했다.',
      mdtable(['요구','executed source:line','근거','검증수준'],[[x['requirement'],Path(x['source']).name+':'+x['line'],x['artifact'],x['verification']] for x in conf]),
      '각batch entry에서 최종L8 target100개를한번만만든다. 이후physicalL4..8을순서대로돌며현재state의K와L8readout을다시읽고 R=Z−Y, effective_ratio=w_i/sum(w_i..w_4), residual=effective_ratio·R를원native projected directsolve에넣는다. 앞층write후다음층의residual은현재edited state다. 마지막에모든층현재keys를다시읽어cache_c각1append한다. wrapper추가finalize0이다.',
      mdtable(['physical/P index','causal score','normalized weight FP32','remaining ratio FP32'],[[x['physical_layer']+'/'+x['score_index'],x['score'],x['weight_FP32'],x['remaining_ratio_FP32']] for x in csvread(CAKE/'CAKE-layer-weights.csv')]),
      '위 ratio는원score/temperature.1의CPU산술도출이며raw GPU bit-parity가아니다. 따라서L8 normalized weight약.1514라도마지막remaining ratio는1이다. 10000개target z의layer8/order/hash/norm,500solve shape/dtype,1000key calls(5write+5history/batch),100history pass/500layerupdate가저장telemetry와일치한다. 원Residual/K/solve tensor 자체는미저장이므로이후독립tensor재현은불가하다.',
      '원layers4..8/L2=10/decay=.4/clamp=.5/temperature=.1/원causal score0..4 유지. P physical4..8→asset0..4→local0..4. 기존BASE_ALPHAEDIT는decay=.5/clamp=.75로다르다. 모델revision8afb486c,FP32/eager/seed20260907·고정context·matmulTF32false/cudnnTF32true와canonicalMB16을재사용했다. Writer right-padding/BOSfalse, canonical helper는manual left-padding이다. 원README torch2.6/transformers4.51.3과실제2.9.1+cu128/4.44.2 차이를공개한다. 전체환경/수치동일성을주장하지않는다.',
      '## 5. 실제 write·history·비개입의 근거 경계',
      '100commit은W5개/cache hash와직전entry를연결하고RNG/context도99인접점에서일치한다. 평가before_after_exact와nonselected pointer/version guards,원본context/P최종assert기록을확인했다. 이는저장tensor재해시가아닌metadata/source/telemetry검산이다. layer-actions.csv의increment norm합(path length)은L4..8 각각 '+', '.join(f'{v:.7g}' for v in s['layer_path_lengths'].values())+'이며net endpoint delta는NOT_RECORDED다. 경로길이를누적net변화로부르지않는다.',
      '사용자 “weight 저장하지말라” 지시에따라현재W/M tensor checkpoint0. Hash/RNG/context만으로복원가능checkpoint·exactrestart·GPUcontinuationPASS라고쓰지않는다. 저장하지않은CP의누락을오류로분류하거나재생성하지않았다. 원본raw·실행source·다른CP변경0.',
      '## 6. 비용·조기종료·저장',
      mdtable(['항목','실측 또는 근거값'],[['Slurm allocated GPU-sec/h',f"32194 / {32194/3600:.6f}"],['program seconds',s['program_seconds']],['load/setup seconds',s['load_setup_seconds']],*[[k,v] for k,v in s['totals'].items()],['GPU peak allocated bytes',s['peak_GPU_allocated_bytes']],['현재 output files/bytes',f"{s['raw_files']} / {s['raw_bytes']}"],['native loss evaluations (log-derived)',sa['CAKE_actual_log_losses']],['Adam updates (log-derived)',sa['CAKE_log_Adam']],['early-stop requests / zero-update',f"{sa['CAKE_log_early_stop']} / {sa['CAKE_log_zero_updates']}"]]),
      'Log의각Computing-right-vector→Init/Delta/Target boundary10000개와loss line을원compute_z break/step순서에대조해76952loss/66952Adam을도출했다. 이는native정수counter가아닌로그복원이며377개는25loss/24step,9623개는earlystop,16개는0update다. 0target-Adam도batchwrite0을뜻하지않는다. Request별prompt/log원문은local-only며보고에는집계만둔다.',
      'target/key/solve는edit_seconds안의nestedtimer다. evaluation_seconds와program총시간사이남은overhead를purewriter라고추정하지않는다. Purewriter/readout/별도JSONI-O는NOT_SEPARATED/NOT_RECORDED,checkpoint I-O는user no-tensor다. SchedulerMaxRSS32046396K와GPUpeak는다른지표다. Utilization·통제된speedup은미측정이다. 이CPU리뷰새GPU시간0.',
      '## 7. Coverage와 재현',
      '원NLL/분모/identity/finite/ties/strict·100commit·99metadata links·12fullseen·원source/호출routing·actualcost는확인했다. 저장W/M/전체모델off-on/derivative/GPUcontinuation은미측정. W0재평가·GLUE/MMLU/새downstream·baseline rerun0. CSV raw-inventory/source-inventory/analysis-manifest와rooted-receipt에입출력SHA를결속한다. PNG는CSV기반직접코드생성. README의CPU재현명령에서출력은새namespace를사용하고scheduler조회는기존receipt를재사용한다. 최종효능/우열/원인및후속선택은GH소유다.']
    write(CAKE/'diagnostic-report-ko.md','\n\n'.join(lines))
    r.table(CAKE/'coverage.csv',[dict(requirement=x['requirement'],status=x['verification'],boundary=x['artifact']) for x in conf]+[
       dict(requirement='W/M checkpoint CPU reload',status='NOT_SAVED_USER_DIRECTED',boundary='No tensor replay/backfill'),
       dict(requirement='GPU continuation/derivative',status='NOT_TESTED',boundary='No new GPU in this review')])

def overview(schedule_rows):
    inventory=[]
    for x in schedule_rows:inventory.append(dict(experiment=x['arm'],job=x['job'],status='COMPLETED_0_0_NEW_CPU_REVIEW',scope='W0_B100x100_10k' if x['arm']=='CAKE' else 'W0_B100x10_1k',report=str((CAKE if x['arm']=='CAKE' else CAP).relative_to(WT)/'diagnostic-report-ko.md'),new_review=True))
    historical=[('CAP1 EP-TW-1','47962','ep-tw1-c4-2026-09-15-v1/gate-skip-r1/completed-review-v1/diagnostic-report-ko.md','COMPLETED_REUSED_NUMERICAL_NOT_ESTABLISHED'),
       ('BLUE/native14chains + W0','39307/39283_1..5/40441..46/42657/42658/42673','blue-native-lifelong-comprehensive-review-2026-09-11-v1/diagnostic-report-ko.md','SEALED_COMPLETE_REUSE'),
       ('BLUE/L4/L8 canonical6','39307/39283_1..5','blue-l4-l8-lifelong-b100x100-review-2026-09-09-v3/factual-report-ko.md','SEALED_COMPLETE_REUSE'),
       ('BLUE/JVP/O fivearm1k','38940/38997/38988','blue-alphaedit-fivearm-sequential1000-review-2026-09-07-v1/factual-report-ko.md','SEALED_COMPLETE_REUSE'),
       ('JVP L8 takeover','38433_4/5','alpha-jv-l8-only-sequential1000-review-2026-09-07-v1/factual-report-ko.md','SEALED_COMPLETE_REUSE'),
       ('lowcost static core6','46451','low-cost-write-donor-pilot-2026-09-13-v1/core-completed-review-v1/diagnostic-report-ko.md','SEALED_COMPLETE_REUSE'),
       ('lowcost seq10 sixarm','46475_0..5','low-cost-write-donor-seq10-2026-09-13-v1/completed-review-v1/diagnostic-report-ko.md','SEALED_COMPLETE_REUSE'),
       ('write-refresh Middle','47020_0..3 + reusedN4REFIT4','refit4-write-refresh-seq1000-2026-09-14-v1/completed-review-v1/diagnostic-report-ko.md','SEALED_COMPLETE_REUSE'),
       ('BG1','47592teacher only','bg1-c4-ours-first-2026-09-15-v1/g0-factual-report-ko.md','CALIBRATION_MISSING_SCIENCE_NOT_SUBMITTED; NO_REAUDIT'),
       ('EP original/repair technical','47884/47942','ep-tw1-c4-2026-09-15-v1/fd-repair-r1/diagnostic-report-ko.md','FAILED_PRIOR_RCA_REUSED; NO_REDIAGNOSIS'),
       ('Checkpoint preservation','183CP','checkpoint-migration-server2-2026-09-11-v1/factual-migration-ko.md','STORAGE_TASK_COMPLETE_REUSE_NO_RAW_ACCESS')]
    reuse=[]
    for name,job,path,status in historical:
        p=BASE/path;assert p.is_file();reuse.append(r.ref(p));inventory.append(dict(experiment=name,job=job,status=status,scope='PRIOR_SEALED_SCOPE_NOT_NEW_REAUDIT',report=str(p.relative_to(WT)),new_review=False))
    inventory.append(dict(experiment='ORBODE cumulative',job='37649',status='STOPPED_USER_REVIEW_RAW_DELETED_USER_AUTHORITY_NO_REAUDIT',scope='LINK_ONLY',report='experiment-reports/servers/server4/orbode-cumulative-b100x10-exhaustive-2026-09-07-v1/',new_review=False))
    r.table(OVER/'run-inventory.csv',inventory);r.save(OVER/'evidence-reuse-manifest.json',dict(task=TASK,reused_reports=reuse,history_raw_reaudits=0,ORBODE='STOPPED_LINK_ONLY'))
    write(OVER/'diagnostic-report-ko.md','\n\n'.join(['# Server4 종료 실험 종합 사실 리뷰',f'Instruction `{TASK}`. 최신main7ef4fc05에서독립worktree를만들었으며sharedroot와sweep a7bfd2f의dirty pause2파일을보존했다.',
      'CAKE48101과CAP10/100/NORM_ONLY48148/48149/48150은한정scheduler조회에서모두COMPLETED0:0이며각actualterminal도확인했다. 이번에NOT_COMPLETED또는신규실행실패로남은대상은없다. 과거EP기술실패는그대로실패이며CAP1수치미검증상태도유지한다.',
      mdtable(['대상','job','범위','상태'],[[x['experiment'],x['job'],x['scope'],x['status']] for x in inventory]),
      '[CAKE 10k 상세 보고](../cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v1/diagnostic-report-ko.md)',
      '[EP alpha cap sweep 1k 상세 보고](../ep-tw1-alpha-cap-sweep-2026-09-15-v1/completed-review-v1/diagnostic-report-ko.md)',
      'CAKE W100:9840/10000,17755/20000,62935/100000. CAP1/10/100/NORM_ONLY W10:first1000의독립표는각보고서에있다. 서로다른규모/시점/W50warm을같은분모로합산하지않는다. CAKE B10의동일first1000는별도참고표다.',
      '신규실행allocation CAKE32194초 + cap22933초 =55127GPU초(15.313056GPUh). CAP1재사용7694초/teacher98초/기존실패473+69초는신규지출이아니다. 네job start/endinterval상최대2GPU할당이며utilization은미측정이다. 리뷰새GPU0.',
      '새30개capCP fullSHA/CPU shape/dtype/finite/headerbridge·27links·3000target/30native solve/30final history를검산했다. CAKE는100commit/99metadata links/10000target/500solve/500layerhistory이며W/M tensor미저장useroverride에따라복원가능성을주장하지않는다.',
      '과거정본은위report/hash를재사용했다. 14chain whole10k 비교의Original은native/BLUE를명시적으로구분했다. source-backed설계동작·모든후보·반대문항전이·NLLtail·비용은각family보고와CSV에보존했다. 과학우열/인과기여/승격·후속선택은GH소유다.',
      'CPU reviewer 수리: CAKE weight_state/cache_sha256/정수0 schema, NORM_ONLY strict-loss ID의문자열정렬vs숫자정렬(동일set3개),실행closure의cake_native_lifelong 단축namespace경로를반영했다. 실패한CPU분석namespace는보존했으며runtime·수치threshold·raw수정0. 현재cap lock의구형 numerical_rationale alpha1설명은실제numerical_policy와구분했다.',
      '새Slurm/model/forward/evaluator/FD/ULP/teacher/baseline/rsync/delete0. 사용자pause를그대로보존한완료리뷰recall이며task완료후STOP/automatic_resume=false. 미측정Report/Audit/MMLU/FutureN·GPUreplay는이번에채우지않았다.']))

if __name__=='__main__':stage()
