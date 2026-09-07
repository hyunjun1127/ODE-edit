"""Create-once raw-free Korean review package from sealed CPU analyses."""
import argparse,csv,hashlib,json,math,subprocess
from pathlib import Path
from ..reporting import csvwrite,jwrite,sha
from .performance import MODELS,ARMS,numeric,rows,hash_json

PACKAGE='experiment-reports/servers/server4/alpha-jv-l8-only-sequential1000-review-2026-09-07-v1'
TITLE='AlphaEdit sequential 1,000 — Official/JV/L8-only 양모델 결과 검토'

def num(v):
    if v is None or v=='' or isinstance(v,str) and v.startswith('NOT_'):return 'NR'
    if isinstance(v,bool):return str(v)
    if isinstance(v,int):return str(v)
    if isinstance(v,float):return f'{v:.7g}'
    return str(v).replace('|','/')
def table(rr,columns):
    out=['|'+'|'.join(label for _,label in columns)+'|','|'+'|'.join('---' for _ in columns)+'|']
    out+=['|'+'|'.join(num(r.get(key)) for key,_ in columns)+'|' for r in rr]
    return '\n'.join(out)
def read(p):return json.loads(Path(p).read_text())
def parse(p):return [numeric(r) for r in rows(p)]
def metric_cells(r):
    d=dict(r)
    for k in ('RS','PS','NS'):d[k]=f"{r[k+'_num']}/{r[k+'_den']} ({100*r[k+'_rate']:.2f}%)"
    d['strictPS']=f"{r['PS_strict_num']}/{r['PS_strict_den']}"
    for k in ('rewrite','rephrase'):
        d[k+'_acc']=f"{r[k+'_target_new_all_tokens_correct_count']}/{r[k+'_target_new_row_count']}"
        d[k+'_token_acc']=f"{r[k+'_target_new_correct_token_count']}/{r[k+'_target_new_target_token_denominator']}"
    return d
def safe_copy(src,dst):
    with Path(dst).open('xb') as f:f.write(Path(src).read_bytes())

def build(repo,perf,mech,integrity,plots,output):
    repo,perf,mech,integrity,plots,output=map(Path,(repo,perf,mech,integrity,plots,output))
    ir=read(integrity/'integrity-receipt.json');ib=dict(ir);iroot=ib.pop('root_sha256')
    assert ir['status']=='TERMINAL_FULL_REHASH_PASS' and hash_json(ib)==iroot
    assert (ir['batches'],ir['requests'],ir['checkpoints'],ir['nonfinite_count'],ir['failure_member_count'])==(20,2000,6,0,0)
    for name,key in [('raw-member-inventory.json','raw_inventory_sha256'),('batch-integrity.json','batch_integrity_sha256'),('checkpoint-integrity.json','checkpoint_integrity_sha256')]:
        assert sha(integrity/name)==ir[key]
    tests_path=perf.parent/'validation-v1/tests.json';assert read(tests_path)['status']=='PASS'
    output.mkdir(parents=True,exist_ok=False)
    sources=[];external=[]
    safe_copy(tests_path,output/'focused-tests.json')
    sources.append(dict(group='tests',path=str(tests_path),sha256=sha(tests_path),bytes=tests_path.stat().st_size))
    for name in ['messages/head/2026-09-07-alpha-jv-l8-server4-results-review.md',
      'experiment-reports/global/alpha-jv-sequential-mechanism-audit-2026-09-07-v2/gh-mechanism-review-ko.md',
      'experiment-reports/servers/server1/alpha-jv-sequential1000-layer-review-2026-09-07-v1/factual-report-ko.md']:
        p=repo/name;sources.append(dict(group='full_read_authority',path=name,sha256=sha(p),bytes=p.stat().st_size,lines=len(p.read_text().splitlines())))
    for group,path in [('performance',perf),('mechanism',mech),('integrity',integrity),('plots',plots)]:
        for p in sorted(path.iterdir()):
            if not p.is_file():continue
            row=dict(group=group,path=str(p.absolute()),sha256=sha(p),bytes=p.stat().st_size)
            if p.suffix=='.csv':
                with p.open() as f:row['rows']=sum(1 for _ in csv.DictReader(f))
            sources.append(row)
            if p.name in ('l8_seen_scalar_rows.csv','l8_rewrite_retention_records.csv'):
                external.append(dict(row,classification='LARGE_RAW_FREE_SCALAR_TABLE_LOCAL_ONLY'));continue
            if p.suffix not in ('.csv','.json','.png','.md'):continue
            safe_copy(p,output/p.name)
    ext=Path('/data/janghj/ODE-edit/local/state/alpha-jv-migration-server4-20260907/tech-r1/external-main')
    for name in ('neighborhood_transition_summary.csv','paired_seen_endpoint_metrics.csv','run_registry.csv','failure_registry.csv'):
        p=ext/name;safe_copy(p,output/('reference_'+name));sources.append(dict(group='reference',path=str(p),sha256=sha(p),bytes=p.stat().st_size))
    scheduler=Path('/data/janghj/ODE-edit/local/alpha-jv-l8-only-results-review/scheduler-once.json')
    safe_copy(scheduler,output/scheduler.name);sources.append(dict(group='scheduler',path=str(scheduler),sha256=sha(scheduler),bytes=scheduler.stat().st_size))
    jwrite(output/'external-scalar-table-identities.json',external)
    final=parse(perf/'final_metrics.csv');main=[r for r in final if r['arm'] in ARMS]
    current=parse(perf/'current_batch_metrics.csv');seen=parse(perf/'seen_prefix_metrics.csv')
    cohorts=parse(perf/'retention_cohort_metrics.csv');parts=parse(perf/'endpoint_partitions.csv')
    delta=parse(perf/'final_deltas.csv');age=parse(perf/'l8_age_strata.csv')
    batch=parse(mech/'batch_mechanism.csv');node=parse(mech/'node_mechanism.csv');cost=parse(mech/'cost_by_arm.csv')
    b10=[r for r in batch if r['batch']==10]
    report=[]
    def section(title,text):report.extend(['',f'## {title}','',text])
    report+=['# '+TITLE,'',
      'Canonical review v1 / 2026-09-07. **B1→B10 cumulative W/M sequential, batch당100, 총1,000 requests/arm.** 아래 대표 성능은 각 arm의 최종 W10 하나에서 전체1,000개를 재평가한 값이다. Current B100 또는 online-at-write 합계가 아니다. 기존 O/JV4 chain과 Server4 fresh L8 replacement2 chain만 포함한다.','',
      table([metric_cells(r) for r in main],[('alias','모델'),('arm','arm'),('RS','RS'),('PS','PS'),('strictPS','strict PS'),('NS','NS')]),'',
      '각 셀에서 RS 분모1,000 rewrite prompts, PS2,000 rephrase prompts, NS10,000 neighborhood prompts; strict PS 분모1,000 requests. Missing arm을0으로 채우지 않았다. 상세 표: [main_six_arm_table.csv](main_six_arm_table.csv).']
    section('1. Executive FACT / 해석 경계',
      'Llama: L8-only는 JV 대비 RS −0.50pp, PS +0.60pp, NS −0.45pp이다. Official 대비 세 primary 지표 모두 낮다. L8-only의 최종 rewrite 실패82건 중78건은 처음부터 실패했고, 성공 후 소실은4/922이다. JV는 각각75건과2/925이다. 두 chain 모두 B10에서 신규 acquisition 실패가 주된 문제이며, 모든 과거 edit의 forgetting으로 부르면 안 된다.\n\n'
      'Qwen: L8-only는 Official 대비 RS +0.30pp, PS +2.35pp, NS +3.57pp이다. JV 대비 RS −0.20pp, PS +2.00pp, NS −0.42pp이다. 따라서 이 stream에서 JV locality 이득의 큰 부분과 비슷한 현상이 early-layer write가 정확히0인 support-restricted controller에서도 관측된다. 이는 다층 mixing의 필수성을 지지하지 않지만, 동일 prompt가 보존되었다는 paired causal 증거 또는 endpoint equivalence 검정은 아니다.\n\n'
      '핵심 confound: O/JV는 Server2 RTX A6000, L8-only는 Server4 RTX PRO 6000 Blackwell에서 실행했다. Model revision·pinned assets·scientific kernel은 결속하지만 backend/library 전체 bitwise parity는 보증하지 않는다. 이후 W/M/z도 arm별로 달라진다. Single stream·모델별 descriptive evidence이며 promotion=false, 신규 sweep/lifelong release 없음.')
    section('2. Metric reading guide',
      '|기호/지표|정의·단위|해석|\n|---|---|---|\n'
      '|RS / PS|rewrite / 각 rephrase prompt에서 length-normalized target-new NLL < target-true NLL|높을수록 좋음. Tie 실패; PS는 request 평균 NLL 비교가 아님|\n'
      '|NS|각 neighborhood prompt에서 true NLL < new NLL|높을수록 좋음. Teacher-forced true-token accuracy와 다름|\n'
      '|strict PS / strict NS|request의2/10 prompts 모두 canonical preference 성공|request 단위, 각각 분모1,000|\n'
      '|rewrite/rephrase acc|teacher-forcing에서 해당 target의 모든 token top1이 맞는 prompt 수|free-generation accuracy 아님; token accuracy는 별도 correct tokens/target tokens|\n'
      '|NLL|−mean target-token log probability, nat/token|각 target likelihood는 낮을수록 높음. New와true를 분리|\n'
      '|preference margin|RS/PS=true NLL−new NLL, NS=new NLL−true NLL|양수 성공. 높을수록 preference가 강함|\n'
      '|mean / median / p90 / max|동일 category의 prompt-level 분포|request-strict와 혼합 금지. NR는 미기록이며0이 아님|\n'
      '|N0, e, V|entry residual scale s_i 고정; source weighted residual e; V=½‖e‖²|양의 tiny scale을 사후 floor/삭제하지 않음|\n'
      '|g,H,G,c|weighted response의 residual inner product, response Gram, native Gram, nonnegative coefficient|signed g_l c_l은 예측 진행량; 실제 ΔV와 별도|\n'
      '|F, ΔW|node velocity와 실제 FP32 materialized weight difference|Layer-wise Update Magnitude=‖ΔW_l‖F. share의 제곱norm 기준 여부를 명시|\n'
      '|Qraw, Q, work|native M_entry+L2I quadratic; Q=Qraw/qref; work=ΣhQ(F)|Frobenius magnitude/net action과 다른 단위. Net은 batch entry→endpoint|\n'
      '|B, b|B=V+λ∑hQ, reserve b=V0−B|CSV barrier_increment=ΔB; 양수는 reserve 감소. Continuous KKT≠finite-step guarantee|\n'
      '|finite defect|ΔV+h(‖Ψc‖²+λcᵀGc)−½h²‖Ψc‖²|실제 ΔB와 다름. 부호를 성공/실패로 새 gate화하지 않음|\n'
      '|model error / materialization mismatch|예측 response 대비 실제 activation 차이 / virtual→materialized normalized discrepancy|functional NLL·NS와 동일량이 아님|')
    section('3. Provenance, terminal 및 무결성',
      '한 번의 bounded scheduler 확인: 38433_4(child38434) COMPLETED0, 02:24:36; 38433_5(child38433) COMPLETED0, 02:08:39. task-owned active0. 이후 scheduler polling0. Submitted/executed source44602a1a80554c67da0ef9646b43d843104785f2/tree cdb089a139f180c822d1e6bf44fe3d27b858c1ce; source parent77358b1546d1baf83b3e251afcce663b08d7bfd7/tree b64e84f2af7c708405dd6a8a9f018d3ece985c58.\n\n'
      'O/JV는 job37980의 sealed publication0d0a0131e4a6a2a645dfa6530377d420a084d136를 재사용했다. Original report SHA b9b7fd9b7f37f782ee5fe81608d1b80e942e701ccf01b9896d4d79401d2617ad. Server2 raw/checkpoint 재해시는 미수행이며 Git publication48members 검증과 구분한다.\n\n'
      '공통 sample root40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd; 각 arm10×B100, final1,000 unique requests. Pinned EasyEdit14cea8245f06715684592ab55184939b99d70784/tree9c52aadbc0883da422badf0a730fff21aaa3a8a7. Raw prompt/logit/generation은 이 package에 포함하지 않는다.\n\n'
      'L8 raw 두 terminal/20 batch, W/M inter-batch18 links, target request2,000/recompute0, history append20, actual W/M checkpoints6개 및 final W0/M0 restore를 별도 integrity receipt로 결속했다. Target once는 batch target_count와 pinned source 경로의 결합이며 별도 compute_z invocation counter는 미기록이다. FULL-FP32 model/storage, controller/native scalar FP64는 기존 source 계약이다. Autocast/BF16/FP16/quantization0; runtime right-padding은 accepted NativeDictionary path 그대로이며 과거 다른 실험의 left-padding 규칙으로 바꾸지 않았다. 기록 없는 tokenizer token IDs/version을 추정하지 않는다.\n\n'
      '전체 raw member SHA/bytes와 source/runtime/state 검증 범위는 integrity JSON/CSV에 있다. 현재/seen-prefix 평가는 committed selected W에 결속되며 final-W10과 B10 current를 구분한다. M0 cold-init은 각 arm 최초1회만; 이후 M/W를 누적한다. Source-inspected assertions와 이번 CPU tensor rehash를 같은 보증으로 합치지 않는다.')
    section('4. Final W10 상세 NLL / margin / secondary accuracy',
      '모든 NLL/paired-margin 분포는 prompt-level이다. O/JV publication은 marginal NLL 분포를 제공하지만 prompt-pair 원본이 없으므로 margin median/p90/max는 NR로 표시한다. Mean margin만 paired means의 차로 정확히 계산 가능하다. 서로 다른 marginal quantile을 빼서 margin quantile을 만들지 않았다. W0는 reference이며 새 run이 아니다.')
    for prefix in ('rewrite','rephrase','locality'):
        rr=[]
        for r in final:
            for target in ('new','true'):
                rr.append(dict(model=r['alias'],arm=r['arm'],target=target,n=r[prefix+'_target_'+target+'_row_count'],
                    **{s:r[prefix+'_target_'+target+'_nll_'+s] for s in ('mean','median','p90','max')}))
        report+=['',f'### {prefix} NLL (nat/token; prompt 단위)','',table(rr,[('model','모델'),('arm','arm'),('target','target'),('n','n'),('mean','mean'),('median','median'),('p90','p90'),('max','max')]),'',
            f'### {prefix} canonical preference margin (nat/token)','',table(final,[('alias','모델'),('arm','arm')]+[(prefix+'_margin_'+s,s) for s in ('n','mean','median','p90','max')])]
    report+=['','### Teacher-forced secondary accuracy','',table([metric_cells(r) for r in main],[('alias','모델'),('arm','arm'),('rewrite_acc','rewrite all-token prompt acc'),('rephrase_acc','rephrase all-token prompt acc'),('rewrite_token_acc','rewrite token acc'),('rephrase_token_acc','rephrase token acc')]),'',
      'Qwen L8-only rephrase-new mean은 JV보다0.29198 nat 낮고 PS는+2.00pp이다. Llama L8-only rephrase-new mean은0.03276 낮지만 p90은 약0.03775 높아 tail과중심이 같은 방향은 아니다. 분포 전체와 binary preference를 하나의 성능 개선으로 뭉치지 않는다.']
    section('5. 누적 성능과 모든 current B100: 서로 다른 W/분모',
      '전체 seen-prefix RS/PS/NS는 W1(100), W5(500), W10(1,000)에서만 기록했다. 중간 B2–B4/B6–B9 PS/NS는 NOT_RECORDED이며 보간/재평가0. Seen rewrite retention은 모든 B에 기록했다. 아래 current 표는 그 B의 새로운100개만 평가한 acquisition이다.')
    report+=['','### All-seen cumulative checkpoints (18행)','',table([metric_cells(r) for r in seen],[('alias','모델'),('arm','arm'),('batch','W checkpoint'),('request_count','seen requests'),('RS','RS'),('PS','PS'),('strictPS','strict PS'),('NS','NS')])]
    for model in MODELS:
        report+=['',f'### {model}: current B100 전체30행','',table([metric_cells(r) for r in current if r['alias']==model],
            [('arm','arm'),('batch','B'),('RS','RS'),('PS','PS'),('NS','NS'),('rewrite_target_new_nll_mean','RW-new mean'),('rephrase_target_new_nll_mean','RP-new mean')])]
    section('6. Acquisition / forgetting / recovery / overwrite',
      'Final failure를 처음부터 실패한 request와 성공 후 소실한 request로 분리했다. 조건부 forgetting 분모는 at-write 성공 request이고, 전체 canonical 분모1,000도 유지한다. 동일 subject/relation의 later target change1건은 declared overwrite 후보일 뿐 원인 확정이 아니며 primary에서 제외하지 않았다.')
    report+=['',table(parts,[('alias','모델'),('arm','arm'),('at_write_success','at-write 성공 /1000'),('initially_failed','처음 실패 /1000'),('at_write_success_to_final_failure','성공→실패'),('conditional_failure_denominator','조건부 n'),('initially_failed_to_final_recovery','처음실패→회복'),('final_RS','final RS'),('nonoverwrite_failure','nonoverwrite 소실'),('overwrite_candidates','overwrite 후보')]),'',
      'Llama JV B10 current RS25/100, L8-only22/100이며, 앞9 batches는 각각900/900 at-write 성공이었다. Final에서는898/900과896/900이다. 따라서 near-stall과 신규 target realization 문제는 L8-only에서도 관측되며, 작은 early-layer mixture를 없앴다고 사라지지 않았다. Qwen L8-only는999/1000 at-write 성공, 이후4/999 소실; JV는1000/1000과3/1000이다.',
      '', '### Final cohort retention: cohort마다100 requests','',table([r for r in cohorts if r['batch']==10],
       [('alias','모델'),('arm','arm'),('cohort','편집 B'),('current_success','final RS /100'),('at_write_success','at-write /100'),('at_write_success_now_failure','소실'),('initially_failed','최초실패'),('overwrite_candidate_count','overwrite 후보')]),'',
      '모든330 cohort×evaluation-B 행은 retention_cohort_metrics.csv; heatmap은 각 request가 편집된 B와 재평가 W를 구분한다. L8 request-level11,000 scalar records는 로컬-only 외부 입력으로 hash/경로를 봉인했다. B별 평균을 독립 반복실험처럼 취급하지 않았다.',
      '', '### L8 cumulative age strata (18행)','',
      'Age bins는 매 checkpoint의 seen-prefix 내 early20%, middle60%, recent20%의 상대 위치다. 고정 cohort의 인과적 age effect가 아니며 case composition이 달라진다.', '',
      table(age,[('alias','모델'),('batch','W'),('stratum','stratum'),('request_denominator','requests'),('RS_num','RS num'),('RS_den','RS den'),('PS_num','PS num'),('PS_den','PS den'),('NS_num','NS num'),('NS_den','NS den')])]
    section('7. Neighborhood 보존 및 paired availability',
      'L8-only−Official NS는 Llama−370/10,000, Qwen+357/10,000이다. JV 대비는 각각−45,−42이다. W0 대비 L8-only NS는 Llama−16.06pp, Qwen−11.28pp이므로 원래 지식을 완전히 보존한 것은 아니다.\n\n'
      'Qwen L8-only의 neighborhood true NLL mean은 Official보다0.59764 nat 낮고 JV보다0.09382 낮다. 그럼에도 JV보다 NS가0.42pp 낮으므로 NLL 평균과 paired preference는 동일한 정보가 아니다. Llama true NLL은 Official보다0.44889 높다. Full shifts는 final_deltas.csv에 new/true·mean/median/p90/max 모두 기록했다.\n\n'
      '**L8↔W0/O/JV의 동일 prompt success→loss/recovery는 NOT_RECORDED_PUBLISHED_PROMPT_PAIRS.** Server2가 공개한 paired summary는 O↔JV 및 W0→O/JV용이며 L8 bit와 join할 prompt 원본이 없다. 기존 O/JV transition summaries는 reference_*.csv로 보존했지만 L8 loss/recovery를 aggregate gain에서 역산하지 않았다. Server2 raw 복구를 기다리거나 새 evaluator를 실행하지 않았다.')
    report+=['',table(delta,[('alias','모델'),('reference','L8−reference'),('RS_delta_pp','RS Δpp'),('PS_delta_pp','PS Δpp'),('NS_delta_pp','NS Δpp'),('locality_true_nll_delta_mean','NS true NLL Δmean'),('locality_new_nll_delta_mean','NS new NLL Δmean')])]
    section('8. 물리적 layer write / native work / signed progress',
      'L8_ONLY_NATIVE는 Official hparams.layers=[8]이 아니다. 원래 L4–8 inventory/P indexing과 모든 entry direction의 qref를 유지하고, 매 node current-state L8 JVP와 nonnegative1변수 solve만 실제 support로 쓴다. Full5-layer post-endpoint history finalization은 유지된다. L4–7 actual materialized write는 L8의80/80nodes에서 정확히0이다.\n\n'
      'Magnitude=‖ΔW_l‖F; squared magnitude=‖ΔW_l‖F². 아래 L8 share는 **제곱norm 비중**이다. Native raw work는 M_entry+L2 metric이며 h를1회 곱한 velocity work, net native action은 batch entry→materialized endpoint이다. Signed h∑g_lc_l은 예측 진행량이다. W0→W10 dense net은 공개 비교표에 미기록이며 batch-net 합으로 대체하지 않는다.')
    for model in MODELS:
        report+=['',f'### {model}: 모든30 batch 물리량','',table([r for r in batch if r['alias']==model],
           [('arm','arm'),('batch','B'),('total_batch_net_norm','‖ΔW‖F'),('L8_energy_share','L8 squared share'),('L4_7_batch_net_energy','L4–7 squared'),('signed_predicted_progress','signed hΣgc'),('raw_native_work','raw work'),('normalized_native_work','work/qref'),('materialized_net_native_raw','net native raw'),('history_work','history work'),('L2_work','L2 work'),('final_V_ratio','V/V0')])]
    section('9. Barrier/N0/finite-step와 Llama B10',
      '동일 recorded matrix로 재계산한 continuous KKT/dissipation, finite-step defect, 실제 ΔB는 서로 다른 주장이다. ΔB>0(=reserve b 감소)은 Llama JV2/40, L8-only1/40이고 Qwen 두 arm은0/40이다. 새 결과 거절 threshold는 추가하지 않았으며 모든 finite endpoint를 유지했다. Full node160행과 node-layer800행을 CSV로 보존한다.\n\n'
      'Llama B10 JV와 L8-only 모두 case4228에서 tiny positive N0를 보이지만 W/M/z가 다른 chain이다. L8-only의 100% L8 share는 절대 write가 거의0일 때의 비중이므로 유용한 write 집중의 증거가 아니다. 같은-state causal N0 ablation이나 floor repair는 하지 않았다. λ/sweep/normalization 후속 선택은 이 보고서 권한 밖이다.')
    report+=['',table(b10,[('alias','모델'),('arm','arm'),('minimum_active_scale','N0 min'),('minimum_scale_case_ids','case IDs'),('node0_response_Gram_diagonal_max','node0 Hdiag max'),('total_batch_net_norm','‖ΔW‖F'),('final_V_ratio','V/V0'),('materialization_discrepancy_normalized','materialization mismatch'),('RS_num','current RS /100')]),'',
      'Llama JV/L8 B10 norm=5.21220e−5/7.90996e−6, V/V0=.99994719/.99999903, N0 min=3.57471e−5/1.65121e−5다. H88/최대diag는4.40228e6/8.41862e6. 이것은 공유되는 qualitative near-stall이며 정확한 동일 state failure 재현은 아니다. Qwen B10 L8는 V/V0=.025902, current RS99/100이고 정상적인 비영 write를 유지했다.']
    section('10. 시간·연산량: JVP만 5→1',
      'L8-only는 main JVP20→4/B100(전체200→40)지만, native dictionary는 entry5 + node마다5×4 =25 builds/B100, 전체250을 유지한다. Key/history/target 최적화/evaluator는 사라지지 않는다. 따라서 whole-runtime1/5라고 주장하지 않는다. Terminal forward/backward count는 호출 수이며 FLOPs가 아니다.\n\n'
      'Server4 L8 process_seconds는 Llama8671.4393, Qwen7712.1008이다. JV 대비 비율약.52968/.59074는 하드웨어/동시성/원래 target optimizer 차이가 섞인 wall ratio일 뿐 통제된 speedup은 아니다. Model-load·target·writer(endpoint 포함/제외)·seen eval·history/snapshot bracket·checkpoint·commit을 분리한다. History bracket에는 snapshot/restore가 포함되어 순수 append 시간이라고 부르지 않는다.')
    report+=['',table(cost,[('alias','모델'),('arm','arm'),('process_seconds','process s'),('forward','forward'),('backward','backward'),('main_JVP','main JVP'),('native_key_captures','keys'),('solve_instrumented','solve calls'),('target_wall','target s'),('write_including_endpoint_wall','writer+eval s'),('endpoint_seconds','endpoint s'),('seen_wall','seen s'),('history_seconds','history bracket s'),('checkpoint_wall','checkpoint s'),('commit_wall','commit s'),('peak_gpu_bytes','peak GPU bytes')])]
    section('11. Figures / 재현',
      '각 figure는 저장된 CSV만 읽는 headless Python plotting code로 생성한다. 모델순서 Llama→Qwen, arm순서 O→JV→L8를 고정했다. 미기록 cumulative PS/NS를 보간하지 않는다. Weight title은 Layer-wise Update Magnitude이며 equal-share/ideal line 또는 bars: footer가 없다. Code/input/output SHA와 재생성 명령은 plot receipt에 결속한다.')
    for p in sorted(output.glob('*.png')):report+=['',f'![{p.stem}]({p.name})','',f'`{p.name}`: current는 batch당100/200/1000, cumulative는 해당 W의 seen100/500/1000과×2/×10 prompts, retention은 cohort당100이다. 물리량은 batch 또는 node/layer 단위이며 prompt 수로 복제하지 않는다. 구체 축/단위는 figure와 plot receipt 참조.']
    section('12. Exclusions / missing / FACT–INFERENCE–DECISION',
      'Old Server2 38306_4/5는 takeover를 위한 user-directed external termination이다. GH/SH2 전달상 각각 B1=100 후 B2_TARGET에서 취소, W1/M1만 있고 finalW10은 없다. 이 prefix의 consumption0, final scientific denominator0이며 새38433에 합치지 않았다. 새38433의 completed2 chains는 partial이 아니다. 기존O/JV37980 네 chain은 재실행0.\n\n'
      '미기록: Server2 remote raw/checkpoint 재해시, L8-vs-reference prompt loss/recovery, O/JV margin quantiles, runtime library/driver version 일부, 별도 per-request H/JVP 원인분해, W0→W10 dense net displacement, current/seen을 넘어선 lifelong10k. NR를0 또는 PASS로 채우지 않았다. Full library version inventory가 없으므로 hardware간 exact trajectory parity도 주장하지 않는다.\n\n'
      '**FACT:** L8-only에서도 Qwen의 Official 대비 locality 이득과 Llama의 B10 near-stall이 관측된다. Full JV는 L8보다 양모델 NS가각각45/42 prompts 높지만 PS는 낮고 RS 차이는5/2 requests다. **INFERENCE:** early-layer avoidance·반복 response feedback·amplitude/native geometry가 다층 mixing 없이도 이 결과와 양립한다. 작은 full-JV mixture의 추가 이득은 일부 지표에서 가능하지만 같은-state/같은backend 비교와 paired prompt 원본이 없어 원인을 확정할 수 없다. 작은 share 자체는 성공도 실패도 아니다. **DECISION:** REVIEW_READY_FOR_GH, promotion=false, 새 하이퍼파라미터 선택/production patch/lifelong release 없음. 이 task는 analysis-only이며 main 통합은 GH 검토 후다.')
    section('13. Artifact inventory / source identity',
      '분석 코드: `project/run_scripts/alpha_native_response_ode_v31_sequential/l8_takeover_analysis/`. Executed source와 analysis source를 분리하며 report commit은 Git HEAD로 별도 전달한다. 모든 small package members 및 외부 local-only scalar tables는 analysis-manifest/rooted receipt로 결속한다. 원본 source/result/log/checkpoint는 변경하지 않았다. Models/GPU/evaluator/Slurm new actions0; main_push0.')
    members=[]
    for p in sorted(output.iterdir()):
        if p.is_file():
            r=dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size)
            if p.suffix=='.csv':r['rows']=sum(1 for _ in csv.DictReader(p.open()))
            members.append(r)
    report+=['',table(members,[('path','파일'),('rows','rows'),('bytes','bytes'),('sha256','SHA256')])]
    (output/'factual-report-ko.md').write_text('\n'.join(report)+'\n')
    source_head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    source_tree=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD^{tree}'],text=True).strip()
    source_code=[dict(path=str(p.relative_to(repo)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted((repo/'project/run_scripts/alpha_native_response_ode_v31_sequential/l8_takeover_analysis').glob('*.py'))]
    manifest=dict(schema='alpha-jv-l8-review-package/v1',instruction_id='ODEEDIT-S06-ALPHA-JV-L8-SERVER4-RESULTS-REVIEW-V1',
        analysis_source_head=source_head,analysis_source_tree=source_tree,source_code=source_code,
        executed_source_head='44602a1a80554c67da0ef9646b43d843104785f2',reference_publication='0d0a0131e4a6a2a645dfa6530377d420a084d136',
        inputs=sources,external_tables=external,members=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size,
          **({'rows':sum(1 for _ in csv.DictReader(p.open()))} if p.suffix=='.csv' else {})) for p in sorted(output.iterdir()) if p.is_file()],
        expected_denominators=dict(completed_arms=6,requests_per_arm=1000,RS_per_arm=1000,PS_per_arm=2000,NS_per_arm=10000),
        main_push=0,new_model_GPU_evaluator_Slurm_actions=0,scientific_promotion=False)
    manifest['member_root']=hash_json(manifest['members']);jwrite(output/'analysis-manifest.json',manifest)
    receipt=dict(status='REVIEW_READY',member_root=manifest['member_root'],manifest_sha256=sha(output/'analysis-manifest.json'),
        report_sha256=sha(output/'factual-report-ko.md'),root_sha256=hash_json(manifest),task_owned_active_jobs=0,main_push=0,scientific_promotion=False)
    jwrite(output/'rooted-receipt.json',receipt);verify(output);return receipt

def verify(output):
    output=Path(output);m=read(output/'analysis-manifest.json');r=read(output/'rooted-receipt.json')
    assert sha(output/'analysis-manifest.json')==r['manifest_sha256'] and hash_json(m)==r['root_sha256']
    assert hash_json(m['members'])==m['member_root']==r['member_root']
    for x in m['members']:
        p=output/x['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],p
        if 'rows' in x:assert sum(1 for _ in csv.DictReader(p.open()))==x['rows']
    assert sha(output/'factual-report-ko.md')==r['report_sha256']
    return True

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    for arg in ('performance','mechanism','integrity','plots','output'):p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.repo,a.performance,a.mechanism,a.integrity,a.plots,a.output)))
