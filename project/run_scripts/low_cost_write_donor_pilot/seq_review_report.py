"""Generate the factual Korean report from audited raw-free tables."""
import argparse,csv,json
from pathlib import Path
ARMS=['N4','S875','S75','FULL8','RES8','REFIT4']
def rows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def fmt(v):
 if isinstance(v,float):return f'{v:.8g}'
 return str(v).replace('|','/').replace('\n',' ')
def table(rr,cols):
 return '\n'.join(['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']+['| '+' | '.join(fmt(r.get(k,'NOT_RECORDED')) for k in cols)+' |' for r in rr])
def run(root):
 p=Path(root);load=lambda n:rows(p/(n+'.csv'));s=[]
 def add(t):s.append(t)
 def sec(t):add('\n## '+t+'\n')
 first=load('first-final-table');bm=load('batchmetrics');fm=load('finalpopulationmetrics');gen=load('general-panels');paired=load('paired-transitions');nl=load('nll-distributions');cost=load('compute-summary')
 add('# Low-cost six-arm B100×10 순차 편집 완료 사실 보고\n')
 add('Instruction: ODEEDIT-S06-LOWCOST-SEQ10-COMPLETED-DETAILED-REPORT-SH4-V1. 실행46475_[0–5], B51–B60. 공통 L4 W50/M50에서 각자 새 1,000요청을 처리한 6개 독립 경로다. 기존 5,000+신규 1,000의 **actual W60 전체 seen6000**가 첫 표의 평가 대상이다. 새 6,000 edits 또는 full10k 실험이 아니다. 후보 선택·인과 해석·claim 판정은 GH 소유이며 scientific_promotion=false. 원본 core/report/raw는 변경하지 않았다. 아래 PASS는 적시된 검산만 뜻한다.')
 display=[]
 for a in ARMS:
  r=next(x for x in first if x['arm']==a);q={'arm':a}
  for m in ['RS','PS','NS']:q[m]=f"{r[m+'_n']}/{r[m+'_d']} ({100*float(r[m+'_rate']):.6f}%)";q[m+' Δpp']=float(r[m+'_delta_pp_vs_N4'])
  display.append(q)
 add(table(display,['arm','RS','PS','NS','RS Δpp','PS Δpp','NS Δpp']))
 add('분모·부등식·prompt/target/order/hash는 원시 NLL에서 독립 재산출했다. 위의 수치 차이는 같은 평가문항을 서로 다른 최종 모델에서 비교한 산술 차이다. Fullseen에서 잘라낸 old/suffix/current를 별도 독립 시행으로 합산하지 않는다. 원자료 identity와 검산 수준은 metric-reduction-receipt.json 및 state-verification.json 참조.')
 sec('1. 지표 사전과 모집단 경계')
 glossary=[
 {'지표':'RS','정의':'rewrite prompt에서 평균 target-new NLL < target-true NLL','분모/단위':'prompt 1/request; W60 6000','방향':'높음'},
 {'지표':'PS','정의':'각 rephrase prompt에서 new NLL < true NLL; 두 prompt 모두 성공은 별도 request-strict','분모/단위':'prompt 2/request; W60 12000','방향':'높음'},
 {'지표':'NS','정의':'각 neighborhood prompt에서 true NLL < new NLL; token preservation이 아님','분모/단위':'prompt 10/request; W60 60000','방향':'높음'},
 {'지표':'NLL','정의':'−Σ log p(target token / prefix와 이전 정답 token) / target token수','분모/단위':'nats/target-token; prompt 평균 또는 request-cluster 평균을 구분','방향':'원하는 target NLL 낮음; 경쟁 target 단독 방향은 해석 주의'},
 {'지표':'desired margin','정의':'R/P=true NLL−new NLL, N=new NLL−true NLL','분모/단위':'nats/target-token 차이; 양수 성공, 0 tie 실패','방향':'높음'},
 {'지표':'strict/token secondary','정의':'strict는 모든 target token top1 정답 prompt; token은 정답 token 합/전체 target token 합','분모/단위':'strict prompt, token 실제 token수; canonical 선호와 별개','방향':'높음'},
 {'지표':'mean/median/p90/p95/p99/max','정의':'평균/50/90/95/99백분위/최대, 선형 quantile; IQR=q75−q25','분모/단위':'PROMPT 또는 REQUEST_CLUSTER_MEAN 명시; 분위수는 CI 아님','방향':'NLL tail은 낮음, margin은 높음'},
 {'지표':'loss/gain','정의':'동일 identity에서 성공→실패 / 실패→성공; net=gained−lost','분모/단위':'ALL와 before-success/before-failure 조건부 분모 구분','방향':'loss 낮음/gain 높음'},
 {'지표':'Wiki128','정의':'고정 sequence별 valid next-token NLL의 단순평균','분모/단위':'128 sequence, 24999 predicted tokens; token pooled 평균 아님','방향':'낮음'},
 {'지표':'MMLUdev32','정의':'고정32문항 BLUE alternative 확률의 유일 최대 선택; tie/underflow invalid','분모/단위':'correct/32와 invalid 정수; generation parser/F1 아님','방향':'correct 높음/invalid 낮음'},
 {'지표':'Layer-wise Update Magnitude','정의':'실제 저장 FP32 weight 차이의 Frobenius norm; CPU FP64로 norm 집계','분모/단위':'parameter 좌표 norm; layer share=norm/선택 layer norm 합','방향':'기술량; 높음/낮음 자체 우월성 없음'},
 {'지표':'path vs net','정의':'path=Σ ∥Wb−Wb−1∥F; net=∥W60−W50∥F','분모/단위':'실제 batch increment; subfit path와 같지 않음','방향':'방향 없음'},
 {'지표':'M4/M8','정의':'native key Gram history; 최종 endpoint keys로 선택 layer당 batch append1','분모/단위':'행렬 norm/bytes/hash, residual realization ratio 아님','방향':'방향 없음'}]
 add(table(glossary,['지표','정의','분모/단위','방향']))
 add('Current B51…B60는 매번 다른 100개를 쓰므로 곡선 및 online 합계는 누적 retention이 아니다. Historical128은 고정 old 표본일 뿐 old5000 전체가 아니다. B55 suffix500와 B60 suffix1000/fullseen6000만 해당 actual W에서 재평가한 누적 모집단이다. 미저장 중간 fullseen을 보간하지 않았다. B60 Current/suffix/old/Historical은 fullseen의 동일 NLL rows 재사용이다.')
 sec('2. 실행·입력·설정 provenance')
 add('실행 source `5e96dcb3745977b1f273e3f5afbee61167248d49`, tree `6e9f8bdb432392fd5f0d7d0937ff7058666944c0`; control tip93c3e4f와 구분. Archive `3dadca9463ab131846bff5227fe1bbbb540dd5b8f238bfad53b15b121ed0eb42`; execution.lock `695a2d985d5abb8fae1cc6f0b1933022895a47a71142aa032a9da3b6be1bbb28`. 분석 시작 main83ca22bd/tree38a6318d. analysis source identity는 최종 manifest에 별도 기록한다.')
 add('데이터 `/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json`, SHA `3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1`; ordered root `5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729`. B51=[5000,5100), … B60=[5900,6000). 새 unique1000, arm-request6000, 총60batch; order/resample/replacement 변경0.')
 add('Llama-3-8B-Instruct revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, 모든 모델 parameter FP32/eager. Torch2.9.1+cu128/Transformers4.44.2, writer tokenizer add_bos=False·pad=EOS·right; evaluator 원 tokenizer BOS 정책·수동 left padding·MB16을 유지. BLUE311b076a, singleton blue=True/L2=1, physical L4/P asset0→local0 및 L8/asset4→local0. base five-layer blue=False/L2=10과 다른 방법이다. 원 optimizer 최대25 loss 평가/24 Adam update, source-exact clamp/context를 유지했다. Source/member·실제 config 및 공통 prepared 참조는 execution-member-identities.csv에 모두 결속한다.')
 add('공통 prepared SHA `4ea9e5a7733725ab513571845691a28e4ed321884d749156e256fb992a55f933`는 W4/W8/M4/M8/context/RNG를 포함한다. M8은 공통 We에서 과거5000 event를 B100 FP32 순서로 한 번 재인코딩한 기존 준비물을 재사용했다. 이번6chain M8 전수 재구성0. FULL8/RES8는 이후 자기 endpoint M8만 append; scalar/refit의 미사용 M8/W8은 고정한다. Historical BLUE 원 chain 재현으로 부르지 않는다.')
 policies=[{'arm':'N4','alpha':1,'second':'없음','append':'M4=1'}, {'arm':'S875','alpha':.875,'second':'없음','append':'M4=1'}, {'arm':'S75','alpha':.75,'second':'없음','append':'M4=1'}, {'arm':'FULL8','alpha':1,'second':'현재 state L8 fresh z/K/solve','append':'M4/M8 각1'}, {'arm':'RES8','alpha':.75,'second':'현재 partial L8 fresh z/K/solve','append':'M4/M8 각1'}, {'arm':'REFIT4','alpha':.75,'second':'현재 partial L4 fresh z/K/solve','append':'M4=1, 두 번 아님'}]
 add(table(policies,['arm','alpha','second','append']))
 add('매 arm·batch에서 자기 entry로 fresh native D4=WN4−Wentry를 얻으며 이전/다른 arm D4를 재사용하지 않는다. 첫 fit 뒤 임시 W4를 entry로 돌리고 alpha1은 native snapshot exact copy, .875/.75는 FP32(entry+alpha×FP32(native−entry)). 두 fit 사이 current M append0. 평가는 최종 materialized W에서 하고 이후 같은 W/M/RNG로 이어진다. 평가 점수는 controller/fallback/alpha선택에 입력되지 않았다.')
 sec('3. 종료 상태와 state/checkpoint 검산')
 add(table(load('scheduler'),['job','arm','state','exit_code','start','end','elapsed_seconds','allocated_gpus','mem_MiB']))
 add('6/6 scheduler COMPLETED0과 scientific record 검산을 구분했다. 당시 agent가 직접 관측한 초기 gate는 N4 B51→B52뿐이다. 이번 recall에서 나머지 저장 receipt를 사후 확인하며 과거 관측으로 소급하지 않는다. 신규 GPU/forward/Slurm 변경0. 입력 재사용 검증·신규 full SHA·CPU tensor 검증을 각각 manifest에 구분했다.')
 for name in ['state-verification.json','metric-reduction-receipt.json']:
  if (p/name).exists():
   d=json.loads((p/name).read_text());add(name+'의 scalar 결과:\n\n'+table([{'field':k,'value':v} for k,v in d.items() if isinstance(v,(str,int,float,bool))],['field','value']))
 add('예상 범위는60commits/54 이전commit→nextentry links/18CP(B51/55/60)/90fit·solve/9000request-z/L4append60/L8append20. 실제 체크 목록·member SHA·크기·dtype·shape·finite·선택 tensor hash·history counters는 state-history.csv/checkpoint-inventory.csv/raw-member-inventory에 있다. 증분 journal이 존재해도 GPU continuation replay 또는 bitwise 재실행 성공으로 부르지 않는다. Shared base/P/stats/context와 공통 prepared가 복원 closure의 일부다. 새 full-model load·GPU off/on parity는 NOT_TESTED. Parameter pointer/version 및 선택 W/M/P/context/RNG를 감시했지만 module buffers 전체별 byte parity는 별도 미검증이다.')
 add('6개 output directory의462파일/45,736,205,134 bytes를 새 full SHA/size/stable-stat 검산했다. 이는 attempt 전체 control/stdout/source를 뜻하지 않으며 별도 inventory로 결속한다. CP/increments/evaluation/commit은 기존 runtime member SHA와 대조했다. Target-key-readout capture는 기존 commit에 이전 file SHA가 없어 현재 full SHA/finite/cardinality를 봉인한 수준이다. .75/.875 중간 native 후보 전체 weight는 저장되지 않아 alpha 검산은 source와 기록된 materialization에 결속하고 새 native solve 재구성은 하지 않았다. Alpha1 endpoint-copy SHA 및 B51 실제 increment=CP−prepared FP32는 직접 검산했다.')
 sec('4. 최종 모집단별 성능과 고정 패널')
 for pop in ['entry_old','suffix','current','historical']:
  rr=[]
  for a in ARMS:
   d={r['metric']:r for r in fm if r['arm']==a and r['population']==pop and r['group']=='ALL'}
   rr.append({'arm':a,**{m:f"{d[m]['numerator']}/{d[m]['prompt_denominator']} ({100*float(d[m]['rate']):.5f}%)" for m in ['RS','PS','NS']}})
  add('\n### W60 '+pop+'\n');add(table(rr,['arm','RS','PS','NS']))
 add('\n### W60 general\n');add(table([r for r in gen if r['batch']=='60'],['arm','wiki_nll','wiki_sequences','wiki_tokens','mmlu_correct','mmlu_denominator','mmlu_invalid']))
 add('전 arm Wiki128/24999 token, MMLU32이다. 고정32 개발 반복·known corpus 한계가 있으며 전체 MMLU 또는 general 능력 동등성으로 읽지 않는다. Audit128/N1280 및 MMLU68/FutureN 별도 성능조회·평가0.')
 sec('5. 모든60batch current·historical·general 곡선')
 for a in ARMS:
  rr=[]
  for b in range(51,61):
   d={(r['population'],r['metric']):r for r in bm if r['arm']==a and int(r['batch'])==b and r['group']=='ALL'};g=next(r for r in gen if r['arm']==a and int(r['batch'])==b)
   rr.append({'B':b,**{pop+' '+m:f"{d[(pop,m)]['numerator']}/{d[(pop,m)]['prompt_denominator']}" for pop in ['current','historical'] for m in ['RS','PS','NS']},'Wiki':g['wiki_nll'],'MMLU':g['mmlu_correct']+'/32'})
  add('\n### '+a+'\n');add(table(rr,['B','current RS','current PS','current NS','historical RS','historical PS','historical NS','Wiki','MMLU']))
 sec('6. Suffix 누적 retention·문항별 loss/recovery')
 rr=[r for r in paired if r['contrast']=='AT_WRITE_TO_SUFFIX' and r['group']=='ALL' and not r.get('cohort_batch')]
 add(table(rr,['after','batch','metric','prompt_denominator','before_numerator','after_numerator','lost','gained','retained','failed_both','conditional_loss_rate']))
 add('before는 각 문항의 자기 at-write W에서의 성공이며 한 모델 상태가 아니다. after는 W55/W60 단일 실제 상태다. lost 조건부 분모는 before-success, gained 조건부 분모는 before-failure. 매 시점 모든 old 문항을 평가하지 않았으므로 정확한 최초 forgetting/recovery 시점은 NOT_RECORDED. At-write pooling은 batchmetrics.csv의 ONLINE_AT_WRITE_1000_DIFFERENT_STATES 행으로 별도 제공한다. Case별 subject/relation/target 최종 event 비교로 ACTIVE_TARGET/SUPERSEDED/UNKNOWN을 나누고 원 ALL분모는 보존한다. 의도 추정이나 성능 기반 제외는 하지 않았다.')
 add('60개 원 cohort의 W60 RS/PS/NS와 strict/token은 cohort-final.csv(1080rows), old/new/early/middle/recent 및 active 보조집계는 finalpopulationmetrics.csv에 있다. 전체 10×60 checkpoint-cohort matrix는 평가되지 않았으므로 존재한다고 주장하지 않는다.')
 sec('7. 고정 대비의 양방향 전이와 NLL 손실')
 rr=[r for r in paired if r['batch']=='60' and r['population']=='fullseen' and r['group']=='ALL' and r['before']!=r['after']]
 add(table(rr,['before','after','metric','prompt_denominator','lost','gained','delta_numerator','delta_pp','new_nll_delta_mean','new_nll_delta_p95','true_nll_delta_mean','desired_margin_delta_mean']))
 add('최종 NS에서 N4→RES8 lost1472/gained1555, S875574/720, S75742/1060, FULL81067/1097, REFIT41154/1434다. 따라서 총점의 증가가 모든 기존 성공 문항 보존을 뜻하지 않는다. S75는 N4 대비 fullseen NS +318, PS −50; RES8은 NS +83, PS +10, RS −8이다. 원인·우월성 선택은 이 표만으로 확정하지 않는다. RES8−S75는 추가 fitting, RES8−REFIT4는 같은 추가 target-call 예산의 layer 차이, RES8−FULL8은 첫 write 축소 차이를 포함하며 각 trajectory의 W/z가 달라진다.')
 sec('8. NLL와 signed margin 분포 및 secondary accuracy')
 for pop in ['fullseen','entry_old','suffix','current']:
  add('\n### W60 '+pop+' PROMPT 분포\n')
  rr=[r for r in nl if r['batch']=='60' and r['population']==pop and r['group']=='ALL' and r['aggregation_unit']=='PROMPT']
  add(table(rr,['arm','metric','field','n','mean','median','p90','p95','p99','max']))
 add('전체60batch의 prompt 및 request-cluster 평균 분포는 nll-distributions.csv에 완전 수록했다. Prompt와 cluster를 같은 분모로 섞지 않는다. New/true NLL을 각각 보존하며 NS true 약화와 new 경쟁 강화의 차이는 paired-transitions.csv 두 필드에서 분리한다. 모든 strict/token 정답수와 실제 token분모는 batchmetrics/finalpopulationmetrics의 new_*/true_*열에 있다.')
 rr=[r for r in fm if r['population']=='fullseen' and r['group']=='ALL']
 add(table(rr,['arm','metric','new_strict_numerator','prompt_denominator','new_token_correct','new_token_denominator','true_strict_numerator','true_token_correct','true_token_denominator']))
 sec('9. Static core B51 대조와 source 경계')
 rr=load('static-core-comparison');add(table(rr,['arm','population','metric','before_numerator','after_numerator','lost','gained','new_nll_delta_mean','new_nll_delta_max','true_nll_delta_max']))
 add('기존46451 static6와 신규 B51 current/Historical의36개 metric 성공 count 차이는 모두0이다. 수치 분포 차이는 위 표 그대로이며 count equality를 모든 tensor/kernel parity로 확대하지 않는다. 이전 static은 branch-entry RNG reset, 신규 sequential은 first fit 뒤 자체 RNG chronology를 이어간다는 source 차이를 entry receipt에 미리 기록했다. 선택 W/M/context/RNG actual hashes는 state 검산 범위에만 결속한다. Fullseen6000은 static core에 없어 대체 비교0.')
 sec('10. 실제 layer action·history')
 if (p/'layer-action.csv').exists():
  rr=load('layer-action');add(table(rr,list(rr[0])[:12]))
 add('실제 batch-net delta와 endpoint-net, 그 norm의 합인 path를 구분한다. Finalizer의 key Gram append는 update write가 아니며 residual realization/native metric 또는 intrinsic capacity로 재명명하지 않는다. 저장되지 않은 공통 terminal activation/realization은 NOT_RECORDED. Nonselected weight 보존은 runtime 시작/종료 bytes 및 중간 pointer/version evidence이며 새 GPU replay 검증은 아니다.')
 sec('11. 실측 비용·동시성·미분리 항목')
 add(table(cost,['arm','allocated_GPUh','program_seconds','policy_instrumented_online_seconds','first_fit_seconds','second_fit_seconds','finalization_seconds','evaluation_seconds','online_ratio_N4','request_z','loss_evaluations','adam_updates','early_stop_requests']))
 account=json.loads((p/'compute-accounting.json').read_text());add(table([{'field':k,'value':v} for k,v in account.items()],['field','value']))
 add('총33475 GPU-sec=9.298611 GPUh. 같은 node의 개별1GPU×최대2개 동시 실행이었다. Prepared/M8 기존276.130524초는 이번 allocation에 중복 더하지 않는다. Donor 정책 준비비용을 신규 배포에 상각한다면 별도1회 비용으로 명시해야 한다. Online ratio는 계측 포함 같은 component의 비율이며 순수 writer는 NOT_SEPARATED. 추가 target 단계의 조기 종료로 call수와 loss/Adam횟수가 다르다. 로그에서 request-z segment별 loss count를 source loop와 결속해 Adam=loss−1로 계산했고 이는 native structured Adam counter와 구분한다. Exact loss/clamp event/per-z synchronous GPU시간은 NOT_RECORDED. CPU 분석 도중 python 기본환경 matplotlib 부재는 분석환경 오류였으며 기존 venv 코드 생성으로 처리, 실험 실패/분모 변경0.')
 sec('12. 참고선·누락·검산 한계')
 if (p/'quality-frontier.csv').exists():
  qq=load('quality-frontier');add('모든 arm·batch 참고선 판정은 quality-frontier.csv에 수록한다. 단독 또는 AND gate로 탈락시키지 않았다. 컬럼 및 최종batch 전체:\n');add(table([r for r in qq if r.get('batch')=='60'],list(qq[0]) if qq else []))
 add('감사 N1280/MMLU68 미사용, policy selection 미실행, 다른 entry/full10k/general 확증0. 단일 fixed order의 repeated measurements이며 독립 반복 시행이나 iid neighborhood1000으로 유의성을 주장하지 않는다. Bootstrap CI는 이번 보고에 산출하지 않았다. 상태/hash PASS는 kernel/model-level off-on parity나 과학적 원인 규명을 대신하지 않는다. Finite poor outcome 모두 원분모 포함, imputation/rescue/제외0. 이전 stopped ORBODE 및 타 실험에 접근/재개하지 않았다.')
 sec('13. 재현·산출물 inventory')
 add('실행 원본은 frozen source archive 및 executed-source-identities.csv로 추적한다. 본 보고서의 analysis code와 CSV/PNG는 Git에 보존하나 checkpoints/tensors/rawprompt/fullstdout는 local-only이다. Raw root: `/data/janghj/ODE-edit/local/low-cost-write-donor-seq10/20260913-v1/attempt-v1/output`. NO_BROADCAST_NOT_REQUIRED: CPU 분석에 기존 local 자료만 사용하여 신규 전송·삭제0.')
 add('실제 재현 명령은 reproduction.md에 기록한다. PNG/report는 `python -m project.run_scripts.low_cost_write_donor_pilot.seq_review_plots --root <package>` 및 `python -m project.run_scripts.low_cost_write_donor_pilot.seq_review_report --root <package>`. 실제 실행 Python과 명령/입력SHA/출력SHA는 plot-receipt.json에 기록한다. 새 model forward는 필요하지 않다. package full rehash와 source binding은 analysis-manifest.json/rooted-receipt.json에서 확인한다. 신규 output 462files/45,736,205,134bytes를 full SHA 검산했으며 198개는 이전 runtime 기록 SHA와도 대조했다. 나머지는 현재 bytes의 신규 inventory이며 이전 봉인 SHA가 있었다고 주장하지 않는다. 18CP selected state 합계25,367,842,794bytes. 중간 alpha의 native candidate weight 자체는 미저장으로 source/materialization 기록 이상 재계산을 주장하지 않는다.')
 for r in json.loads((p/'plot-receipt.json').read_text()):add(f"\n### {r['path']}\n\n![{r['path']}]({r['path']})\n\n{r['caption']} SHA256={r['sha256']}\n")
 inv=[]
 for f in sorted(p.glob('*.csv')):
  with f.open() as q:count=sum(1 for _ in csv.reader(q))-1
  inv.append({'table':f.name,'rows':count})
 add(table(inv,['table','rows']))
 add('최종 상태: 완료6chain의 CPU 결과 검산·raw-free publication 범위. 전체pilot의 후보/audit/추가suffix 또는 장기 안정성 승인 아님. Claim decision=PENDING_GH_REVIEW. TASK_COMPLETE_STOP은 package/main 검증 후 별도 completion receipt에 기록한다.')
 (p/'diagnostic-report-ko.md').write_text('\n\n'.join(s)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
