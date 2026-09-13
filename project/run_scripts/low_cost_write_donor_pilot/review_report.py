"""Raw-free Korean report assembly from verified tables (no evaluation)."""
import argparse,csv,json
from pathlib import Path
from .review_provenance import sha,write,csvwrite
ARMS=['N4','S875','S75','FULL8','RES8','REFIT4']
def rows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def fmt(v):
 if v is None or v=='':return 'NOT_RECORDED'
 if isinstance(v,float):return f'{v:.8g}'
 return str(v).replace('|','/')
def table(rs,cols):
 return '\n| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(fmt(r.get(c)) for c in cols)+' |' for r in rs)+'\n'
def run(root,attempt,state_path):
 root=Path(root);attempt=Path(attempt);load=lambda p:json.loads(Path(p).read_text())
 state=load(state_path);write(root/'state-verification.json',state)
 action=[]
 for r in state['rows']:
  total=sum(r['delta_norm'].values())
  for layer,n in r['delta_norm'].items():action.append(dict(arm=r['arm'],layer=int(layer),norm=n,share=n/total if total else 0,unit='FROBENIUS_STORED_ENDPOINT_MINUS_COMMON_ENTRY',snapshot_sha256=r['snapshot_file_sha256']))
 csvwrite(root/'layer-action.csv',action)
 lock=load(attempt/'execution.lock.json');cp=load(root/'comparison-capsule.json');acc=load(root/'accounting.json')
 cfg4=load(lock['config4']);cfg8=load(lock['config8'])
 write(root/'source-config-compatibility.json',dict(execution_commit=lock['worktree_head'],analysis_base='ddb70506d90d2c53642251e4683759a4ee0d9bfd',blue_head=lock['blue_head'],editor_sha256=lock['editor_sha256'],model_revision=lock['model_revision'],snapshot=lock['snapshot'],config4=cfg4,config8=cfg8,config_sha256={x:sha(lock[x]) for x in ['config4','config8']},parameter_dtype='FP32',torch=lock['torch'],transformers=lock['transformers'],attention='eager',tf32_matmul=lock['tf32_matmul'],tf32_cudnn=lock['tf32_cudnn'],target_mode='SAME_HOST_FRESH_NATIVE',native_full_five_layer_L2_10=False,writer_padding='right; add_bos_token=False',counterfact_evaluation='canonical MB16 manual left padding, default evaluator tokenizer, no position override',general_evaluation='Wiki one-sequence; MMLU source alternative four forwards/row; not MB16',source_equation='Original BLUE singleton L2=1; fit/finalization split; model-level logger off/on NOT_TESTED'))
 core=rows(root/'reference-and-core-table.csv');m=rows(root/'endpoint-metrics.csv');dist=rows(root/'nll-distributions.csv');pairs=rows(root/'paired-transitions.csv');cost=rows(root/'compute-ledger.csv');quality=rows(root/'quality-frontier.csv')
 practical=[]
 for r in core:
  if r['state'] not in ARMS:continue
  cc=next(c for c in cost if c['arm']==r['state'])
  for name,value,threshold,unit in [('historical_NS_gain_shortfall_to_1pp',1-float(r['historical_NS_delta_vs_N4_pp']),0,'pp shortfall'),('current_NS_degradation',-float(r['current_NS_delta_vs_N4_pp']),0,'pp loss'),('online_ratio_1_5',float(cc['online_ratio_to_N4']),1.5,'instrumented ratio'),('online_ratio_2',float(cc['online_ratio_to_N4']),2.,'instrumented ratio')]:
   practical.append(dict(arm=r['state'],reference=name,value=value,threshold=threshold,status='WITHIN' if value<=threshold else 'EXCEEDS',unit=unit,enforcement='NO_AUTOMATIC_REJECTION'))
 csvwrite(root/'practical-references.csv',practical)
 overview=[]
 for r in core:
  if r['state'] not in ARMS:continue
  d={'arm':r['state']}
  for p in ['current','historical']:
   for k in ['RS','PS','NS']:d[p+' '+k]=f"{r[p+'_'+k+'_n']}/{r[p+'_'+k+'_d']} ({100*float(r[p+'_'+k+'_rate']):.4f}%)"
  d['Wiki NLL']=float(r['wiki_sequence_mean_nll']);d['MMLU correct']=r['mmlu_correct']+'/32';d['MMLU invalid']=r['mmlu_invalid'];overview.append(d)
 report='''# Low-cost write/donor core — Server4 상세 사실 보고

이 보고서는 job **46451**의 여섯 정적 endpoint를 다룬다. 전체 pilot 완료·후보 채택 보고가 아니다.
Instruction: ODEEDIT-S06-LOW-COST-WRITE-DONOR-CORE-DETAILED-REVIEW-SH4-V1.
claim-decision=`PENDING_GH_REVIEW`, scientific_promotion=false. Audit·5-batch suffix 미실행.

## 1. 첫 실제 성능표

모두 같은 Server4 Llama-3-8B-Instruct, 공통 L4 W50/M50, B051(고정10k ordinal5000–5099)의 실제 저장 endpoint다.
Current와 Historical은 다른 고정 패널이다. 아래 수치는 full-seen5100 또는 lifelong final10k가 아니다.
RS/PS/NS는 prompt-level strict NLL 선호이고 tie는 실패다. Wiki는 sequence 평균, MMLU는32문항 alternative 정수 정답이다.
'''+table(overview,list(overview[0]))+'''
N4 대비 S875는 Current PS −1/200, NS +3/1000이다. S75는 PS −2/200, NS +7/1000이다.
FULL8은 PS +2/200, Current NS −2/1000, Historical NS +5/1280이다.
RES8은 Current PS 순변화0, NS +1/1000, Historical PS −1/256, NS +2/1280이다.
REFIT4는 Current PS 순변화0, NS +5/1000, Historical NS 순변화0이다.
MMLU는 S75·RES8에서 N4보다 정답1개 적고 나머지는 순변화0이다. 모든 arm Current RS100/100이다.
총점 유지가 같은 문항 유지라는 뜻은 아니다. RES8/N4 Historical NS는 loss11/gain13,
S75/N4는 loss5/gain5, REFIT4/N4는 loss8/gain8이다. RES8/N4 Current PS도 loss1/gain1이다.
이는 원시 수치와 산술 차이이며 원인·우열·후속 선택 판단은 GH 소유다.

## 2. 지표 읽는 법

- 각 prompt target의 NLL은 teacher forcing의 target-token 평균 음의 log probability이며 단위는 nats/token, 낮을수록 그 target likelihood가 높다.
- RS=rewrite에서 new NLL < true NLL; PS=paraphrase 각각에서 new < true; NS=neighbor 각각에서 true < new. 성공률은 성공 prompt 수/해당 prompt 수, 높을수록 정의된 선호가 많다. tie는 모두 실패다.
- Desired signed margin은 R/P에서 true−new, N에서 new−true이다. 양수일 때 성공이며 클수록 두 target 간 해당 방향 차이가 크다. true/new NLL 자체와 margin을 별도로 제공한다.
- `new_strict`/`true_strict`는 모든 target token의 top-1이 정답인 prompt의 수다. `token_accuracy`는 맞은 target token/전체 target token이다. NLL pair 선호와 다른 secondary이며 NS로 대체하지 않는다. request-all-success는 해당 request의 모든 prompt 성공을 요구하는 별도 secondary다.
- Prompt 분포와 request-cluster mean 분포를 분리한다. 후자는 같은 request의 P2/N10 값을 먼저 평균한다. mean/median/IQR/p90/p95/p99/max는 기술통계다. Quantile은 정렬 배열의 (n−1)p 선형보간으로 계산한다. 이는 누락값 imputation이 아니다.
- Lost는 기준 성공→후보 실패, gained는 기준 실패→후보 성공이다. net=gained−lost. Conditional loss denominator는 기준 성공 수, recovery denominator는 기준 실패 수다. 같은 request 내 prompt 상관이 있어1000/1280 independent trials라는 유의성 주장을 하지 않는다.
- Wiki128은 기존 고정 input_ids의 모든 다음-token 위치24999개를 평가하되 primary는128개 sequence 평균 NLL의 산술평균이다. token-weighted 평균은 별도 CSV다. Full-vocabulary teacher KL 또는 perplexity 평가로 부르지 않는다.
- MMLUdev32는 기존100문항의 outcome-independent hash subset이다. BLUE alternative branch에서 네 suffix의 exp(−NLL) unique maximum을 선택하며 tie/underflow 동률은 invalid. Generation/weighted F1/full57subject benchmark가 아니다.
- D4=WN4−We; Sα=FP32(We+α FP32(D4)). Layer-wise Update Magnitude는 실제 저장 endpoint−We의 Frobenius norm이다. Share는 두 norm의 합으로 나눈 비율이며 energy/native metric이 아니다. REFIT4 net norm은 두 write path length 합이 아니다.

## 3. 실행·입력·상태 provenance

Scheduler 단발 확인: COMPLETED/0:0, 2026-09-13 19:22:12–19:57:44,2132초,
Server4 1GPU/8CPU/60416M. 이전 agent 마지막 관측은 PENDING/initial NOT_YET_RUN이었다.
이번 recall에서만 저장된 INITIAL_VALID를 검증했다. 초기 receipt는 RES8까지5endpoint와 partial second-fit 후 entry 복원이며 core_complete=false; 후속 core-terminal은6endpoint 완료다.

실행 source7ece056c33fbb4246245c15f5f7c2a678315c05c/tree352cdd8ca3f3d7b6f2b15f3ed6a6f775fb478443.
Archive d6cf34ab416ee313a01c8491ac0ab9d15a0f39d88a5ea8f7ead43b439dbe4312,
lock a672a782c20543b66add9ed83dc5cf04432e9f98f6cc2646e40b68e3627a9317.
분석 시작main ddb70506d90d2c53642251e4683759a4ee0d9bfd와 실행 source를 구분한다.
원본 실행 코드는 수식 변경 없이 own-scope 그대로 게시하며 runtime 재실행은 없다.
BLUE311b076a, adapter b51dcf5, schema repair58f50a를 frozen closure에 결속했다.
모델 revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2; dataset SHA3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1,
ordered root5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729.
FP32/eager, torch2.9.1+cu128, transformers4.44.2, TF32 matmul=false/cudnn=true.
원본 singleton BLUE L2=1·native layer-local z이며 native five-layer blue=False L2=10 baseline이 아니다.
모든 실제 config key/value와 source/hashes는 source-config-compatibility.json 및 execution-member-identities.csv에 있다.

S2에서 필요한 정확 W50/M50 한 CP만 선택 수신했고 S1의 Historical/Wiki 완료 패널 identity를 재사용했다.
S1 native WN을 matched reference로 사용하지 않았다. Original-target replay0/same-host target-replay0,
Server4 same-host fresh-native1이다. 원 model/P/stats 대형자산의 과거 검산은 실행lock 및 성공한 job의 선행 fullSHA 검사에 결속 재사용했다.
신규 output34파일 전체14,853,827,915 bytes를 새 fullSHA/size/stable-stat 검사했다. 누락/실패endpoint0; 평가중복/비유한/imputation0은 독립 reducer 범위에서 확인했다.

### 3.1 분기·history·snapshot

7개 CPU mmap weights_only snapshot(prepared+6endpoint)의 selected W4/W8/M4/M8를 FP32/finite/tensorSHA로 확인했다.
공통 entry W/M/RNG/context 복원과 endpoint→평가 state exact 결속을 저장 receipt/소스 assertion/실제 tensor로 대조했다.
S875/S75 CPU FP32 materialization은 actual D4 기준으로 byte 일치한다. α1은 N4 snapshot copy;
α0은 source/기존 synthetic 검사만 있으며 실행 core endpoint가 아니다.
두 번째 fit은 FULL8 actualN4, RES8 actualS75, REFIT4 actualS75에서 각각 새 z/K/R을 계산했다.
4개 captured z-order roots는 서로 다르며 fits400z/4solve와 일치한다.

M8은 공통 We에서50×B100=5000 과거 event/context를 한 번 재인코딩한 FP32 CPU Gram이다.
Physical L4→Passet0→singleton0, L8→Passet4→singleton0이다. 같은 M8를 FULL8/RES8의 fit-entry에 썼다.
각 fit의 current history append0; 최종 N4/S875/S75/REFIT4 각L4 once,
FULL8/RES8 각L4+L8 once로 총8append다. REFIT4 두 fitting을 두 번 append하지 않았다.
M8 과거 K는 hash만 저장되어 CPU만으로 재인코딩 numerical replay는 하지 않았다.
M4 endpoint hashes가 여섯 endpoint에서 같은 사실은 raw에 보존한다.
이를 history 누락 또는 model-level parity 증명으로 단정하지 않는다.

종료 process-restore는 selected W0 exact와 RNG 복원, method-state process discard다. parameter version 원복이나 entry M으로 process 종료했다고 하지 않는다.
비선택 parameter 보존은 runtime pointer/version 및 full-byte assertion에 근거한다. 새 full model GPU 재검증0.
모든 checkpoint는 selected W/M/context/RNG + pinned pretrained/P/source closure로 복원하는 snapshot이며 full standalone pretrained model이 아니다.
GPU continuation replay 및 비계측 native와 model-level parity는 NOT_TESTED. AST history-loop 분리 동등성 검사는 이보다 좁은 소스 검사다.
'''
 report+='\n### 3.2 실제 layer action\n'+table(action,['arm','layer','norm','share'])
 report+='\n## 4. Current/Historical 및 W0/entry 상세\n\nW0와 ENTRY는 같은 고정 패널의 준비 reference이며 전10k W0가 아니다. 모든 표는 원분모를 유지한다.\n'
 for p in ['current','historical']:
  rr=[r for r in m if r['panel']==p and r['population']=='ALL']
  report+='\n### '+p+' 전체 prompt 성능·secondary\n'+table(rr,['state','metric','numerator','prompt_denominator','rate','ties','new_strict_numerator','true_strict_numerator','new_token_correct','new_token_denominator','true_token_correct','true_token_denominator','request_all_success_numerator','request_denominator'])
 report+='\n### Historical active/superseded\n\n124 active-target /4 superseded requests다. status는 entry5000 이전 exact subject/relation의 최신 target 기준이다. 새로운 동의어/semantic collision을 검출했다고 주장하지 않는다. 원 ALL 분모와 보조 집단을 분리한다.\n'
 report+=table([r for r in m if r['panel']=='historical' and r['population']!='ALL' and r['state'] in ARMS],['state','population','metric','numerator','prompt_denominator','request_denominator','new_strict_numerator','true_strict_numerator'])
 report+='\n## 5. NLL 및 margin 전체 분포\n\n아래는 각 core의 prompt-level 완전표다. Request-cluster mean의 mean/median/IQR/p90/p95/p99/max는 같은 nll-distributions.csv의 별도 aggregation_unit 행에 모두 제공한다. 0개 집단은 n0, 통계 NOT_RECORDED이며 보간하지 않는다.\n'
 for a in ARMS:
  report+='\n### '+a+' NLL/desired margin (nats/token)\n'+table([r for r in dist if r['state']==a and r['population']=='ALL' and r['aggregation_unit']=='PROMPT'],['panel','metric','field','n','mean','median','iqr','p90','p95','p99','max'])
 report+='\n## 6. Matched 대비·양방향 전이\n\n각 대비는 동일 prompt/target identity와 순서로 join했다. 순차 trajectory의 장기 효과가 아닌 같은 entry의 1-batch 정적 대비다. 각 true/new NLL 변화와 request-cluster paired 분포 전체는 paired-transitions.csv 및 paired-request-distributions.csv에 있다.\n'
 selected=[r for r in pairs if r['population']=='ALL' and ((r['before']=='N4' and r['after'] in ARMS) or (r['after']=='RES8' and r['before'] in ['S75','REFIT4','FULL8']))]
 report+=table(selected,['before','after','panel','metric','prompt_denominator','lost','gained','retained','failed_both','delta_pp','new_nll_delta_mean','true_nll_delta_mean','desired_margin_delta_mean'])
 report+='\nRES8−S75 Current PS는 +2/200, NS는 −6/1000이다. RES8−REFIT4 Current PS는 같고 NS −4/1000, Historical NS +2/1280, PS −1/256이다. RES8−FULL8 Current PS −2/200, NS +3/1000, Historical NS −3/1280이다. MMLU RES8은 REFIT4/FULL8보다1개 적다. 이 반대방향들을 단일 우월성 판정으로 합치지 않는다.\n'
 report+='\n### W0/entry→endpoint 전이\n'+table([r for r in pairs if r['population']=='ALL' and r['metric']=='NS' and r['before'] in ['W0','ENTRY']],['before','after','panel','prompt_denominator','before_numerator','after_numerator','lost','gained','conditional_loss_rate','conditional_gain_rate'])
 report+='\n## 7. General sentinel·참고선\n'+table(rows(root/'paired-sentinels.csv'),['before','after','panel','denominator','lost','gained','delta_correct','worse','better','mean','p95','max'])
 report+='\nConfidence interval은 이번 package에서 계산하지 않았다(NOT_RECORDED). 한 entry/order의 paired 기술통계이며 checkpoint 반복이나 N prompt를 독립 반복으로 다루지 않는다. 아래 참고선은 기계적 WITHIN/EXCEEDS/NOT_RECORDED이고 AND gate/후보 탈락 규칙이 아니다.\n'+table(quality,['state','reference','metric','value','threshold','status','unit','denominator'])
 report+='\n품질 참고선78행 중 초과10행: S875/S75/RES8/REFIT4의 Current PS newNLL 평균 및 paired q95가 각각 참고선을 초과한다. S75 Current R strict 순손실2와 P strict 순손실11은 각각1/4개 참고선을 초과한다. 원시값은 위 표에 모두 있다. 아래 Historical NS의1pp 참고선은 shortfall=1−gain을 사용하여 양수일 때 EXCEEDS로 표기하며, 이득 양수 여부·허용 여부와 동일하지 않다.\n'+table(practical,['arm','reference','value','threshold','status','unit'])
 report+='\n## 8. 비용·저장량\n'+table(cost,['arm','policy_instrumented_online_seconds','online_ratio_to_N4','second_fit_seconds','materialization_seconds','finalization_seconds','evaluation_seconds','policy_M8_setup_seconds'])
 report+='\n### Fit별 실제 호출/로그 복원\n'+table(rows(root/'fit-cost.csv'),['arm','compute_z','loss_evaluations','adam_updates','compute_z_seconds','compute_ks_seconds','solve_seconds','seconds','actual_delta_norm'])
 report+=f'''\n총 scheduler allocated GPUh={acc['allocated_GPUh']:.9f}, elapsed2132초. 프로그램 계측총 {acc['seconds']:.6f}초, model load {acc['preparation']['model_seconds']:.6f}초, 공통 M8준비 {acc['preparation']['M8_seconds']:.6f}초다.
N4 loss2500/update2400, FULL8 244/144, RES8 316/216, REFIT4 316/216; 총400z/3376loss/2976updates.
이는 stdout 400개 순차 segment와 원본 break-before-Adam source에 근거한 사후 CPU 복원이며 raw structured fit의 iteration counter로 오인하지 않는다.
원본 최대25loss/max24Adam budget은 유지됐지만 모든 z가 최대횟수를 소비한 것은 아니다.
Peak allocated GPU bytes={acc['peak_allocated_bytes']}, reserved={acc['peak_reserved_bytes']}; Slurm hostMaxRSS52.27G와 다른 지표다.
출력14,853,827,915 bytes는 experiment output만이며 input/model cache까지 포함한 disk consumption이 아니다.
기존2–8GPUh/20–40GiB는 제출 전 추정으로 실측과 구분한다.

Policy 원가에는 첫 native fit을 각 policy에 포함한다. 연구 총지출에서는 native fit 한 번, M8준비 한 번이다.
FULL8/RES8에 표시한 같은 M8setup을 두 번 실제 실행비용으로 합산하지 않는다. Setup을5개 batch에 균등 상각하면 donor 추가55.226105초/B100이라는 단순 산술이며 실제 suffix timing이 아니다.
Online 비율1.5x/2x reference는 모든 후보에서 이내지만 diagnostics/hash/capture 포함 계측이다. Pure writer cost는 NOT_SEPARATED이고 cross-host controlled speedup이나 저비용 claim을 확정하지 않는다.
P/load/restore/I-O/diagnostic 각각의 별도 총시간은 NOT_RECORDED; 총시간에서 차감한 나머지를 특정 항목으로 이름붙이지 않는다. Technical failure receipt0, 이 지정attempt의 취소/재시작0이다.
'''
 report+='\n## 9. Audit·후속 범위와 미검증\n\n사전 seal audit128/N1280은 개발228case와 알려진 exact subject/relation disjoint를 CPU 확인했다. MMLU32/68 row hash disjoint다. 이번 검토에서 audit identity metadata는 읽었지만 audit 성능은 평가·열람하지 않았다. 이미 본 fixed10k/기존MMLU100 corpus이므로 globally blind라고 부르지 않는다. Unknown semantic overlap은 제외되지 않았다. Audit/선택/FutureN/suffix 실행0; fullseen5500은 evaluator 연결만 준비됐고 측정0이다.\n\n새forward/replay/nativewrite 없이 저장자료만 분석했다. 모델 수준 off/on parity, GPU continuation, full-vocabulary teacher, causal mechanism attribution, 5-batch 지속성, full10k안정성, 파라미터버전 복원, 미기록 native metric/JVP는 모두 미검증 또는 NOT_RECORDED다. Finite poor outcome을 제외하지 않았으며 임의 후보 선택도 하지 않았다.\n'
 report+='\n## 10. 재현·artifact inventory\n\n원문: project/proposals/2026-09-13-low-cost-write-donor-pilot-gh-instruction.md 및 plans/global/2026-09-13-low-cost-write-donor-pilot-design.md. 상세 상태 감사는 audits/servers/server4/2026-09-13-lowcost-core-detailed-review/state-audit.md를 참조한다.\n\n모든 CSV는 원시 prompt 없이 집계/hash만 저장했다. 원본 tensor/log/평가row는 local-only에 그대로 있다. Raw broadcast는 NO_BROADCAST_NOT_REQUIRED.\n\n```bash\npython3 -m project.run_scripts.low_cost_write_donor_pilot.review_metrics --root <attempt>/output --out <package> --dataset <fixed10k>/counterfact.json --panel-lock <attempt>/panel-lock.json --contract plans/global/2026-09-13-low-cost-write-donor-pilot-contract.json\npython3 -m project.run_scripts.low_cost_write_donor_pilot.review_provenance --attempt <attempt> --out <package>\npython3 -m project.run_scripts.low_cost_write_donor_pilot.review_panels --attempt <attempt> --out <package>\npython3 -m project.run_scripts.low_cost_write_donor_pilot.review_report --root <package> --attempt <attempt> --state <state-cpu-review-v1.json>\npython3 -m project.run_scripts.low_cost_write_donor_pilot.review_plots --root <package>\n```\n'
 captions={x['path']:x['caption'] for x in load(root/'plot-receipt.json')} if (root/'plot-receipt.json').exists() else {}
 for f in ['core-performance.png','nll-tails.png','paired-locality.png','layer-update-magnitude.png','compute.png']:report+=f'\n![{f}]({f})\n\n'+captions.get(f,'Caption: plot-receipt.json')+'\n'
 report+='\n각 figure의 exact denominator/missing policy/소스·입력·출력SHA는 plot-receipt.json에 봉인한다. 코드 고정Agg/seed20260913/DPI160/size/색상/ordering; 재실행byte동일 검증을 기록한다.\n'
 report+='\n### 전체 CSV 및 검증 JSON\n'
 inv=[]
 for p in sorted(root.iterdir()):
  if p.suffix in ['.csv','.json'] and p.name not in ['analysis-manifest.json','rooted-receipt.json']:
   inv.append(dict(file=p.name,rows=len(rows(p)) if p.suffix=='.csv' else 'JSON',bytes=p.stat().st_size,sha256=sha(p)))
 report+=table(inv,['file','rows','bytes','sha256'])
 (root/'diagnostic-report-ko.md').write_text(report)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--attempt',required=True);p.add_argument('--state',required=True);a=p.parse_args();run(a.root,a.attempt,a.state)
