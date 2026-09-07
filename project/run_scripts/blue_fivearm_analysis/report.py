"""Korean factual report assembled from independently checked aggregate tables."""
import argparse
from .common import *
from .metrics import headline

def run(repo):
 out=repo/OUT_REL
 def load(n):return csvread(out/(n+'.csv'))
 final=load('final_metrics');cur=load('current_batch');seen=load('seen_prefix');rw=load('allseen_rewrite');ret=load('retention_cohort')
 dist=load('nll_margin_distributions');parts=load('endpoint_partitions');cost=load('compute');actions=load('layer_action');compat=load('source-config-compatibility');trans=load('prompt_transitions')
 f={r['arm']:r for r in final};lines=[]
 def add(s):lines.append(s+'\n')
 def t(rs,cols):add(table(rs,cols))
 def rates(rs):
  result=[]
  for r in rs:
   z=dict(arm=r['arm'],batch=r.get('batch',0),requests=r.get('request_count',NA))
   for metric in ['RS','PS','NS']:
    if r.get(metric+'_num','')!='':z[metric]=f"{r[metric+'_num']}/{r[metric+'_den']} ({100*float(r[metric+'_rate']):.2f}%)"
   z['PS_all2']=f"{r.get('PS_strict_num',NA)}/{r.get('PS_strict_den',NA)}"
   result.append(z)
  return result
 add('# BLUE / L4-only / L8-only / JVP / JVP-L8 — Llama sequential 1,000 상세 비교')
 add('Canonical review v1. **B100×10 cumulative W/history sequential**. 첫 표는 각 arm의 실제 최종 W10 하나에서 같은1,000 requests 전체를 평가한 값이며, current B100 또는 online-at-write 합계가 아니다. 5개 요청 arm과 별도 Official reference를 모두 유지한다. Qwen BLUE는 원본 모델별 config 미지원으로 미실행이며 이 Llama 분모에 넣지 않는다.')
 t(headline(final),['arm','state','RS','PS','NS','rewrite_acc','rephrase_acc'])
 add('RS=1,000 rewrite prompts, PS=2,000 rephrase prompts, NS=10,000 neighborhood prompts/arm. Acc는 teacher-forced all-target-token prompt accuracy이다. L4 사전 CPU/smoke/fidelity 검증은 **SKIPPED_USER_DIRECTED**이며 사후 무결성 검산으로 이를 PASS로 바꾸지 않는다.')
 add('## 1. 주요 수치와 비교 경계')
 for a,b in [('BLUE_L4_ONLY','BLUE'),('BLUE_L8_ONLY','BLUE'),('BLUE','JVP'),('BLUE_L8_ONLY','JVP_L8'),('JVP_L8','JVP')]:
  ds=[100*(float(f[a][m+'_rate'])-float(f[b][m+'_rate'])) for m in ['RS','PS','NS']]
  add(f'{a} − {b}: RS {ds[0]:+.2f}pp, PS {ds[1]:+.2f}pp, NS {ds[2]:+.2f}pp. 동일 request inventory의 서로 다른 최종 endpoint를 비교한 산술 차이다.')
 add('원본 BLUE first+last [4,8]은 L4-only 대비 이 최종 표에서 세 primary 점수가 모두 높지는 않다. L4-only와 L8-only는 editable layer뿐 아니라 z 최적화·관측 위치도 다르다. BLUE와 JVP는 sample은 같지만 target policy, contexts, seed, tokenizer writer 설정, controller와 환경이 달라 method-only 인과효과로 분리할 수 없다. BLUE 성공이 기존 JV 실패의 원인을 입증하지 않는다. 본 사용자 instruction은 상세 비교 해석을 허용하지만 보고는 관측과 산술 차이에 한정하며 scientific_promotion=false이다.')
 add('## 2. 지표와 읽는 법')
 glossary=[
 {'지표':'RS / PS','정의':'각 rewrite / rephrase prompt에서 new NLL < true NLL','분모/단위':'request당1 / 2 prompt; 높을수록 성공률 큼'},
 {'지표':'NS','정의':'각 neighborhood prompt에서 true NLL < new NLL','분모/단위':'request당10 prompt; 높을수록 성공률 큼; tie 실패'},
 {'지표':'strict PS','정의':'동일 request의 rephrase2개 모두 canonical preference 성공','분모/단위':'request; teacher-forced exact와 다름'},
 {'지표':'new/true NLL','정의':'−mean(target token log probability), teacher-forced','분모/단위':'nat/token; 각 target likelihood는 낮을수록 높음'},
 {'지표':'margin','정의':'RS/PS=true−new NLL, NS=new−true NLL','분모/단위':'nat/token; 양수일 때 canonical 성공'},
 {'지표':'rewrite/rephrase acc','정의':'해당 target 모든 token의 top-1 정답인 prompt 비율','분모/단위':'prompt; free generation 아님; new/true 분리'},
 {'지표':'token acc','정의':'정답 target tokens / 전체 target tokens','분모/단위':'token; prompt accuracy와 혼합하지 않음'},
 {'지표':'mean/median/IQR/p90/max','정의':'동일 category prompt별 스칼라의 평균/중앙/중간50%/90백분위/최댓값','분모/단위':'prompt; request cluster 독립성을 가정한 유의성검정 없음'},
 {'지표':'Layer-wise Update Magnitude','정의':'실제 저장 FP32 W_exit−W_entry의 Frobenius norm','분모/단위':'layer×joint B100; request100번으로 복제하지 않음'},
 {'지표':'magnitude share','정의':'각 layer norm / 모든 selected layer norm의 합','분모/단위':'batch; squared share와 구분'},
 {'지표':'JV residual e / V','정의':'N0 entry scales로 정규화한 fixed-L8 target residual / ½||e||²','분모/단위':'JV 정의; BLUE native layer-local residual과 직접 동치 비교불가'},
 {'지표':'JV native work','정의':'history+L2 metric의 source quadratic, qref로 정규화','분모/단위':'Frobenius 아님; BLUE에서는 미기록'},
 {'지표':'at-write loss / recovery','정의':'처음 성공→나중 실패 / 처음 실패→나중 성공','분모/단위':'동일 request identity; acquisition 실패와 forgetting 분리'}]
 t(glossary,['지표','정의','분모/단위'])
 add('NA / NOT_AVAILABLE는 미기록 또는 이 실행에서 확인할 수 없는 항목이며 0이 아니다. PS는 request별 평균 NLL의 선호 비교가 아니라 개별 prompt 비교다. Marginal NLL quantile 차이로 margin quantile을 만들지 않았다.')
 add('## 3. Source/config/계산 의미와 provenance')
 add('BLUE 원본 HEAD311b076a92e4ed0f14f5c8b4909732da781bc5f7/tree f3c933c31cba2fe979c5c34546a99a72e6beb763. BLUE helper HEAD1075540b45c29269e690ac63aae44758d8d63174/tree172b6b9b5c0de4e3aa9e94a05920edaa84a2b323. L4/L8 adapter는 사용자 지시대로 local-only archive를 보존했고 tracked runtime으로 옮기지 않았다. 실행·분석 source는 서로 다른 identity다.')
 t(compat,['arm','job','source_head','source_tree','layers','target_policy','seed','context_hash','config_sha','archive_sha'])
 add('**실제 Llama 설정 확인:** pinned JVP/O YAML도 v_weight_decay=0.5, L2=1이다. Qwen의0.001을 Llama에 복사해 차이라고 서술하지 않는다. BLUE variants는 layers 외 optimizer 기본값 동일: v_num_grad_steps25, v_lr0.1, clamp0.75, loss layer31, decay0.5, KL0.0625, nullspace threshold0.02. Native optimizer의 early loss stop은 원본 그대로다. Full config·tokenizer·asset SHA는 source-config-compatibility.csv 및 source-asset-inventory.csv에 있다.')
 t(compat,['arm','model_revision','z_decay','L2','v_num_grad_steps','v_loss_layer','dtype','attention','tf32_matmul','tf32_cudnn','gpu','torch','transformers'])
 add('공통 Llama revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, model parameter8,030,261,248 FP32. BLUE는 transformers4.44.2/torch2.9.1+cu128 및 eager attention; original README-era 의존성 대신 Llama3/tuple-hook 지원 환경을 쓰는 portability 차이를 봉인했다. Writer tokenizer add_bos=False/right/pad=eos, evaluator는 별도 default tokenizer/right. JVP writer/evaluator는 같은 default tokenizer이며 use_cache=False; BLUE는 model config 기본값을 유지한다. TF32 matmul은 모두False, BLUE cudnn=True와 JVPFalse는 차이로 남긴다. dtype가 같다는 이유로 cross-host bitwise parity를 주장하지 않는다.')
 add('### 알고리즘 차이: L8 one-shot ≠ JVP-L8')
 mathrows=[
 {'arm':'BLUE','target':'각 L4→L8 진입 current W에서 z_l*=compute_z(W,l)','write':'[P_l(KKᵀ+M_l)+L2 I] X_l=P_l K (z_l*−h_l)ᵀ; ΔW_l=orient(X_l)','history':'선택2 layer에 native final key pass, batch append1'},
 {'arm':'BLUE_L4_ONLY','target':'batch entry의 native L4 z*','write':'같은 closed-form, L4 1solve/batch; full local residual','history':'L4 history1; other layer update0'},
 {'arm':'BLUE_L8_ONLY','target':'batch entry의 native L8 z*','write':'같은 closed-form, L8 1solve/batch; full local residual','history':'L8 history1; other layer update0'},
 {'arm':'JVP','target':'batch entry L8 z* 고정, N0 scale 고정','write':'N=4,h=.5,T=2; refreshed native directions+JVP; c≥0 response NNLS; ΔW=Σnode h c_l D_l/√q_l','history':'원본5 layer finalization/history, terminal materialization commit'},
 {'arm':'JVP_L8','target':'JVP와 같은 fixed-L8 policy','write':'full5 entry qref와 dictionary, L8 support만 적용; 4 refreshed steps와 JVP','history':'5 layer history inventory 유지; nonL8 physical write0'},
 {'arm':'O_NATIVE','target':'entry fixed L8 z*','write':'stock AlphaEdit ordered5 layer/remaining-layer allocation','history':'stock5 layer history'}]
 t(mathrows,['arm','target','write','history'])
 add('JVP node의 L8 비중이 높더라도 coefficient 선택, native metric 정규화, repeated steps/current keys, target-context, history inventory가 달라 one-shot과 같은 연산이 아니다. Source expression별 line/SHA는 source-findings.csv에 제공한다. 다른 실험의 stopped ORBODE correctness 감사는 재개하지 않았다.')
 add('## 4. 무결성, 분모, 저장 상태')
 add('단발 scheduler:38940 COMPLETED0 01:06:20;38988 COMPLETED0 00:58:30;38997 COMPLETED0 01:01:01. 세 task 모두10/10 B100와1000/1000 requests, 27/27 interbatch W/M hash links,30 history append,9 selected-W/dense-M checkpoints(1/5/10)의 CPU reload/hash 확인. Terminal flags W0/cache bytes·pointer restore True, nonfinite/failure0. Version counter는 copy_ 때문에 복원되지 않으며 source도 NOT_CLAIMED로 명시한다. Evaluation 동안 before/after signature exact를 별도로 검증한다.')
 add('Raw/publication members478개를 SHA/size로 재해시했다. 별도 source/asset/dependency1,804 unique members도 재해시했다. O/JVP는 Server2 Git publication 재해시 수준이며 원격 raw/checkpoint를 검증했다고 하지 않는다. Local JVP-L8는 기존 inventory의 Llama chain raw를 재해시하고 최종 NLL pair를 독립 집계했다. BLUE finalW10은 checkpoint W10과 same endpoint로 확인했으며 terminal 후 W0 복원된 모델을 final로 평가한 것이 아니다.')
 add('**L4 기록 오류:** native-observation의 projector_asset_index=4는 observer 상수 오류다. execution.lock와 runtime은 allp[[0]]이고, 원본 P tensor에서 index0을 CPU로 추출한 SHA가 실제 runtime projector_sha256와 정확히 일치한다. 따라서 실제 P 선택은 L4용0이며 이 field의 provenance만 잘못됐다. 기존 source/raw는 변경하지 않았다. 사전 검증 생략 사실은 그대로 유지한다.')
 t(load('checkpoint-integrity'),['arm','batch','weights','cache_shape','status','sha256'])
 add('Checkpoint는 선택 weight와 실제 dense cache 및 context/request metadata이며 full-model standalone 저장본이 아니다. Pinned immutable base model과 결합해야 복원 가능하다. 단일-layer main의 비편집 weight 검사는 pointer/version/shape/dtype이며 full bytes는 smoke에서만 수행했다(L4 smoke 미수행). Original BLUE는 source selected-write 경로와 selected state 확인 범위다. 해시 PASS를 전체 model-level observer on/off parity PASS로 확대하지 않는다.')
 add('공통 sample ordered_root='+SAMPLE_ROOT+'. 동일 IDs/order/B100 경계, dataset raw-record/request hash 및 target bytes를 독립 대조했다. BLUE끼리 final prompt identity/order도 정확히 맞는다. BLUE↔JVP의 per-prompt identity scheme에는 token IDs 포함 여부 차이가 있어 이번 cross-family prompt-loss pairing은 하지 않고 publication/request-cohort 범위로 제한했다. Duplicate/imputation/exclusion으로 표본을 바꾸지 않았다.')
 add('### 원본 W0 reference')
 t(load('W0-reference-comparison'),['arm','W0_RS','W0_PS','W0_NS','reference_RS','reference_PS','reference_NS'])
 add('동일 sample에서 W0의 RS71/1000, PS227/2000, NS8820/10000이 기록됐다. 일치하는 aggregate 자체를 환경 bitwise equality로 간주하지 않는다.')
 add('## 5. 최종 NLL/margin 및 secondary accuracy')
 for cat in ['rewrite','rephrase','locality']:
  add('### '+cat+' — final W10 prompt-level 분포')
  t([r for r in dist if r['scope']=='final' and r['category']==cat],['arm','quantity','n','mean','median','p90','max'])
 sec=[]
 for a in ARMS:
  r=f[a];z={'arm':a}
  for cat,tag in [('rewrite','RS'),('rephrase','PS'),('locality','NS')]:
   side='true' if tag=='NS' else 'new';prefix=cat+'_target_'+side
   z[cat+'_prompt_acc']=f"{r[prefix+'_all_tokens_correct_count']}/{r[tag+'_den']}"
   z[cat+'_token_acc']=f"{r[prefix+'_correct_token_count']}/{r[prefix+'_target_token_denominator']}"
  sec.append(z)
 t(sec,['arm','rewrite_prompt_acc','rewrite_token_acc','rephrase_prompt_acc','rephrase_token_acc','locality_prompt_acc','locality_token_acc'])
 add('Locality acc는 target-true teacher-forced diagnostic이며 NS가 아니다. NLL tail의 극단값은 포함하며 nonfinite를 제외해 성능을 보정하지 않았다. Prompt marginal quantiles와 paired margin quantiles는 표의 quantity로 분리한다.')
 add('## 6. Sequential current와 cumulative: 모든 batch')
 add('current는 W_k에서 새 cohort100만, all-seen rewrite는 W_k에서 과거를 포함한100k, full checkpoint는 W1/W5/W10의 전체 seen RS/PS/NS다. Intermediate PS/NS는 미기록이며 보간·새 평가0. Final W10은 seen checkpoint10과 동일 평가 member이므로 별도 표에 재사용해도 독립 측정으로 중복 계산하지 않는다.')
 add('### Full seen-prefix checkpoint 전체18행')
 t(rates(seen),['arm','batch','requests','RS','PS','PS_all2','NS'])
 for a in ARMS:
  add('### '+a+' — current B100와 seen rewrite')
  rs=[]
  for b in range(1,11):
   c=next(x for x in cur if x['arm']==a and int(x['batch'])==b);w=next(x for x in rw if x['arm']==a and int(x['batch'])==b)
   rs.append(dict(B=b,current_RS=f"{c['RS_num']}/{c['RS_den']}",current_PS=f"{c['PS_num']}/{c['PS_den']}",current_NS=f"{c['NS_num']}/{c['NS_den']}",
    seen_RS=f"{w['RS_num']}/{w['RS_den']}",current_new_NLL_mean=c['rewrite_target_new_nll_mean'],seen_new_NLL_mean=w.get('rewrite_target_new_nll_mean',NA),seen_margin_mean=w.get('rewrite_margin_mean',NA)))
  t(rs,['B','current_RS','current_PS','current_NS','seen_RS','current_new_NLL_mean','seen_new_NLL_mean','seen_margin_mean'])
  b10=next(x for x in cur if x['arm']==a and int(x['batch'])==10)
  add(f"{a} B10 신규 cohort RS={b10['RS_num']}/100, PS={b10['PS_num']}/200, NS={b10['NS_num']}/1000. 최종 전체 RS={f[a]['RS_num']}/1000과 다른 분모다.")
 add('### 각 checkpoint의 상세 분포')
 for a in ARMS:
  add('#### '+a+' seen-prefix NLL / success-oriented margin')
  t([r for r in dist if r['scope']=='seen' and r['arm']==a],['batch','category','quantity','n','mean','median','p90','max'])
 add('## 7. Acquisition 실패, 이후 forgetting/recovery, cohort retention')
 add('### Online own-batch 합계와 final W10: 동일 request, 다른 평가 state')
 t(load('online-final-comparison'),['arm','RS_num','RS_den','RS_final_num','RS_final_minus_online_pp','PS_num','PS_den','PS_final_num','PS_final_minus_online_pp','NS_num','NS_den','NS_final_num','NS_final_minus_online_pp'])
 add('online 열은 각 cohort 편집 직후 서로 다른 W에서 평가한 값의 합이다. final 열은 단일 W10이다. Δ는 final−online percentage point이며 최종 headline은 항상 final이다.')
 t(parts,['arm','all_denominator','at_write_success','initially_failed','at_write_success_to_final_failure','initially_failed_to_final_recovery','final_RS','overwrite_candidates'])
 for a in ARMS:
  p=next(x for x in parts if x['arm']==a)
  add(f"{a}: 처음 실패 {p['initially_failed']}/1000, 처음 성공 후 final 실패 {p['at_write_success_to_final_failure']}/{p['at_write_success']}, 처음 실패 후 final 회복 {p['initially_failed_to_final_recovery']}/{p['initially_failed']}. 처음 성공−소실+회복={p['final_RS']}/1000.")
 add('JVP의75개 at-write 실패와 JVP-L8의78개 at-write 실패는 B10 current에서 발생했다. 이후 소실2/4건과 합쳐 최종실패77/82건이다. BLUE/L8 one-shot은 online RS1000, L4 one-shot은999로 측정됐다. 이 차이는 신규 acquisition과 이전 edit 소실을 분리해 보아야 하며 tiny residual 등 미기록 원인을 확정하지 않는다.')
 add('### Final W10의 edit-age cohort: 각 cohort 분모100')
 t(load('final_age_cohort'),['arm','cohort','age','current_success','at_write_success','initially_failed','at_write_success_now_failure','prior_failure_now_recovery','margin_mean'])
 add('cohort1이 가장 오래된 편집,cohort10이 가장 최근 편집이다. 모든 W_k×cohort 삼각행렬330행은 retention_cohort.csv와 heatmap에 완전 제공한다. 각 BLUE request group/target hash에서 overwrite candidate1개가 있으며 같은 raw 표본에 유지했다. 의도된 overwrite의 인과적 확정이 아니라 metadata candidate다. Candidate를 빼서 보고 점수를 올리지 않았다.')
 add('### 동일 prompt의 loss/recovery — 확인 가능한 BLUE pair만')
 t(trans,['arm','reference','metric','batch','denominator','reference_success','arm_success','success_to_loss','failure_to_recovery','stable_success','stable_failure'])
 add('NS 총점이 같거나 비슷해도 loss와 recovery가 상쇄될 수 있다. 이 표는 각 exact hash prompt pair의 변화이며 prompt들을 독립 관측으로 간주한 p-value나 우월성 주장을 하지 않는다. JVP/O의 기존 NS transition summary는 reference_neighborhood_transition_summary.csv에 original scope로 보존한다.')
 add('### BLUE own-at-write → final: 동일 prompt의 변화')
 t(load('atwrite-final-prompt-transitions'),['arm','metric','denominator','reference_success','arm_success','success_to_loss','failure_to_recovery','stable_success','stable_failure'])
 add('## 8. 실제 layer update, controller와 residual 측정 범위')
 action_summary=[]
 for a in ARMS:
  for layer in range(4,9):
   rs=[r for r in actions if r['arm']==a and int(r['layer'])==layer]
   if not rs:continue
   action_summary.append(dict(arm=a,layer=layer,batches=len(rs),mean_magnitude=np.mean([float(r['batch_net_norm']) for r in rs]),
    sum_batch_magnitude=sum(float(r['batch_net_norm']) for r in rs),mean_share=np.mean([float(r['batch_net_magnitude_share']) for r in rs]),
    B1_norm=rs[0]['batch_net_norm'],B10_norm=rs[-1]['batch_net_norm']))
 t(action_summary,['arm','layer','batches','mean_magnitude','sum_batch_magnitude','mean_share','B1_norm','B10_norm'])
 add('Magnitude는 실제 batch entry→endpoint FP32 차이이고,10개 batch norm 합은 전체 W0→W10 net norm이 아니다. BLUE의 각 selected layer는 batch당1write이나 별도 substep path telemetry는 없으므로 JVP node path work와 동일 척도로 합치지 않는다. 비선택 layer update0은 known support다. BLUE native history/L2 action은 미기록이며 Frobenius를 native라고 이름 붙이지 않았다. BLUE layer-local target residual과 JVP fixed-L8 normalized V는 target/time/정규화가 달라 직접 pooling하지 않는다.')
 add('JVP/JVP-L8의 source-sealed node mechanism, batch mechanism, normalization을 각각 node_mechanism.csv, batch_mechanism.csv, normalization.csv로 보존했다. 이들 수치가 없는 BLUE 행을0으로 생성하지 않았다. 새로운 공통 activation forward, residual 재측정, mechanism replay는 하지 않았다. 성능 차이와 update norm·target/config·시간 차이가 동반된 사실만 제시하며 원인 분리는 하지 않는다.')
 add('## 9. 계산량과 시간: 서로 다른 source/hardware')
 t(cost,['arm','process_seconds','scheduler_elapsed_seconds','allocated_gpu_hours','target_seconds','edit_seconds','evaluation_seconds','key_seconds','compute_z','solve_instrumented','source_expected_solves','main_JVP','peak_gpu_bytes','local_storage_bytes'])
 add('BLUE original은 native compute_z2000 calls/1000 edits(두 selected layer), L4/L8는각1000 calls. z call은 optimizer iteration과 다르며 내부25-step maximum/원본 early stop은 그대로다. Single-layer solve counter10, original BLUE solve20은 source의1/layer/batch에서 정해지는 count로 별도 표시하며 measured counter로 위장하지 않았다. Original key calls40,single-layer20. History append는각10batch commit이다.')
 add('BLUE edit_seconds에는 target/key/native solve/finalization 및 observer overhead가 들어 있다. target_seconds와key_seconds는 그 부분집합이므로 total에 다시 더하지 않는다. 기타 edit 시간을 solver-only라고 부르지 않는다. Process에는 load,W0 evaluation,checkpoint/hash와terminal restore도 포함된다. Solver/history/materialization-only 시간 및 전체 forward/backward수는 BLUE schema 미기록으로 NA다. 실제 GPU utilization 미기록; allocated GPU-hours는 Slurm dedicated1GPU elapsed다.')
 add('O/JVP는 Server2 RTX A6000, BLUE/JVP-L8는 Server4 RTX PRO6000. JVP200 main JVP calls, JVP-L840 calls를 publication에서 재사용했다. Total solve counter는 auxiliary/observer 경로를 포함할 수 있어 원본BLUE20closed-form과 같은 정의라고 단정하지 않는다. Full detailed recorded counters와batch timing은 compute.csv/compute_by_batch.csv, 기존 서브단계 정의를 그대로 보존했다. Cross-hardware wall-time ratio는 통제된 algorithm speedup이 아니다.')
 add('### 기술 실패 lineage(과학 분모0)')
 t(load('blue-technical-exclusions'),['job','classification','denominator','prior_batches','stage','error','process_seconds','scheduler_gpu_elapsed'])
 add('38929는 tokenizer optional attribute,38932는 evaluator import binding 실패로 편집 전 종료됐다. 해당 원본 실패 root/log는 불변. 이번에는 기술 실패의 scheduler를 재조회하지 않았고 process seconds만 파일에 기록된 값으로 제공한다. 이전 JVP/L8 technical exclusions는 reference_failure_registry.csv 및 historical report에서 별도 유지한다. 새 GPU/rescue/retry0.')
 add('## 10. Figures / 재현 / 누락')
 for fig in read(out/'figures/plot-manifest.json'):
  add(f"### {fig['path']}\n\n![{fig['path']}](figures/{fig['path']})\n\n{fig['caption']}\n\n출력 SHA `{fig['sha256']}`. Missing 처리: {fig['missing']}.")
 add('Figure는 repository Python+Agg backend/고정style·DPI160·seed0로 생성했다. 입력 CSV SHA/source SHA/실행 명령은 figures/plot-manifest.json에 있다. 재생성 PNG byte-stability 검사는 focused-tests.json에 결속한다. Codex visualization/imagegen/manual edit0. Weight figure title은 정확히 Layer-wise Update Magnitude이며 균등분배선/하단 bars 문구가 없다.')
 t(load('availability'),['item','status'])
 add('## 11. FACT / 제한 / 종료')
 add('FACT:5개 requested arms와Official reference의 final actualW10×전체1000을 비교했다. BLUE3는 정상 scheduler exit/terminal 및 raw 재해시,selected-W/M CPU hash 연결을 확인했다. L4의 P index receipt 오류는 source·actual P tensor hash 대조로 실제0/label4로 구분했다. L4 사전 검증 생략은 유지한다. 모든 낮은 성능과 at-write 실패를 분모에 포함했다.')
 add('제한: 단일1000stream, cross-repository/context/tokenizer/backend/controller 차이, 일부 비용·공통residual 미기록, O/JVP remote raw 부재. 같은 aggregate나source 검사로 model-level observer parity/causal equivalence를 확정하지 않는다. Qwen 미지원은 이번Llama성능으로 대신하지 않는다. Scientific promotion=false. 신규 실험·후속모니터링 없음; report/main통합 후 TASK_COMPLETE_STOP.')
 add('## 12. Artifact inventory와 실행 명령')
 add('Local raw roots와각source/assetmember는 raw-member-inventory.csv/source-asset-inventory.csv에 절대경로·size·SHA로 결속했다. Raw model/weights/cache/checkpoint/prompts/log는 Git에 포함하지 않는다. BLUE original/source 및 local L4/L8 archive는 그대로 보존하며 이 package에는 분석 코드만 새로 추가했다. 기존 O/JV publication0d0a0131과 JVP-L8 report SHA a7b07d16ea36e0aefe7aa16fb2c259275ee1879d486f084a469f9f94dd414732를 immutable reference로 쓴다.')
 add('재현 순서(새 빈 output namespace에서): metrics → audit → details → provenance → plots → report → tests/package. 모델이나evaluator를 import/실행하지 않는다. 분석 source 경로: project/run_scripts/blue_fivearm_analysis/. 결과는 create-once 파일을 사용한다. 전체 SHA/root는 analysis-manifest.json 및 rooted-receipt.json에 기록한다.')
 inv=[]
 for p in sorted(out.rglob('*')):
  if p.is_file() and p.name not in ['factual-report-ko.md','analysis-manifest.json','rooted-receipt.json']:inv.append(dict(file=str(p.relative_to(out)),rows=len(csvread(p)) if p.suffix=='.csv' else NA,bytes=p.stat().st_size,sha256=sha(p)))
 t(inv,['file','rows','bytes','sha256'])
 (out/'factual-report-ko.md').write_text('\n'.join(lines))
 if not (out/'layer-action-summary.csv').exists():csvwrite(out/'layer-action-summary.csv',action_summary)
 print('REPORT_WRITTEN',sha(out/'factual-report-ko.md'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);run(p.parse_args().repo)
