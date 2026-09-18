# GH → SH4: Single-layer edit-preserving correction v1 — M 재사용 우선 / M만 제출

instruction_id: ODEEDIT-S06-SINGLE-LAYER-EDIT-PRESERVING-CORRECTION-M-SH4-V1
nonce: ODEEDIT-GH-SH4-ENFC-M-REUSE-INITIAL-GATE-20260918-R1
source_session: 01a04939-8873-7673-8dca-4c7fc5e31af0
target_session: 01a04939-b5c7-7a03-ba2d-ef3343d62cfd
target_server: server4
expected_cwd: /data/janghj/ODE-edit
repository: hyunjun1127/ODE-edit

## 1. 최신 사용자 결정과 우선순위

이번 task는 새 single-layer EN-F 구현과 실제 기술 검증, M 기존 증거 감사 및 누락분 제출이다. 이전 local-z/repair/EP-TW를 재개하는 요청이 아니다. Server4 합계 GPU cap=2.

최신 사용자 원문:
> 우선 M만 제출하고 본실험 초기 gate 확인 후 모니터링 중지 하는 것으로 수정하자

이는 바로 앞 “설계의 단계별 확대·중단 gate에 따라 최종 보고까지 계속” 지시를 **대체**한다. **S/R/L 제출 및 자동 확대 권한은 이번에 없다.** 모든 stage cells를 한꺼번에 등록하지 않는다. 원 설계와 전체 cells는 연구 사양으로 보존하되 execution allowlist는 필요한 T와 M의 누락 실행/평가로 제한한다. M 결과의 과학 gate는 본실험 초기 실행 gate와 다르며, 상세 완료 분석/후속 단계 판단은 사용자 recall 후다. 이번 compact 인계에서 M 전체 완료나 최종 claim을 주장하지 않는다.

우선순위:
1. 최신 M-only / actual main initial gate pause / cap2 사용자 결정.
2. 최초 사용자 원문 전체의 M 재사용 지침(동폴더 user-instruction-ko.md). 설계 §2의 일괄 N4 재실행 및 §9 M 일괄 실행보다 우선한다.
3. 원 design/contract/cells의 변경 없는 방법·수치·평가·저장·기술 계약.
4. 이전 task 운영정책. 과거 no-checkpoint, derivative skip, mean-loss plateau, cap sweep 또는 multi-layer 정책을 이번 task에 상속하지 않는다.

원 shared checkout와 사용자 dirty, 기존 sealed source/raw/teacher/jobs를 보존한다. 최신 main에서 새 clean codex/server4-single-layer-edit-preserving-correction-m-v1 branch/worktree를 만들고 아래 신규 namespace만 사용한다. 기존 paused/STOP task 자동 재개0.

## 2. FULL_READ / 정본 전달 / 구현 상태

원문 전체 및 아래 정본을 읽고 local authoritative에 create-once 봉인한다. GH 원경로와 repository 상대경로 mapping은 source-input-manifest.json을 사용한다. GH /mnt 경로가 S4에 있다고 가정하지 않는다.

- plans/global/2026-09-18-single-layer-edit-preserving-correction-design-v1.md SHA d0dc2fcf22c5b4a1f214743d8b8ee37fdb4d3c07a90d3cfc85a014a1befdcc06
- companion contract-v1.json SHA c93b345aac614a87eff31ca0e7172c413ee849e830947a5c627fbcafdb2234d0
- companion cells-v1.csv SHA ff142374491d1f99999be1e938ec4bb592b89b530839cc4d0f6a28601352f41c (원 CRLF bytes 보존)
- dispatch/references/07_single_layer_method_positioning.md SHA 5233db01620ad75cc00d2146488858e20188e3721578d53b8c9d0551bf779522
- dispatch/references/06_blue_l4_detailed_audit.md SHA 719417dbb99da11166dc9b0fb78492aa311426ff6fb5793e1805ac91f42557b8
- audits/global/2026-09-18-single-layer-edit-preserving-correction/의 geometry_reference.py, test_geometry_reference.py, validate_design.py 및 기존 CPU/설계 receipts.
- 본 envelope/사용자 원문/최신 PROTOCOL/현재 등록 session과 자원정책.
- 실제 원 native AlphaEdit_main/compute_z/compute_ks/P mapping/history 및 재사용할 actual SL-ZFlow all-token adapter/transaction/evaluator의 frozen code와 import closure.

외부 두 prose는 byte-exact mirror다. 그 안의 GH 절대 empirical/figure 링크가 S4에 없다는 이유로 source를 바꾸거나 관련 raw를 전부 전송하지 않는다. 필요한 증거는 reuse matrix의 구체적 항목에 맞춰 기존 S4 보존 receipt와 local artifact부터 찾고, 부족한 정확 파일만 owner에게 조회한다.

제공 geometry 코드는 작은 NumPy FP64 CPU 참조이며 production rank estimator/model runner가 아니다. 기존11 toy tests 및 설계 consistency PASS는 실제 Llama T PASS가 아니다. validate_design.py와 test main은 원 결과를 쓰는 동작이 있으므로 정본에 직접 실행하여 덮지 않는다. 필요시 unittest 또는 task-local harness에서 검증하고 새 receipt로 구별한다. main의 실제 SL-ZFlow 실행 source5d149fec 계열을 참고하되 해당 API/모델/수치/입력 경계를 확인한다. writer@K 축약 cache를 복사하여 EN-F 공간을 CA로 제한하지 않는다.

## 3. M 재사용 감사가 첫 산출물 — 실행보다 먼저

먼저 m-reuse-decisions.csv/json과 m-execution-plan.csv를 만든다. M은 O0 first1000의 [0,100),…,[900,1000) **10개 독립 W0/M0 cold batch**다. Sequential B2 이후, warm5k, 다른 arm trajectory endpoint를 cold 결과로 바꾸지 않는다.

각 batch × arm × 진단/평가 항목별로 다음 중 하나를 명시한다:
- REUSE: 조건과 필요한 증거 충족; 해당 fit/optimization/eval 중복 실행0.
- EVAL_ONLY: 동일 유효 endpoint가 있고 평가/진단만 부족; 그 endpoint에서 부족한 observer만 실행.
- RUN_MISSING: 결과 부재 또는 구체적 비교조건 불일치; 부족한 실행 단위만 등록.
- REFERENCE_ONLY: 관련 역사적 근거지만 직접 M 대체 불가.

필수 열: cell/batch/arm/diagnostic ID, status, 원 source/config/receipt/raw/endpoint/target/update 경로·현재가용성·SHA, 비교 조건, 충족/누락 evidence, 구체적 mismatch, 재사용 계산단위, 새 실행/forward/gradient/fit 필요성, shared-native capsule ID, technical-validation 수준, 비용 lineage, 재실행 금지 항목. 디렉터리나 실행 시각만 달라진 것은 mismatch가 아니다.

반드시 확인:
- 모델/revision/tokenizer/physical module/native hparams/dtype/attention/TF32/library와 microbatch.
- 요청 ID/순서/100개 구성/context 실제 tokenization/W0/zero M4.
- 같은 M episode의 native z/Δ_N/W_N/A/K/P/M과 preservation inputs/teacher/loss.
- correction space/optimizer·trial budget/current guards/selector 및 observer 정의/분리.
- 실제 raw 추적성/실행 로그/target/update/endpoint 또는 정확 복원 근거. CP 경로 부재만으로 전체 결과를 RUN_MISSING이라 하지 않고 owner의 migration map/다른 retained artifact/정확 복원 가능성부터 확인한다. 반대로 metadata만으로 미보존 실물 endpoint를 있다고 주장하지 않는다.

재사용은 성능으로 판단하지 않는다. 나쁜 결과/실패/정상 native fallback도 같은 증거 기준으로 보존한다. 기술 실패는 실패 관측 자체를 재사용하되 정상 endpoint 완료로 세지 않는다. 이름이나 평균 RS/PS/NS 일치만으로 동등 실험이라고 하지 않는다. fixed-z L8 screen, EP-TW cap, SL-ZFlow, sequential local-z는 geometry/objective/state가 달라질 수 있으므로 이름을 바꿔 CA/EN-F라고 부르지 않는다.

기존 native capsule이 적합하면 **native fit을 반복하지 않고 같은 W_N에서 누락 arm만 실행**한다. 평가만 부족하면 z fitting/correction optimization0. 한 episode의 arm들에 서로 다른 native reference를 조용히 섞지 않는다. 공통 capsule 미충족으로 재사용할 수 없는 항목은 정확 mismatch를 먼저 기록한다.
M 전체 증거가 충족되면 M 신규 GPU 실행을 전부 생략한다. 새 구현의 T 기술 검증은 별도이며 M 전체 재실행 명분이 아니다. 전부 REUSE이면 판정표/필요 T 증거와 “M_NEW_SUBMISSIONS=0; MAIN_INITIAL_NOT_APPLICABLE_REUSED”를 인계하고 pause한다. 새로운 M을 억지로 만들어 초기 gate를 얻지 않는다.

감사표와 신규 실행 계획을 GH에 중간보고한 뒤 본 승인범위의 구현/준비를 계속한다. GH 중복 raw/GPU 감사나 단계별 재승인을 기다리지 않는다.

## 4. 변경하지 않을 방법과 실제 데이터 계약

Llama-3-8B-Instruct revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, zero-based model.layers.4.mlp.down_proj.weight [4096,14336]만 편집한다. FP32/eval/eager/matmul+cuDNN TF32off, seed20260916, native L2=1/lr.1/decay.5/clamp.75/KL.0625/25loss·24Adam/.05native total early stop 그대로. 모든 새 M episode W0/zeroM4, warm checkpoint0. 기존 native P4 수치 불변. cold7 context capsule을 검증해 공통 사용하고 부재시 설계 허용대로 W0에서 한 번 생성/새identity를 표시하며 과거 byte parity를 주장하지 않는다.

Fixed10k S4 전용 /data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1, 모델 전에 verify/load_prefix. dataset3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1, O0whole5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729, first1k40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd. 임의 shuffle/filter/drop0.

주 EN-F는 native endpoint W_N을 고정하고 full-token old/new rewrite union K_E에 D K_E=0, D P*=D를 만족하는 공간에서 고정 W0 forward KL을 낮춘다. Native essence는 correction lock에 넣지 않는다. canonical 포함 여부와 모든 valid token/prefix provenance를 기록하며 pad 제외/임의EOS추가0. official P/N 또는 새 paraphrase는 lock/학습/guard/selection에 넣지 않는다. 현재 fixed token 반응 보존을 미관측 PS/과거 edit/자유생성 보장으로 확대하지 않는다.

P_raw는 native 그대로; P*는 native allowed range의 직교 projector, provenance 또는 contract의 eigen 검사만 사용한다. Q=V(I−JJ†)Vᵀ/J=VᵀK_E이며 projector 단순 순차곱0. FP32 captured keys를 FP64로 처리하고 tau=max(shape)*eps64*sigma_max, ambiguity [tau/10,10tau] 고정. 성능에 맞춘 rank cutoff 변경0. REPAIR_SPACE_EMPTY/RANK_UNRESOLVED는 설계의 no-correction 결과로 보존하고 잘못된 projector는 기술 실패로 구별한다. native output span 제한/추가 native norm cap/target ball 재투영0.

M의8arm=N4/SCALE/CA/KL-P/EN-S/EN-F/EN-COV/EN-F4, 추가는 **설계된** CA-EXACT geometry 및 RAND± paired diagnostic만이다. RAND±를 chain/후보선택 arm으로 만들지 않는다. CA는 orth(row(A))의 Euclidean weight-space projected gradient이며 EP-TW optimizer 재현이 아니다. EN-COV는 W0 cumulative full-input activation drift이고 outputKL 선택0. EN-F4만4gradient/round6trial 총24, 나머지 gradient1/trial8. 동일 capsule·목적의 initial G는 공유 가능하되 accepted 상태가 달라지면 fresh G를 쓴다.

기본 Polyak L/chi와 .5 backtrack/Armijo1e-4는 contract 그대로, 실제 FP32 이동 p=<G,W_trial−W_current>로 판단한다. 첫 resolved accepted만 선택하고 bad/rejected probe 비용 포함. EN-F4 ideal D_acc64와 actual D를 분리, immutable W_N에서 FP32 materialize; +D/−D rollback0. SCALE은 clipped actual displacement 사용. finite actual no-move/no-resolved-step/native fallback은 분모에 남기며 nonfinite gradient/accepted state, wrong state/OOM은 기술 실패다.

S64/Dev128 기존C4/teacher192 exact identity를 검증 재사용, W0 full-vocab FP32 logp, signed FP64 KL(p0||pW); vocab sum→128 positionsmean→64docmean. BOS+256입력257, scoring[128,256), inputtokens16448/scored8192구별. negative KL clamp0. Dev는 observer, Report256은 이번 M scope에서 **method 평가/공개0**; 후속L계약 유지. 새로운 teacher 대량 생성이나 교체를 기본으로 하지 않는다.

모든 correction은 Current 개별 sequence NLL+1e-4 및 native pair/TFstrict 성공 ID subset을 지킨다. 평균 .05 plateau나 count-only검사로 대체하지 않는다. M은 매번cold이므로 이전episode의 history/Past64를 넘기지 않는다. 이전 실험 mean-E screen을 복사하지 않는다.

## 5. 필요한 T 및 M 제출 전 actual 구현 검증

설계 cold8/N4repeat2 T를 실제 Llama에서 수행한다. 기존 적합 기술 evidence는 정확한 구현·조건 범위를 구별하고 새 EN-F의 미검증 부분을 생략하지 않는다. CPU toy나 과거 derivative skip waiver로 새 T를 PASS 처리하지 않는다.
- 실제 fixed-input key stationarity, native/P binding, full-token projector/null residual.
- direct physical selected-weight leaf gradient와 cached all-token suffix VJP relative≤1e-4.
- EN-F/CA/random signed FD: 고정12scale, 실제 FP32 perturbation/noise/ULP 기록, adjacent resolved2 및1%/signal10 기준. 0방향·smallAD는 정해진 absolute/no-direction 분류이지 좋은 probe만 고른 PASS가 아니다.
- FP32 actual D와 protected full-vocab logits/NLL/strict IDs, teacher/noop/forward parity, nonselected state 및 rollback.
- ceilings FP64Q/DKE1e-10, FP32DKE/leak1e-5, protectedlogitmax1e-3/RMS1e-4,NLL1e-4,IDchanges0,noopNLL1e-5/logit1e-4, KLfloor/resolveddecrease1e-6 그대로. EN-COV floor는 독립 T roundoff10배를 M 전 lock.
- source/config/import/model/tokenizer/teacher/context/P/batch/preparation 및 gradient-state identities.
- actual kernel/cache/projection scalability·GPU/host/disk 비용. Dense full_matrices toy를14336차원 전체 token에 무비판 복사하지 않는다. exact geometry를 유지하는 matrix-free/streaming 구현은 가능하되 approximate sketch를 exact rank로 표시하지 않는다.

최소 technical 구현 오류 수정은 source/attempt를 새로 봉인하고 영향받은 검증만 수행한다. 성능이 나쁘거나 correction공간0이라는 이유로 threshold/예산/방법을 바꾸지 않는다. 실제 계약 모순/자원상 실행 불가능이면 구체적 TECHNICAL_HOLD/RESOURCE_BLOCKED를 보고하고 과학조건 변경 전 멈춘다. 이전 실패자료와 allocation비용 보존.

## 6. 실행·자원 envelope — cap2 / T 후 M만

Slurm ALLOWED: 본task에 필요한 bounded T 및 최소 technical repair, reuse matrix로 정당화한 M EVAL_ONLY/RUN_MISSING만. REUSE 재실행/S/R/L/추가 method/baseline sweep/Late/warm실험 권한0.

T_READY+source/config/input/reuse plan seal 후 필요한 M 실행단위를 **모두 정상 등록·inspection·release**한다. 한 episode의 native 공유를 해치지 않는 bundled runner/array를 쓰고 중복 fit0. cap2 범위에서 두 slot을 활용하되 복제를 만들지 않는다. 기존 작업의 active/admitted pending 동시가능 용량 포함, 다른jobcancel/hold/config변경0. 독립 M 전부 사전등록한 프로그램은 초기 인계 뒤 남은 batch/arm/observer/저장을 자연 진행하되 S/R/L conditional callback은 넣지 않는다.

각 job 기본1GPU/8CPU/mem60416MiB/exportNONE/Requeue0. walltime/단계별 비용 및 peak/disk reserve는 실제 T와 reuse량으로 산정·명시한다. GPUhour hardcap=null(지정 없음), 호출 상한은 시간예산이 아니다. M 최대10sharednativefit/1000target/24000Adam,70optimizationcontroller/100gradient(EN-COV10activation별도)/720trial + RAND±20observer; **이는 전부 새로 실행하라는 수치가 아니라 재사용 전 상한**이다. T/teacher/key/projection/observer/I/O는 별도비용.
GPU여유와 실제 disk·inode·host memory를 admission 직전에 점검한다. 메모리 상한 무단증가0. 본설계 M **모든 final L4 endpoint 보존 의무**가 있으므로 과거 v2 W/M미저장정책을 상속하지 않는다. 큰 raw/tensor는local-only. 기존 타taskCP삭제/재생성0. 저장공간이 부족하면 새 scope의 실측 계획과 해결 가능한 한계를 보고하고 임의 endpoint미저장·평가축소로 우회하지 않는다.

## 7. 초기 gate와 정확한 모니터링 중지 경계

최신 정책은 **본실험 M_INITIAL_VALID 후 중지**다. technical PENDING/T_PASS/source push만으로 종료하지 않는다. 필요한 구현·기술→모든 필요한 M 정상제출까지 진행하며 bounded 관찰한다. 첫 batch 성능을 기준으로 다른 M을 선별하지 않는다.

가능하면 EN-F가 포함된 최초 실제 M episode에서 shared native capsule binding, 실제 projector/gradient/Armijo·guard·invariant 경로, selected endpoint(합법적인 native fallback 포함)의 저장/복원·평가 nonmutation·새 independent episode W0 reset isolation을 확인한다. 설계의 정상0공간/0gradient/NO_RESOLVED_STEP/guard실패를 막기 위해 억지 nonzero correction을 만들지 않는다. 정확히 실행된 경로와 미실행 경로를 분리하고 이를 efficacy PASS로 쓰지 않는다. T 또는 N4 한 행만 PASS해서 EN-F actual science PASS로 확대하지 않는다.
- 신규 correction이면 실제 M 실행의 초기 evidence를 확인.
- 누락분이 EVAL_ONLY뿐이면 실제 endpoint 보완 observer/identity/denominator/state nonmutation gate만 관찰하며 새 optimization을 강제하지 않는다.
- 모든 M이 REUSE이면 신규main없음/initialnotapplicable를 명시하여 reuse-only 인계.

M 초기 gate 후 agent polling/scheduler/log/result query/sleep loop/heartbeat/terminalwait/후속submit/분석·상세보고확장/자동재개0. 이미 제출된 M은 변경 없이 자연진행, hold/cancel0. 모든 arm 첫 batch/전체M완료까지 기다리지 않는다. 최소 source+compact reuse/submission/initial handoff를 own-scope main에 게시하고 WAITING_USER_RESUME으로 종료한다.

본실험 PENDING 예외는 **필요 M 전부 정상release 후, 관찰할 RUNNING main이 없고 실제 GPU 할당 부족이 확인된 경우만** 적용한다. Reason=None/기술pending/Dependency/manualhold/CPU-memory단독문제를 GPU부족으로 추측하지 않는다. 일부M running이면 그 actualinitial 확인까지 진행한다. 자원관측은 정확시각·jobIDs·reason과 확인범위만 보고한다. 별도 daemon/callback/자동resume예약0.

사용자 recall 후 상세 M 통합분석과 M→S 과학 gate를 판단할 자료는 지금 프로그램에 저장한다. **gate가 좋더라도 S/R/L 자동등록 금지**. 장기 설계와 최신실행권한을 분리한다.

## 8. 필수 산출물 / 비교 해석 경계

우선 m-reuse-decisions와 skipped/eval-only/new 실행표를 GH에게 전달한다. 재사용·신규 분모와 공통 계산 비용을 중복계상하지 않는다.
- 원 source/path/SHA/availability/실행·분석 lineage, 필요한 capsule/endpoint/target/key/historybinding.
- token/position key provenance 및 rank/spectrum/q/chi/gradient fraction/CA exact 자유도/output span 사후진단.
- eta/idealactualD/actualArmijo/prediction/목적/guard/invariant/stopreason/trialcache ledger와 모든 실패 비용.
- selection seal 후 official CurrentR/P/N/strict/joint/oldnewNLL/lost-gained IDs/greedy32 originalEOS·censoring. Dev128 및 W0-correct N retention, source raw identity.
- M independent cold10batch라는 통계단위,10kbootstrapseed20260918, batch×document 행렬/두축통계; sequential rows를 독립cold로 만들지 않는다.
- “space 존재/functional 방향/실제 보호/independent locality/W0correct N/PS·과거 관측/비용” 분리. Calibration KL만으로 locality개선0; EN-F 대CA와 KL-P 차이를 선언된범위로 해석한다. 후기claim/과학적 종합은 GH 담당.
- 실제 phase별load/nativefit/key·QR/teacherIO/gradient/acceptedrejectedtrial/guard/observer/checkpoint cost 및 GPUallocated/peak를 분리, 계획상한과실측분리.

초기보고 FULL_READ/M0 → Mreusematrix+미구현/비용계획 → actualT와실측 → Mjobmapping/cap → actualM_INITIAL_VALID 또는정확한pending/blocker/reuseonly. 중간보고는 direct로 전달한다. GH 중복 raw/GPU 감사는 선행조건이 아니다. 복잡한 독립구현에만 필요시 boundedsubagent 사용, 파일소유권 분리/원변경보존/마지막pause전전원정지. 단순검사·보고는 직접한다. Red 관점 source/data leak/numeric/state/resource 검사를 수행하고 독립red사용여부 사실대로기록한다. PNG직접코드생성만, GFM표실제렌더/분모/링크/manifest 검산.

## 9. 허용 read/write/publication/broadcast

Read: 원 설계와 정본/참조source, S4기존 완료결과와receipt/retainedCP/native target·update/context/P/teacher/input. 기존live scientific결과나 stoppedORBODE 및다른taskscope 재개0. 자원admission에는 owner/jobresource만한정확인.
다른server의필요existingartifact는 owner에게 exact path/size/hash/completion상태를 묻고 narrowallowlist로만검토한다. 대량transfers/broadcast/새remoteGPU/원본삭제권한0. 필요 transfer의 구체적owner/파일/byte/수신scope를 먼저GH에보고하여 승인받고 독립local준비는계속한다.

Write:
- project/run_scripts/single_layer_edit_preserving_correction/ (새runtime/adapter/tests/analysis/plot)
- /data/janghj/ODE-edit/local/single-layer-edit-preserving-correction/20260918-v1/ (worktree/authoritative/reuse/T/M/inputs/checkpoints/receipts/raw)
- audits/servers/server4/2026-09-18-single-layer-edit-preserving-correction/
- experiment-reports/servers/server4/single-layer-edit-preserving-correction-2026-09-18-v1/
- messages/acks/server4/2026-09-18-single-layer-edit-preserving-correction.md
- messages/server-heads/server4/2026-09-18-single-layer-edit-preserving-correction.md
- runs/odeedit_single_layer_edit_preserving_correction_s4_20260918/
- tasks/status/odeedit_single_layer_edit_preserving_correction_s4_20260918/server4.json

Ownscope source/tests+compact reuse/submission/initialhandoff만 latestmain과nonforce통합허용; 타SH/GH동시변경보존, conflict정확보고/force0. 원설계/globalauthoritative/sharedPROTOCOL/native/shared환경변경0. Raw/model/teacher/prompt/gradient/fullstdoutGit0. NO_BROADCAST_NOT_REQUIRED가 기본. Sessionhelper stale/path지원부족은실제registry/envelope근거와분리하고허위PASS/sharedhelper수정0.

최초ACK에 **M재사용우선, cap2, M-only, 본실험초기후pause, S/R/L未承認, 새endpoint보존**을 명시하고 작업을 착수하라.
