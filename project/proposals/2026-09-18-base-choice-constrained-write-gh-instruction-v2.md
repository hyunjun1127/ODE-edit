# GH 전달 지시문 — BPCW512 구현·cold B100·조건부 SEQ1000

Instruction ID: `GH-BPCW512-COLD-B100-SEQ1000-20260918-V2`

수신: Global Head(GH). 기본 실행 담당: Server1 SH. 상태: **전달용 지시문 작성 완료; 실제 전달·구현·GPU 제출은 미수행**. 아래 명령은 사용자가 본 지시문을 GH에 전달했을 때의 실행 범위다.

## 1. 수행 목표와 완료 경계

**BPCW-v2를 실제 Llama runtime에 구현하고, 동일 W0에서 N4와 BPCW512의 B100 한 batch를 비교하라. 아래 사전 gate를 통과하면 정책 변경 없이 각 arm의 자기 B1 checkpoint에서 B2–B10을 이어 총1,000개 요청을 처리하고 결과를 보고하라.** Gate 통과 뒤 B2마다 사용자 재승인을 기다리지 말라. Gate 실패면 B1의 유효 endpoint·실패 증거·비용을 정리하고 종료하라. 준비 문서나 submit 계획만 남기지 말고, 실제 실행 상태 또는 해소되지 않은 구체적인 의존성을 보고하라.

방법의 핵심은 `native L4 write → 현재 edit의 full-token response를 잠금 → 전체 reference의 W0 답변 선택을 유지하는 최소 보정`이다. Mean KL 복원, 확률 floor, 원래 margin 복원, layer allocation으로 바꾸지 말라. GSS는 전체 local QP의 처리 순서에만 사용한다.

실행 범위는 **N4/BPCW512 두 scientific arm, arm당 최대10 batches**다. B1 native preview를 공유하므로 native target/write는 B1 한 번, B2–B10은 두 chain에서 각각 계산한다. 따라서 최대19 native batch fits이며, logical arm-batch endpoint는20개다. 기술 점검·reference 준비·observer 계산은 별도 계수한다. GSS-order 비교는 저장된 local 문제의 CPU 분석으로 수행한다. R256, probability floor, 추가 layer, 새로운 order/seed, KL 대조 arm, full10k, 외부 baseline sweep은 추가 제출하지 말라.

이번 지시문의 stage/gate는 아래 BPCW-v2만 따른다. 과거 SL-ZFlow/BG/EN의 `B1에서 모니터링 종료`, `T skip`, `native endpoint 수입`, `single-batch gate 없음` 같은 별도 지시를 자동 승계하지 말라. 새 과학 실행에 필요한 기술 오류는 격리된 attempt에서 수정할 수 있지만, 성능을 보고 정책을 바꾸어 같은 episode를 계속하지 말라.

## 2. 기준 파일과 실제 자산을 먼저 결속하라

다음 파일의 실제 bytes·SHA256을 확인하고 GH/SH 실행 checkout에 함께 전달하라. 현재 검토 workspace의 파일은 미커밋일 수 있으므로 Git pull만으로 확보했다고 가정하지 말라.

- [방법·수식](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-18-base-choice-constrained-write-v2.md)
- [실행 계약 JSON](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-18-base-choice-constrained-write-contract-v2.json)
- [실행 cells](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-18-base-choice-constrained-write-cells-v2.csv)
- [CPU 수학 검증 코드](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-base-choice-constrained-write/check_math.py)와 [결과](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-base-choice-constrained-write/math-checks.json)
- [EN의 수치·native 상수 출처](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-18-single-layer-edit-preserving-correction-contract-v1.json): geometry/native/finite guard만 참조하며, KL 목적·arm 목록·teacher·stage는 승계하지 않는다.
- [프로젝트 운영 규약](/mnt/raid5/janghj/ODE-edit/PROTOCOL.md)

`FRCW-v1` 문서/JSON/cells는 superseded이며 실행 대상으로 사용하지 말라. BPCW 설계와 JSON 사이 실제 충돌이 발견되면 GH가 source-backed 정정을 기록하고 실행 전에 봉인하라. 방법 자체가 바뀌는 수정은 별도 version/W0 episode로 분리한다.

검증된 재사용 출발점은 다음과 같다.

- EN 완료 리뷰 checkout: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-enfc-b001-completed-review-v1`.
- 해당 checkout의 `project/run_scripts/single_layer_edit_preserving_correction/`: `binding.py`, `alltoken.py`, `geometry.py`, `runtime.py`, `sequential_state.py`, observer 등을 source review 후 재사용한다.
- 해당 checkout의 `experiment-reports/servers/server1/enfc-single-batch-m-2026-09-18-v1/completed-review-r1/source-evidence-manifest.json`에서 검증한 이전 execution commit은 `3f1941b21538d6a7ad0afd756774ad6a0b605750`이다. 이것은 **새 BPCW 실행 commit이 아니다**.
- 이전 Server1 B001은 Server4 native를 수입한 경로였다. 이번 matched cold-native 계산·시간의 대체 증거로 쓰지 말라. 과거 `T_SKIPPED`도 이번 numerical PASS로 간주하지 말라.

새 R512의 추가128 입력, W0 answer capsule, BPCW model runner는 현재 미완성이다. 이전 S64 full-distribution teacher는 입력 prefix와 목적이 다르므로 새 capsule을 대신할 수 없다. 실제 부족 자산을 준비하고 source/config/data/runtime manifest를 남겨라.

## 3. GH가 SH에 발행할 실행 envelope

- **구현 소유권:** 새 `project/run_scripts/base_choice_constrained_write/` 아래 reference builder, capsule, margin oracle, constrained solver, controller, runner, tests, launch/reduce 도구를 작성한다. 격리된 `codex/bpcw512-cold-b100-seq1000` branch/worktree를 사용한다. 기존 EN/native 코드는 기본적으로 읽기 전용 dependency로 결속하며 변경이 필요하면 GH가 정확한 추가 파일과 이유를 envelope에 기록한다. 다른 작업자의 변경을 되돌리지 말라.
- **SH 문서 허용 경로:** `plans/updates/server1/bpcw512-20260918-v2/`, `audits/servers/server1/bpcw512-20260918-v2/`, `experiment-reports/servers/server1/bpcw512-20260918-v2/`, `messages/server-heads/server1/bpcw512-20260918-v2.md`, 해당 run/task ID의 metadata. `plans/global/`·`project/proposals/` 수정은 GH 소유다.
- **Raw 경로:** 실행 checkout의 ignored `local/bpcw512/20260918-v2/<episode_id>/`. Raw prompt/generation·factor·checkpoint·전체 로그는 Git에 넣지 않는다. 실제 절대 경로를 receipt에 기록한다.
- **제출 권한:** GH가 구현 source와 preflight를 검토하여 B1 및 아래 gate에 종속된 S10 제출을 한 envelope로 승인한다. SH는 그 안에서 실행한다. 이는 scope 내부 GH→SH 절차이며 사용자에게 같은 실험을 다시 승인받기 위한 대기 단계가 아니다.
- **자원:** 이번 실험의 동시 실행은 기본 GPU1개/job1개로 제한하고 두 arm은 같은 GPU/runtime에서 순차 처리한다. 실제 Server1 project GPU/host-memory cap과 활성 작업은 제출 시 다시 확인한다. 과거 문서의 cap2/cap4·job ID·session ID를 복사하지 말라. `scripts/check-slurm-resource-cap.sh server1 <requested_gpus> <requested_mem_mb>`의 실제 결과, node/GPU model, VRAM, CPU/RAM/disk/wall 요청을 봉인한다. 가용량 부족은 pending으로 두며 다른 job을 중단하지 않는다.
- **세션 결속:** GH dispatch에 현재 Server1 담당 session, CWD, repo identity, branch, instruction ID를 명시한다. 과거 launcher의 하드코딩된 session으로 제출하지 않는다.
- **Preflight:** source/config identity, observer 분리, 전체512 사용, native/geometry 불변, resource cap, state 복원 검사를 통과해야 scientific 결과를 유효로 기록한다. 감사 block은 GH에 증거와 함께 올린다.
- **Artifact 공유:** compact manifest/report를 protocol에 맞게 공유한다. 필요한 일반 프로젝트 artifact만 승인된 `scripts/rsync-artifact-broadcast.sh` 경로로 전달한다. 이미 같은 storage에 있어 전송이 불필요하면 receiver path/checksum과 `NO_BROADCAST_NOT_REQUIRED`를 기록한다. 대형 checkpoint를 모든 서버에 자동 복제하지 않는다.

## 4. Model·native·요청 순서를 고정하라

Model은 `Llama-3-8B-Instruct`, revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`다. 유일한 편집 parameter는 zero-based L4 `model.layers.4.mlp.down_proj.weight`, shape4096×14336이다. FP32 weights, eval mode, eager attention, TF32 off, geometry/선형대수 FP64를 사용하고 다른 weight는 고정한다.

두 arm은 pretrained W0 및 exact-zero M4에서 출발한다. W0 기반 native P4는 유지한다. Warm5000 checkpoint를 사용하지 않는다. Native AlphaEdit L4-only의 source/context/key aggregation은 그대로다. 기본 native 상수는 L2=1, v_lr=.1, v_weight_decay=.5, clamp_norm_factor=.75, kl_factor=.0625, 최대25 loss evaluations/24 Adam steps, total-loss early-stop=.05, target layer4다. `.75`는 기존 z clamp이며 native write를75%로 줄이는 비율이 아니다. 요청당 native local-z routine을 한 번만 호출하고 보정용 추가 z는0이다.

O0 first1000 개발 요청을 순서 그대로 B100×10으로 사용한다. 확인할 기존 identity는 dataset SHA `3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1`, whole-order SHA `5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729`, first1000 root `40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd`다. 실제 hash convention/bytes와 결속되지 않으면 통과로 쓰지 말라. 이 요청들은 이미 노출된 개발 자료이며 unseen confirmatory test로 부르지 않는다.

Canonical/native rewrite 경로만 controller에 준다. 별도 paraphrase set을 생성·학습·replay하지 않는다. Official P/N과 Dev128은 candidate 선택·gradient·stopping에 전달하지 않는다. 미래 요청은 order scheduler만 보유하고 controller에는 현재/과거 요청만 제공한다.

## 5. Reference512와 W0 답변 capsule을 구축하라

빠른 primary의 입력 분포는 **C4 behavior-preservation**이다. 검증된 factual QA 정답 bank라고 부르지 말라. 질문형 Wikipedia reference로 바꾸는 것은 별도 data ID와 후속 비교다.

1. Pinned C4-en revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`의 기존 S64+Reserve320=384문서에 같은 구축 규약으로 새128 train 문서를 추가한다. 기존 Dev128/Report256을 재활용하지 않는다. source/document/window/token hash로 중복·split 분리를 검증한다. 결과를 보기 전 결정적인 source 순서·필터·tie 규칙을 manifest로 봉인한다.
2. 각 봉인된256-token window의 처음128 corpus tokens와 기존 BOS 규약으로 입력을 만든다. BOS 중복 삽입·decode/re-encode 변화를 막고 실제 input IDs를 저장한다. Raw continuation이며 instruction wrapper를 추가하지 않는다.
3. W0에서 raw-logit argmax, do_sample=false, num_beams=1, max_new_tokens=16으로 답변 하나를 저장한다. Logits processor 없이 model EOS IDs와 token-ID tie 규칙을 고정한다. 실제 생성된 EOS도 보호하고, 길이 제한 종료에는 가짜 EOS를 붙이지 않는다.
4. prompt IDs, y0, mask/positions, 각 답변 token의 base logp·margin·top8 IDs/logits, EOS/censor, model/tokenizer/decoding hash를 저장한다. 이후 prefix는 항상 `(x_i,y0_<s)`다. Base top8은 provenance일 뿐 candidate 경쟁자 검색 범위가 아니다.
5. Train512 전부를 매 batch 사용한다. 최대 보호 위치는512×16=8192개다. Nested256은 manifest에만 보관하고 이번 arm에는 사용하지 않는다. Dev128도 별도 capsule을 준비하며 optimizer/GSS에는 전달하지 않는다. Report256은 닫아 둔다.
6. W0의 teacher-forced 전체 위치에서 base token 선택이 재현되는지 확인한다. Generation/teacher-forcing의 shift, padding, position/RoPE, EOS 및 numerical parity를 검증한다. 이 단계 오류를 reference filtering으로 숨기지 않는다.

Base 답변은 행동 anchor다. 이미 틀린 답변일 수 있으며 외부 사실 정답이라고 단정하지 않는다. 현재 편집과 동일 사실을 정반대로 요구하는 직접 충돌은 별도 scope ledger에 기록한다. B1에서 충돌하면 효과를 본 뒤 입력을 바꾸지 말고 `REFERENCE_SCOPE_CONFLICT`로 보고한다. 미래 요청을 훑어 유리한 bank를 선정하지 않는다. S10 중 직접 충돌이 확인되면 immutable bank/의도적 edit의 양립 문제를 기록하고 policy 재정의 전 후속 commit을 중단한다. 계산량이나 낮은 margin 때문에 reference를 제거하지 말라.

## 6. 실제 controller 구현 계약

각 batch에서 자기 entry의 native preview `WN=Wentry+DeltaN`를 한 번 만든다. Current rewrite/canonical old/new teacher-forcing 입력의 **전체 valid-token** key를 K_E로 모은다. Supplied old target이 없는 경우 new 경로를 유지하고 pair guard 제외 분모를 기록한다. Dedup은 실제 prefix/position/key provenance를 보존한다.

Fixed native allowed-space projector가 P*=VVᵀ이면 J=VᵀK_E, `Q=V(I−JJ†)Vᵀ`로 두고 D=DQ를 요구한다. 임의 순서의 두 projector 곱으로 대체하지 말라. Native P_raw 자체를 수정하지 않는다. FP64 rank는 original matrix dimensions를 사용한 `max(shape)*eps64*sigma_max`, 기존 EN ambiguity band [.1tau,10tau] 규약을 봉인한다. 결과를 보고 rank cutoff를 바꾸거나 sketch rank를 exact로 부르지 않는다.

보호 목표는 모든 i,s에서 `m_is(W)=logit_W(y0_is)−max_{v!=y0_is}logit_W(v)>=0` 및 동일 argmax ID다. `d_is=logp_W0(y0_is)−logp_W(y0_is)`는 진단만 한다. 확률 floor·KL·원래 base margin 복원항을 넣지 않는다.

Native의 모든 선택이 이미 유지되면 `NATIVE_ALREADY_FEASIBLE`, D=0으로 끝낸다. Kappa만 부족하다는 이유로 gradient를 실행하지 않는다. 보정이 필요하면 수치 reserve `tau_gap=max(1e−4,10*max_repeat_gap_variation)`, `kappa_is=min(base_margin_is,tau_gap)`를 성능 관측 전에 고정한다. tau_gap>1e−3은 numerical issue로 처리한다.

이상적 문제는 `min 0.5||D||_F², D=DQ, 모든 reference 위치의 선택 보존`이다. 구현은 최대2회 local QP와 실제 endpoint 검사로 근사한다.

1. 전체512 입력의 모든 보호 위치/full vocabulary를 검사하고 reference별 최소 `m−kappa`의 position/competitor pair를 선택한다. Initial row는512개다.
2. Batch 안에서 발견한 pair ID `(reference,position,competitor)`는 유지한다. Round2에서는 각 reference의 새 worst pair를 추가·dedup하여 최대1024행으로 만든다. **이전 pair를 새 pair로 교체하지 않는다.** 모든 누적 pair gradient는 현재 center에서 다시 구한다.
3. Actual center `Wk=WN+Dk_actual`에서 pair별 `mu=logit_y−logit_v−kappa`, raw gradient g, h=gQ를 구한다. 최종 total correction D에 대한 RHS는 **`b=−mu+<g,Dk_actual>`**다. FP32 center의 off-Q 오차 때문에 RHS에는 raw g를 쓴다. Increment로 혼동하지 말라.
4. Local primal은 `min 0.5||D||², <h_j,D> >= b_j`. Gram G_ij=<h_i,h_j>, dual은 `min_alpha>=0 0.5 alpha^T G alpha−b^T alpha`, reconstruction은 **`D=+sum_j alpha_j h_j`**다. 임의 Gram ridge·gradient 평균·global clipping·z-ball 투영으로 문제를 바꾸지 않는다.
5. QP의 KKT는 `alpha>=0`, `G alpha−b>=0`, complementary slackness다. FP64 solver tolerance/iteration cap/zero-norm 처리와 실제 primal·dual·gap residual을 CPU 검증에서 결정하여 결과를 보기 전 lock에 남긴다. 라이브러리 success flag만으로 통과시키지 않는다. h=0,b>0 및 증명된 local infeasible, 단순 optimizer failure를 구별한다.
6. Immutable WN에서 `FP32(WN+D_acc64)`를 materialize한다. 실제 차이는 `FP64(Wcandidate)−FP64(WN)`로 측정한다. Add/subtract rollback을 쓰지 않는다. Actual candidate에서 전체512/모든 위치/full vocabulary를 다시 확인한다.
7. Reference 선택을 통과한 첫 후보에만 current/past full guard를 적용한다. Guard 실패면 추가 candidate 탐색 없이 native로 복귀한다. Reference가 실패하면 한 번만 재선형화하고 누적 경계를 갱신한다. 두 번 안에 못 풀면 reference 실패를 명시한 native fallback이다.

Nominal DK_E 실패·NaN/OOM·state/hash 불일치·잘못된 projector·잘못된 gradient는 technical failure로 중단한다. Finite 후보의 reference/quality 실패는 정해진 native fallback이다. Numerical rank unresolved/empty correction space는 별도 status로 남긴다. 원문제의 전역 불가능성으로 해석하지 않는다. Exact tie가 해결되지 않으면 `TIE_UNRESOLVED`를 기록하며 token tie 규칙을 바꾸지 않는다.

## 7. GSS와 저비용 계산을 정확히 연결하라

GSS-inspired ordering은 all-reference gradient를 확보한 뒤 적용한다. h가0이 아니면 `a=h/||h||`, `beta=b/||h||`로 **양변을 함께** normalize한다. Raw gradient cosine을 쓰지 말라.

- 초기32행: 가장 큰 positive normalized violation을 반드시 포함하고 나머지는 max-min cosine diversity 순으로 채운다. Tie는 violation, 안정적인 pair ID 순이다.
- Working QP를 푼 뒤 **전체 exposed-row local inequality**를 검사한다. 누락된 위반 중 최악은 반드시 추가하고, 나머지는 diversity 순으로 최대32개씩 추가한다.
- 모든 행이 만족되고 solver optimality가 확인될 때만 반환한다. Working set은 round2 최대1024행까지 증가할 수 있다. 유사 gradient라는 이유로 영구 폐기하지 않는다.
- 동일 local 문제의 full solve/most-violation order와 objective·해·feasibility를 CPU에서 대조한다. GSS가 정확한 full-QP endpoint 자체를 개선한다고 주장하지 않는다. Local exposed-row full QP와 모든 token-vocab 경계를 포함한 비선형 원문제를 구별한다.

Native가 위험할 때 첫 버전은 모든 reference gradient를 구한다. **GSS의 neural backward 절감 주장은 금지한다.** 이를 위한 lazy gradient selection은 이번 scope가 아니다.

경쟁자 탐색은 no-grad full-vocabulary forward다. Pair backward는 같은 위치의 두 LM-head row로 계산한다. Reference prefix 전체에 대한 L4 출력 derivative A를 유지하여 `h=A(QK)^T`를 factor로 보관한다. Dense4096×14336 gradient를 reference마다 저장하지 않는다. Gram은 factor contraction 또는 정확한 Gram-vector product로 계산하고 factor I/O·Gram 비용도 기록한다.

독립 example들의 pair loss 합을 activation에 미분하면 batch 축에서 per-example A를 얻을 수 있다. 같은 reference의 여러 pair를 한 loss로 합친 뒤 각각의 gradient를 복원했다고 쓰지 말라. Round2의 다중 pair는 example 복제 또는 명시적 VJP로 분리한다. 모델 parameter gradient가 아니라 필요한 activation gradient를 구한다.

K와 upstream cache는 fixed-token 입력에서 재사용한다. Q·downstream A·competitor는 현재 state에 맞게 갱신한다. Downstream KV/cache를 다른 candidate에 재사용하지 않는다. 최대 pair-scalar gradient 평가는 round1 512+round2 1024=1536이며 실제 microbatch backward 호출 수와 구별한다. Full512 candidate scan은 최대2회, 최종 full guard는 최대1회다.

## 8. 수치·품질·history를 검사하라

Current guard는 own-native 대비 다음을 적용한다. 평균값으로 개별 실패를 가리지 말라.

|검사|수용 한도|
|---|---:|
|FP64 Q symmetry/idempotence relative error|1e−10|
|Nominal normalized DK_E residual|1e−10|
|Actual FP32 max-token response `||D_actual k||/max(1,||WN k||)`|1e−5|
|Actual correction의 P* 밖 상대 norm|1e−5|
|보호 경로 full-vocab logits max/RMS 차이|1e−3 / 1e−4|
|보호 경로 new/old NLL max absolute 차이|1e−4|
|각 current new NLL 증가|1e−4|
|Native canonical preference/TF-strict 성공 ID 추가 손실|0|

Full logits는 streaming reduction으로 검사할 수 있다. 모든 logits tensor를 영구 저장할 필요는 없다. 이 수치 검사는 bitwise logits equality 주장이 아니다. Unseen PS 보장은 별도다.

Past64는 이미 도착한 active (subject,relation)의 최신 target을 사용한다. 현재 overwrite key를 제외한 뒤 기존 `SHA256('ENFC-v1|past|case_id')` 순서, tie는 case_id로 최대64개를 선택한다. 성공한 사례만 고르지 않는다. 각 new NLL+1e−4, canonical preference/strict 성공 ID 무손실을 own-native 대비 검사한다. B1의 prior 수가0이면 `NOT_APPLICABLE`과 분모0을 기록한다. Past64는 guard이며 replay loss/GSS bank로 합치지 않는다. 과거 edit를 W0의 옛 답변으로 복원하지 않는다.

Native history M4는 최종 commit마다 한 번 append한다. Rejected candidate·observer·restore 때 append하지 않는다. Native fallback도 유효 commit이면 한 번 append한다. W/L4, M4, latest-version registry, RNG, order cursor, reference identity, ledger를 atomic checkpoint로 결속하고 commit ID로 중복 append를 막는다. B2 이후에는 상대 arm의 history/native/z를 수입하지 않는다.

## 9. B1 안에 기술 검증을 통합하라

별도의 큰 audit 실험을 선행시키지 말고 작은 고정 reference4개와 current prefix로 아래를 점검한다. CPU 수학12개 PASS는 실제 모델 numerical PASS의 대체가 아니다.

- 고정 입력의 L4 key stationarity, cached suffix와 physical full model의 일치.
- Two-logit gradient와 direct autograd 일치, competitor tie를 피한 AD–FD 검사.
- FP64 projector/rank·factor gradient/Gram, actual FP32 DK_E·current response.
- Base prefix prediction shift, EOS/censor, full-vocab competitor, token-ID tie.
- 이전 competitor 유지, 초기 안전 reference가 새로 위반되는 경우, offset이 다른 동일 방향 제약, zero gradient, singular/infeasible local 문제.
- Working-set/full local-QP 일치 및 guard/observer/controller 정보 분리.
- Immutable restore, cache invalidation, history commit1회, checkpoint resume와 중복 commit 방지.

Tolerance와 runtime 값은 작은 기술 검증 뒤 science outcome을 보기 전에 봉인한다. Presealed 현재/과거 품질 한도를 완화하지 않는다. GPU/TF32/microbatch가 달라진 과거 key tensor를 exact 동일하다고 추정하지 않는다.

## 10. Cold B100 결과를 봉인하고 S10 gate를 적용하라

Server1에서 새로 계산한 같은 W0/B1 native preview를 N4와 BPCW512가 공유한다. N4는 native endpoint 자체, BPCW512는 보정된 endpoint 또는 계약상 native fallback이다. 양쪽 selection seal 뒤 공식 R100/P200/N1000을 측정한다. 동일 bytes의 endpoint는 평가를 재사용할 수 있으나 identity 증거와 절감 비용을 기록한다.

아래 조건이 **모두** 충족되면 `B1_GATE_PASS`로 하고 두 arm의 S10을 이어라.

1. 기술 검사 및 source/data/state 무결성 PASS.
2. BPCW selected endpoint의 전체 reference token 선택 보존과 current/past guard PASS.
3. N4 대비 RS·PS 성공 ID의 추가 손실0. 성공률 합계가 같아도 ID 교환으로 손실을 숨기지 않는다.
4. N4 대비 NS net 감소0, W0에서 맞았던 N의 손실 수 증가0.
5. Dev128의 base-choice token flip 수가 N4보다 증가하지 않음.
6. Setup/observer를 분리한 standalone editing time이 matched N4의2배 이내.

Nonzero correction·NS 상승·KL 감소는 필수 조건이 아니다. Native가 이미 안전해서 D=0이어도 S10에서 horizon에 따른 activation을 확인한다. Reference 위반을 남긴 native fallback이면2번은 FAIL이다. 효과 조건과 비용 조건은 별도 column으로 기록하고 비용만 초과하면 `B1_BUDGET_FAIL`로 구별한다. 이 gate는 개발 확장 기준이며 통계적 유의성 증명이 아니다.

RS/PS/NS/Dev는 **봉인된 B1 endpoint를 본 뒤의 stage gate에만** 사용한다. 실패를 보고 같은 B1에서 lambda, delta, row 수, reference, step cap을 바꾸어 통과시키지 않는다. B1 gate FAIL이면 S10 두 arm을 모두 제출하지 말고 필요한 B1 평가·분석을 완료한다.

## 11. 통과하면 같은 정책으로 B2–B10을 실행하라

N4는 자기 B1에서, BPCW512도 자기 B1에서 계속한다. 둘 다 W0에서 시작한 chain이며 B2부터 native z/write/history는 각각 계산한다. Controller 정책·reference·solver cap은 고정한다. 정상적인 finite fallback은 기록하고 chain을 계속한다. Technical integrity failure나 직접 scope conflict는 해당 paired episode의 후속 commit을 중단하고 마지막 유효 checkpoint를 남긴다. 정상 fallback을 이유로 반복 재실행·새 방법 전환을 하지 않는다.

- 매 batch: 현재 R/P/N at-write, 현재 N의 entry→selected 변화. Entry observer는 controller와 분리한다.
- 고정 B1 cohort: 이후 각 endpoint에서 R/P/N을 재측정하여 old forgetting을 기록한다.
- B10: seen1000 전체 R1000/P2000/N10000과 각 사례 at-write→final retention을 계산한다.
- Reference512: 매 native/candidate/selected 상태에서 전체 보호 위치의 token 선택·margin·d를 기록한다. 같은 bytes 상태는 결과 재사용을 명시한다.
- Dev128: B1/B5/B10. Direct greedy는 미리 hash로 고른 reference8개에서 B1/B10 확인한다. Main 전체-bank 지표는 teacher-forced choice 보존이며8개 generation을512개 generation 측정으로 보고하지 않는다.
- Report256은 계속 미개봉이다. EOS 없이16-token에 도달한 reference는 prefix 보존만 주장한다.

매 candidate의 P/N을 측정하거나 가장 잘 나온 checkpoint를 사후 final로 고르지 말라. Normal completion은 arm당 ordinal1000이며 누락 요청/중복 요청/overwrite 분모를 별도 기록한다. Loss/gain ID와 N true/new-target NLL을 함께 보존하여 net NS만으로 locality 원인을 단정하지 않는다.

## 12. 비용과 필수 산출물

Standalone method 비용은 own native z/write, current-key geometry, native reference scan, pair-gradient suffix, factor/Gram/QP, candidate materialization/full512 scan, current/past guards, commit을 포함한다. Base capsule 구축은 setup, 공식 R/P/N·Dev·추가 greedy는 evaluation으로 분리한다. B1에서 native를 공유했어도 두 arm의 standalone 비교에는 해당 native 비용을 각각 포함하고 실제 전체 연구 비용에서는 한 번만 센다. CUDA 동기화·timer 포함관계·I/O와 CPU 대기를 명시한다. Rejected/fallback 계산도 비용에서 빼지 않는다.

필수 SH 산출물은 `experiment-reports/servers/server1/bpcw512-20260918-v2/`에 compact 형태로 남긴다.

|파일|내용|
|---|---|
|`execution-manifest.json`|instruction/episode/source/config/model/data/reference/teacher/runtime hash, 실제 절대 raw 경로·job ID|
|`preflight.json`|항목별 측정값·한도·PASS/FAIL, cap/session/source audit|
|`b1-gate.json`|위6개 조건의 실제 값, 양 arm endpoint SHA, 확대 여부|
|`batch-metrics.csv`|arm/batch/entry/native/selected R/P/N와 분모, loss/gain, norm, q/rank|
|`reference-summary.csv`|512분모 sequence 보존, token flips/분모, EOS/censor, margin/d, top8 밖 competitor|
|`controller-ledger.jsonl`|pair IDs/state hashes, row 수, GSS 순서, primal/dual/KKT, candidate/fallback reason, guard|
|`compute.csv`|setup/native/geometry/forward/backward/QP/guard/eval/commit, tokens/calls, peak CPU/GPU memory|
|`qp-order-audit.json`|동일 local 문제의 GSS/full/violation-order 해·목적·feasibility·solver 시간 비교|
|`artifact-index.json`|native/selected checkpoint·factor·raw metrics의 path/size/hash/retention, 공유 확인|
|`terminal.json` 및 `report-ko.md`|실제 완료 범위·실패 상태·resume 위치·사실과 수치|

Ledger가 커지면 raw를 local에 두고 compact index만 공유한다. 적어도 B1/B10 selected+native, arm별 atomic resume 상태, B1에서 실제 사용한 local-QP 재현 자산은 보존한다. B1이 D=0 skip이면 QP audit는 `NOT_APPLICABLE_NO_LOCAL_PROBLEM`이며 새 보정을 억지로 만들지 않는다. S10에서 첫 local 문제가 생기면 그 자산을 보존하여 audit한다. 저장하지 않은 후보의 exact replay를 주장하지 말라.

SH는 사실·수치·계약상 gate만 보고한다. GH는 `experiment-reports/global/bpcw512-20260918-v2/`에 별도 분석을 작성한다. 핵심 질문은 (a) 전체 reference 선택을 보호했는가, (b) RS/PS를 유지했는가, (c) arrival N 손실·과거 N forgetting이 줄었는가, (d) 비용이 수용 가능한가다. GSS, margin/QP 자체를 novelty로 단정하거나 reference 보존을 unseen locality 개선으로 대체하지 않는다.

## 13. GH 완료 보고

정상적으로 gate를 통과했다면 두 arm의 B10 완료와 검증까지 진행하라. 장시간 scheduler 대기는 `PENDING`으로 보고하고 job ID·사유·재개 위치를 남긴다. 제출 성공만으로 실행/검증 완료라고 쓰지 않는다. 외부 자원/필수 자산으로 완료할 수 없으면 마지막 유효 상태와 실제 미해결 의존성을 남겨라. 과학적 실패와 기술 실패를 분리하고 실패 결과도 보고한다.

최종 보고는 **실행 범위 → B1 gate → S10 실제 범위 → RS/PS/NS와 loss/gain → reference/Dev → 비용 → 기전 한계** 순으로 작성한다. 현재 지시문 작성 당시 상태는 `NOT_DISPATCHED / MODEL_RUNS_0`이며 GH가 실제 실행한 뒤에만 자신의 receipt에 상태를 갱신한다.
