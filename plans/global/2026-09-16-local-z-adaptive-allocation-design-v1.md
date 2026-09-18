# Local-z 중심 adaptive layer allocation 실험 사양 v1

2026-09-16. 최신 사용자 수정 반영: 모든 비교를 동일 pre-edit W0에서 시작한다. 사용자 지시: write layer에서 z를 계산하는 방식과 마지막 layer z를 분배하는 방식 중 주방식을 결정하고 실험을 구체화한다. **주방식은 layer-local z다. Terminal L8-z는 target 설계 대조로 둔다.** 이 파일은 실험 설계이며 새 GPU 실행·원격 제출 결과가 아니다. 기존 진행 실험의 정책을 변경하지 않는다.

## 결정과 검증할 질문

L4-only의 강한 RS/PS를 출발점으로, current 품질 조건을 만족하는 후보 중 독립 보존 reference에서 손상이 작은 write를 batch마다 선택한다. 첫 구현은 학습형 softmax나 ODE 없이 작은 후보 집합을 실제 forward로 평가한다. Request별 gate는 batch controller의 이득과 batch 내부 이질성이 확인된 뒤 확장한다.

- Local-z를 선택한 근거는 현재 fixed10k의 L4-only/BLUE 결과이며 local-z의 보편적 우월성을 가정하지 않는다.
- Gate는 해당 방식의 residual 적용 강도다. layer별 기능적 기여율이나 합이 1인 capacity 배분량이 아니다.
- C4 S64의 W0-output KL은 독립적인 일반 보존 proxy다. 공식 neighborhood 손상 자체를 controller가 관측한다고 쓰지 않는다.
- CAKE의 낮은 PS는 pooled at-write부터 존재한다. Static score가 PS 열세의 인과원인이라는 가정으로 arm을 설계하지 않는다.
- 첫 1,000-request 실험은 이미 조회한 fixed10k의 개발 실험이다. 별도 최적 fixed gate나 blind generalization보다 우월하다는 최종 주장이 아니다.

**추가 layer의 이득은 전제가 아니라 검증할 가설이다.** 사용자의 후속 질문을 반영해 연구 목적을 '여러 layer로 반드시 분산'이 아니라 '편집 품질을 유지하면서 실제 보존 손상이 작은 write를 현재 상태에 맞게 선택'으로 명시한다. Local-z 주방식 결정은 유지하되, LD가 L4D보다 우월해야 한다는 전제는 두지 않는다. L4-only를 선택하는 정책도 정상 결과다.

보존 입력의 일차 출력 변화는 J4 D4 + J8 D8이다. 추가 layer의 영향은 기존 손상과 더해지거나 상쇄될 수 있어 layer 수만으로 locality의 순서를 정할 수 없다. 동일한 목적·제약에서 D8=0이 허용되면 이상적인 다층 최적화 공간은 단층을 포함하지만, 작은 native-direction 후보 집합·C4 proxy·greedy batch 선택이 그 최적값이나 lifelong 지배 관계를 보장하지 않는다.

판단 순서는 L4D−N4로 적응적 강도 조절의 가치를 보고, LD−L4D로 추가 layer 후보의 가치를 보는 것이다. LD가 N4만 이기고 L4D는 이기지 못하면 multi-layer 분담의 기여가 입증되지 않는다. LD가 주로 L4를 선택하면 L8 후보의 품질 탈락, feasible하지만 큰 보존 비용, 거의0인 update, 수치 tie를 구분한다. 사용한 후보 family에서의 결과를 모든 다층 방법의 불가능성으로 확대하지 않는다.

S64에서만 개선하고 Dev128/공식 N에서 이득이 없으면 preservation proxy의 전이와 selector 일반화를 별도 문제로 기록한다. 추가 layer가 유용하지 않으면 L4 내부 request별 residual 강도 또는 방향 조절은 별도 후속 가설이며 자동으로 성공·신규성이 있는 것으로 승격하지 않는다. 첫 일곱-arm W0 실행 범위는 유지한다.

누적 budget과 후보별 locality 진단은 [보완 사양](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-16-local-z-adaptive-allocation-budget-diagnostics-v1.md)을 따른다. W0에서 시작하는70 batches와 online 선택 정책은 유지하고, LD/TD의 B1/B5/B10에 선택 봉인 후 공식 P/N 후보 평가와 고정 reference geometry 계측을 추가한다. 추가 observer prompt-state는 최대66,000개이며 그 비용은 기존190개 online 후보와 구분한다. 임의의 layer별 norm cap이나 KL budget은 도입하지 않는다.

## Target와 실제 write의 정확한 정의

W_e는 batch entry, M_l은 entry history, P_l은 원 projector다. 모든 arm의 AlphaEdit ridge는 1, target hparams는 기존 BLUE/L4 설정을 유지한다. Native 대비 변경은 target family와 명시한 gate/controller다. 원 AlphaEdit/CAKE의 L2=10 결과를 target-only causal control로 사용하지 않는다.

Native solve를 D_l(R;W)=R A_l(W), A_l=[solve(P_l(K_l K_l^T+M_l)+I,P_l K_l)]^T라 표기한다. 실제 실행은 원 direct solve 순서와 native context/key 평균을 유지한다. 이 표기는 inverse 선계산·대칭화·새 solver의 허가가 아니다. z/h는 원 native block readout 좌표이며 arbitrary module output을 빼지 않는다.

**LOCAL**

1. Entry에서 z4*(W_e), h4(W_e), K4를 계산하여 full native L4 endpoint N4를 만든다.
2. a4에 따라 W'=W_e+a4 D4를 materialize한다.
3. 이 W'에서 fresh z8*(W'), h8(W'), K8을 구해 D8을 fitting한다.
4. 최종 후보는 W'+a8 D8이다. 같은 a4 아래 a8 후보는 D8을 공유한다.

**TERMINAL**

1. Entry에서 Z8=z8*(W_e)를 한 번 계산하고 이번 batch 동안 고정한다.
2. L4 residual은 Z8-h8(W_e)다. z8-h4를 사용하지 않는다. L4 key와 공통 writer로 full D4T를 fitting한다.
3. W'=W_e+a4 D4T를 적용한 뒤 현재 h8(W')와 K8(W')를 다시 읽는다.
4. L8 residual은 Z8-h8(W')다. 같은 Z8을 유지하고 current residual만 갱신한다. 최종 후보는 W'+a8 D8T다.

두 방식 모두 모든 token에 weight를 실제 적용한 모델로 후보를 평가한다. Subject activation hook 결과를 후보 weight-write 결과로 대신하지 않는다.

Gate 구현은 기존 donor와 같은 실제 FP32 endpoint 축소다. Full fitting endpoint V와 entry U에서 g=0/1은 U/V를 그대로 복사하고, 나머지는 FP32(U+g*(V-U))다. RHS를 축소한 뒤 solve하는 방식과 실수 연산상 동등하지만 FP32 bytes까지 동일하다고 주장하지 않는다. Native two-layer terminal divisor 경로와의 parity는 실제 차이를 기록한다.

Local (1,0)은 N4, local (1,1)은 같은 entry의 BLUE다. Terminal (1,1)은 L4에서 terminal residual 전량을 먼저 시도하는 정책이다. Terminal 두-layer 균등분배는 (.5,1)이며 원 five-layer AlphaEdit은 1/5,1/4,1/3,1/2,1이다. T75를 원 AlphaEdit 또는 CAKE라고 명명하지 않는다.

## 첫 실행 범위와 일곱 정책

공통 시작은 pre-edit W0다. Llama-3-8B-Instruct revision 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, FP32/eager, fixed10k B1–B10, ordinal [0,1000), B100×10을 사용한다. 일곱 arm 모두 새로 실행한다. 각 arm은 같은 W0에서 시작한 뒤 자기 endpoint/history를 다음 batch로 넘긴다. 같은 1,000요청을 일곱 정책이 처리하며 최종 full-seen도 1,000이다.

M4/M8의 edit history는 최초에 정확히0이다. 원래 지식에서 구한 covariance/projector P4/P8은 공통 자산을 유지하며 0으로 만들지 않는다. 원모델·tokenizer·P·native hparams·context 텍스트/token bytes·입력 순서·RNG·source를 결속한 공통 cold capsule을 한 번 준비한다. Seed는20260916이고 native context 생성이 필요하면 W0에서 한 번만 수행하여 모든 arm에 복사한다. BF16/FP32, kernel, microbatch, padding, TF32 설정을 arm별로 바꾸지 않는다. TF32는 공통 off로 고정하고 기존 teacher의 설정 차이가 있으면 아래 teacher 연결 검사를 거친다.

이 선택은 warm checkpoint 복원·이전 편집 경로 차이를 제거한다. 하드웨어 비결정성이나 표본/order 변동까지 완전히0이라고 보장하지 않는다. 같은 endpoint의 반복 평가와 source/config identity로 남은 수치 변동을 확인한다.

| ID | Target·정책 | Gate/후보 | 역할 | 실행 |
|---|---|---|---|---|
| N4 | L4 local | (1,0) | 주 baseline | 신규 |
| REFIT4 | L4 .75 후 같은 L4 fresh fit | (.75,1), 두 subwrite 모두 L4 | 추가 target 최적화 효과 | 신규 |
| L75 | LOCAL 고정 | (.75,1) | RES8 고정 정책 | 신규 |
| T75 | TERMINAL 고정 | (.75,1) | 같은 gate의 target 대조 | 신규 |
| L4D | L4 local 동적 | a4∈{.75,1}, a8=0 | 단순 강도 조절 대조 | 신규 |
| LD | LOCAL 동적 | a4∈{.75,1}, a8∈{0,.5,1} | 제안 주방법 | 신규 |
| TD | TERMINAL 제안+N4 동적 | LD와 같은 6 gate + 공통 N4 | terminal 제안의 추가 가치 | 신규 |

과학 비교는 신규7 chains×10=70 batches로 고정한다. 과거 N4/REFIT4/RES8의 warm suffix 또는 이전 cold 결과를 이번 paired 신규 arm의 실행 결과로 대체하지 않는다. 기존 결과는 method 선택의 선행 evidence로만 사용한다. 새 unique 요청은1,000개이고 arm-request 관측은7,000개다.

FULL8/BLUE의 기존 warm suffix와 원 fixed10k AlphaEdit/BLUE/CAKE 결과는 historical 참고표에 남긴다. 새 7-arm의 matched causal table과 혼합하지 않는다.

**TD는 pure-terminal 정책이 아니다.** Common N4를 선택할 수 있는 terminal-proposal controller다. LD에서는 N4가 6개 family 후보 안에 있지만 TD에서는 별도 7번째 후보다. TD의 N4 선택률과 terminal 후보의 raw 성능·feasibility를 반드시 공개한다. TD의 이득이 N4 fallback에서만 나오면 terminal allocation의 성공으로 해석하지 않는다.

L75/T75는 raw fixed endpoint를 commit한다. Dynamic-vs-fixed 차이는 후보 탐색·quality screen·N4 fallback 전체의 효과다. 이를 allocation-only 또는 최적 fixed 대비의 효과라고 부르지 않는다. 같은 숫자 gate가 같은 실현 품질을 뜻하지 않으므로 실제 Current 품질과 Pareto 결과를 함께 비교한다.

## Online 입력과 reference

- Current: 지금 도착한 B100의 native rewrite training contexts와 canonical rewrite. 공식 P/N은 입력 금지.
- Base: 기존 C4-WebRef-v2 S64 고정 token과 W0 full-vocabulary teacher. 문서별 128 scored positions 평균 뒤 64문서 평균으로 KL(p0||p_candidate)을 계산한다. Prompt의 모든 token에 candidate weight를 적용한다.
- Past64: 지금까지 수신한 과거 request의 canonical rewrite만 사용한다. Raw (subject,relation)의 최신 target을 active로 삼고 현재 batch에서 overwrite되는 fact는 제외한다. 동일 target 반복은 fact별 최신 event 하나를 대표로 한다. SHA256('LZ-ALLOC-v1|20260916|past|'+stable_event_id) 오름차순으로 최대64개를 선택한다. ID 문자열 인코딩은 실행 manifest에 봉인한다. 성능이나 미래 request 목록으로 선택하지 않는다.
- Past64 입력 ID는 같은 history ledger를 가진 arm 사이에 같아야 하며 각 branch의 candidate/entry/N4 출력은 따로 평가한다. 더 작은 ledger면 전부 사용한다. 공식 Historical128·P/N과 섞지 않는다.
- Dev128은 B5/B10의 observer다. S64 선택·온도·gate 변경에 사용하지 않는다. Report256 teacher 신규 구축은 이번 pilot의 선행조건이 아니다.

S64 teacher는 원 W0로 고정하고 batch-entry 또는 다음 batch의 손상된 모델로 교체하지 않는다. C4 conflict 여부를 미래 fixed10k 목록으로 판별하지 않는다. 기존 공개된 text-overlap checker 결과와 확인 범위를 유지하며 공식 evaluation labels·scores를 controller에 전달하지 않는다.

## Deterministic selector

Dynamic arm은 매 batch 자기 entry에서 만든 실제 N4 endpoint를 공통 비교 기준으로 쓴다. 독립 N4 chain의 미래 checkpoint나 loss를 가져오지 않는다. N4는 해당 batch의 counterfactual action이며 그 계산 비용도 포함한다.

E는 native rewrite contexts에서 target-new의 token→context→request 평균 NLL, S_cur는 canonical rewrite의 target-new TF strict 성공 ID 집합이다. H는 Past64 canonical의 평균 NLL, S_past는 같은 입력의 strict 성공 집합이다. E/H 제한은 batch·memory 평균이며 request/context별 NLL 비열화를 보장하지 않는다. D는 S64의 고정 W0 KL이다.

후보 c의 feasibility는 다음으로 고정한다.

1. E(c) ≤ max(E(N4),0.05)+epsilon_E.
2. S_cur(N4) ⊆ S_cur(c).
3. Past64가 있으면 H(c) ≤ H(N4)+epsilon_E, S_past(N4) ⊆ S_past(c).

0.05 nats/token은 이미 충분히 낮은 Current loss에서 추가 confidence만 보존하느라 약한 write를 모두 배제하지 않기 위한 **사전 고정 plateau**다. Native target의 total-loss early-stop와 동일한 의미가 아니며 PS 보장도 아니다. 이번 pilot 결과를 보고 조용히 바꾸지 않는다. E per-request/context, tail, native 대비 악화와 entry 대비 past 손실을 함께 보고한다. N4 자체가 잃은 과거 edit를 이 조건이 복구한다고 주장하지 않는다.

epsilon_E=1e-4 nats/token, epsilon_D=1e-6 nats/token을 수치 동률 허용으로 사용한다. 동일 endpoint 재평가 오차가 각 허용치의 절반 이내인지 기술 준비에서 확인한다. 실패하면 평가 재현성을 수정하고 성능을 보고 허용치를 키우지 않는다. Strict bit도 재평가에서 동일해야 한다.

Feasible 후보의 D 최솟값에서 epsilon_D 이내인 후보들을 tie set으로 둔다. 우선순위는 공통 N4, 이번 batch L8 변화가 없는 후보, 작은 실제 두 weight의 합성 Frobenius norm, candidate ID 사전순이다. 동일 materialized weight 후보는 deduplicate한다. 과학적 성능에 따른 후보 조기중단은 없으며 선언한 후보를 모두 관측한다.

N4는 기준을 스스로 만족하므로 기술적으로 정상인 batch에는 선택 가능한 endpoint가 있다. N4 선택은 사전에 정의한 정상 정책이며 횟수·사유를 기록한다. NaN·손상된 state·OOM은 N4 fallback 사유가 아니라 기술 실패다. 실패 batch는 entry를 복원하고 미완료로 기록한다.

선택 보장은 그 batch의 **관측 S64와 선언한 training 조건에만** 적용된다. 독립 N4 lifelong chain보다 최종 NS가 높다는 보장, 매 batch W0-KL의 단조 감소, official PS 비열화 보장이 아니다. Past threshold는 common N4 relative이므로 누적 accepted-target drift는 별도 계측한다.

추가 forward 없이 이미 얻은 점수로 'plateau 없이 E≤E(N4)'와 'strict만 적용'의 shadow 선택 ID를 진단할 수 있다. 이 shadow는 실제 commit·후속 state·정책 선택에 사용하지 않으며 반사실 lifelong 결과를 합성하지 않는다.

## State·cache·history

후보는 같은 W/M/P/context/RNG entry에서 격리한다. 후보 생성 순서를 바꿔도 후보 endpoint와 선택 결과가 같아야 한다. Target마다 full model-state identity, physical target layer, context/token/lookup, source/hparams/RNG를 결속한다. 기존 case/layer/clamp만 담긴 cache 파일 이름으로 branch target을 공유하지 않는다.

Local z4는 그 batch의 같은 entry에서만 공유한다. Local z8은 a4별 partial state마다 fresh 계산한다. Terminal Z8은 그 branch의 batch entry에서 한 번 계산하고 고정한다. 두 family의 future target을 공유하지 않는다.

모든 inner fitting/candidate에서 history append는0이다. 최종 selected endpoint에서 N4/REFIT4/L4D는 M4를1회, L75/T75/LD/TD는 M4와 M8를 각각1회 append한다. LD/TD에서 a8=0 또는 common N4를 골라도 잠재적으로 쓰는 L8의 전체 batch history는 append한다. Gate로 history를 가중하거나 요청을 제거하지 않는다. Actual write0과 history state0을 혼동하지 않는다.

모든 arm의 사용 대상 edit memory는 M0=0에서 시작한다. Warm prepared.pt나 과거 BLUE chain의 M을 import하지 않는다. 이후는 각 branch final keys를 append하며 전체 과거 M8 refresh를 추가하지 않는다. 이때의 stale-key 가능성은 별도 기전이며 이번 allocation 변화와 함께 수정하지 않는다.

## 비용과 최소 구현 변경

| 정책 | B100 target request-calls | writer solves | unique endpoints 최대 | history appends |
|---|---:|---:|---:|---:|
| N4 | 100 | 1 | 1 | 1 |
| REFIT4 | 200 | 2 | 1 | 1 |
| L75 | 200 | 2 | 1 | 2 |
| T75 | 100 | 2 | 1 | 2 |
| L4D | 100 | 1 | 2 | 1 |
| LD | 300 | 3 | 6 | 2 |
| TD | 200 | 4 | 7 | 2 |

LD는 N4/local4 1회와 a4 두 상태의 local8 2회다. TD는 공통 N4용 z4와 entry Z8 각1회, N4 solve1+terminal4 solve1+두 partial-state L8 solve2다. Per-target 최대25 loss 평가/24 Adam update이고 실제 early-stop을 기록한다. 위 solve 수는 actual full endpoint를 만든 뒤 gate를 materialize하는 구현일 때의 값이다.

신규7 arm×10 batches의 총 target request-calls는12,000, 최대 Adam288,000/loss300,000, writer solves150, candidate endpoints190이다. Target·candidate Current/Past/S64 forward·history·copy·준비·평가 비용을 별도 기록한다. Fixed arm도 결과 지표는 같은 방식으로 평가하지만 online selector 비용으로 거짓 분류하지 않는다. Cost-matched FLOPs 또는 일정한 runtime 배수를 약속하지 않는다.

재사용 출발점은 보존된 low_cost_write_donor_pilot/fitting.py와 sequential_runtime.py다. 원 native source는 수정하지 않는다. 별도 모듈에서 다음만 추가한다.

1. 명시적 TERMINAL target/readout adapter. Singleton fitter에 z8만 주입하면 h4를 빼는 오류가 생기므로 금지한다.
2. .5 gate를 포함하는 versioned materializer와 candidate-state cache.
3. S64/Past64 평가와 pure deterministic selector.
4. Candidate ledger, source/state/cost receipt, 최종1회 finalizer.

현재 이 설계의 model runner는 구현되지 않았다. CPU reference는 selector·후보 수·대수 검산용이며 real-model numerical parity나 성능 검증이 아니다. 실행자는 별도 codex/ branch의 새 adapter에서 구현하고 기존 진행 run을 수정하지 않는다.

## 기술 검증과 과학 실행

기술 준비는 local (1,0)↔N4, local (1,1)↔같은 entry BLUE, terminal (.5,1)↔원 non-blue [4,8]/L2=1의 source·actual update 연결을 확인한다. FP32 endpoint scaling과 RHS scaling의 오차를 숨기지 않는다. Terminal (.5,1)은 canonical 기준점의 기술 점검이며 첫6-gate scientific menu에는 추가하지 않는다.

Source/target/cache/history·후보 순서 독립성과 Current/S64 재평가의 수치 재현성을 확인한다. S64 teacher/token identity와 공통 cold W0·tokenizer fingerprint의 연결은 이번에 새로 확인한다. 모든 arm의 시작 selected-weight SHA가 W0와 같고 M의 nonzero count가0인지 검사한다. 기존 teacher를 유지할 수 있는지 common W0 forward와 teacher 재평가를 대조한다. Kernel/TF32/token 차이가 재현 허용치를 넘으면 같은 S64·W0에서 teacher를 한 번 재생성하여 전 arm에 동일하게 사용하고 새 teacher ID와 비용을 기록한다. 후보 성능을 보고 teacher를 선택하지 않는다.

기술적으로 유효한 일곱 정책은 선언한 B1–B10을 완료한다. B1 점수만 보고 arm을 삭제하거나 gate를 바꾸지 않는다. 일곱 paired arm은 모두 이번 공통 cold 설정으로 새로 실행한다. GPU를 쓰는 기술 검증과 과학 실행 비용은 분리한다.

## 평가와 주장 경계

매 batch Current R/P/N을 entry와 selected endpoint에서 평가하고 at-write 결과를 문항 ID로 연결한다. Official P/N은 observer만 접근한다. 과거 평가는 해당 시점까지 도착한 요청으로 정의하며 기존 warm 실험의 Historical128을 가져오지 않는다. 공식 observer 문항을 Past64 selector로 재사용하지 않는다.

B5는 first500을, B10은 동일 first500·후반500·전체1000을 평가한다. 각 batch의 current100과 이전까지의 요청은 분리하고 B1의 과거 요청 집합은 empty로 명시한다. ALL/ACTIVE/SUPERSEDED를 분리하고 R/P strict·두 P 모두 strict·true/new NLL·안전 margin·p95/p99 악화·lost/gained를 보고한다. W0→entry, entry→at-write, at-write→final을 구분한다. Base는 S64와 Dev128을 분리한다.

주 비교는 LD−L4D(추가 layer 후보의 가치), LD−L75(선언한 고정 정책 대비 controller 전체), LD−REFIT4(다른 layer와 같은 layer 재fitting), LD−TD(local/terminal 제안+공통 selector), T75−L75(고정 gate의 target-policy 차이)다. 서로 다른 trajectory의 B2 이후 차이는 그 정책의 전체 효과이며 단일 batch 직접효과가 아니다.

선택률·feasibility·N4 fallback 이유·a4/a8 분포·후보별 Current/Past/D를 보고한다. Norm share 대신 실제 edit/보존 출력 변화도 함께 남긴다. LD가 주로 a8=0을 고르면 multi-layer 기여가 입증된 것으로 쓰지 않는다. TD가 N4에 의존하면 그 사실을 전면 보고한다.

성능·비용의 임의 AND hard gate로 연구를 중단하지 않는다. 비교가 유효하면 ALLOW_WITH_LIMITED_CLAIM / NEEDS_TARGETED_CHECK / NOT_SUPPORTED 등 질문별 판정을 하며 PS/strict/old-edit 손실을 숨기지 않는다. 'RS/PS 비열화가 입증됐다'는 통계적 주장은 별도 허용폭·반복 설계 없이는 하지 않는다.

첫 결과가 양성이면, 개발 구간에서 family별 고정 gate를 선택·봉인한 후 다른 stream에서 같은 selector를 검증해야 최적 fixed 대비 동적성 주장을 할 수 있다. 필요하면 고정 ratio의 공통 scalar controller를 추가해 일반적인 strength 조절과 독립 layer gate의 효과를 분리한다. 이를 첫 일곱-arm에 자동으로 확대하지 않는다.

그다음에만 W0→10k의 선택정책·N4·가장 강한 단순 대조를 비교한다. Fixed10k가 이미 개발에 쓰였으므로 새 order는 순서 강건성이고 unseen 검증이 아니다. Request별 gate, L4+L5, preservation-only compensation은 각각 별도 후속 가설이다.

## 재사용 자산과 근거

- 시작 state는 pinned 원모델 W0와 zero edit memory다. 새 cold capsule의 weight/model/token/context/P/RNG 해시는 실행 준비에서 기록하며 기존 warm capsule을 쓰지 않는다.
- Teacher manifest: /data/janghj/ODE-edit/local/bg1-c4-ours-first/20260915-v1/attempt-v1/teacher-output-v1/teacher-manifest.json, SHA256 f81b798f44ce626ac1e2e402ca7438363b1dd60f5681ec0ac17b9e92d096761a.
- Reference768 identity: f5791dd3c986261a252796bd5d7293ece46339d261609449dcfe674970bbfde0. Teacher192(S64+Dev128) 완료 자산을 재사용하며 과거 PENDING 문구를 현재 상태로 쓰지 않는다.
- [기존 donor 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-13-low-cost-write-donor-pilot-design.md), [완료 six-arm 리뷰](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-14-lowcost-seq10-review-ko.md), [reference 계약](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-bg-tw-reference-data-contract.json).
- [Teacher 완료 보고](/mnt/raid5/janghj/.codex/worktrees/odeeditgh-sh4-cake-baseline-cap-report-split-20260916-v1/experiment-reports/servers/server4/ep-tw1-c4-2026-09-15-v1/gate-skip-r1/completed-review-v1/diagnostic-report-ko.md:56).
- [과거 warm 완료 보고: 선행 evidence 전용](/mnt/raid5/janghj/.codex/worktrees/odeeditgh-sh4-cake-baseline-cap-report-split-20260916-v1/experiment-reports/servers/server4/low-cost-write-donor-seq10-2026-09-13-v1/completed-review-v1/diagnostic-report-ko.md:47).

Remote 경로는 기존 보고서의 자산 위치이며 이번 작성에서 원격 tensor를 재해시한 것이 아니다. 새 source commit·실행 manifest·job ID는 아직 없다.
