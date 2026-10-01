# 전체 편집층 공동 최적화와 보존 목적을 비교하는 JLZ 두 버전 설계

2026-10-02 논의 초안이다. L4–L8 전체의 actual-write 공동 최적화, 출력 보존 reference, 계산 효율화, 진단 개선을 두 arm에 공통 적용한다. 두 arm 사이의 유일한 방법 차이는 추가 quadratic write 비용이다. 한 층 전담과 나머지 층의 0 해를 두 arm 모두 허용한다. 기존 job 56684와 server3 실험을 변경하거나 새 실험을 제출한 문서가 아니다.

## 확정된 범위와 초기 제안값

- 편집 후보는 L4, L5, L6, L7, L8의 down_proj 전체다. 사전 layer 선택, top-k, 최소 활성 층 수, 최대 층 점유율 제한을 두지 않는다. 모델의 32개 층 전체를 편집한다는 뜻은 아니다.
- FP32 모델, FP64 geometry, 실제 writer를 포함한 공통 최종 손실, native KL(current‖batch-entry), 기존 group norm과 .75 block clamp를 유지한다.
- Arm A는 추가 write 비용 계수 η=0, Arm B는 η>0이다. **초기 제안은 η=1.0**이며 효과가 교정된 값은 아니다. 학습·NS 결과를 보고 배치 중 계수를 바꾸지 않는다.
- 일반 reference 16개와 과거 편집 reference 최대16개를 배치마다 사용한다. 일반 reference 계수 βG=.0625, 과거 replay 계수 βE=1.0을 첫 제안으로 둔다. 이 계수들도 새 가설의 일부이며 기존 native 설정에서 자동 유도되지 않는다.
- 모든 값은 설계 초안의 기본값이다. 실제 실행 전 단일 frozen 설정으로 결속하고, 이후 변경은 새 버전으로 기록한다. 수치·품질 warning만으로 층을 제외하지 않는다.

## 현재 구현에서 유지하는 계산 그래프

배치 b의 진입 모델과 H에서 K, anchor h, native teacher를 얻는다. A_l=15000 C0_l+H_l, P_l=solve(A_l+K_l K_lᵀ,K_l)를 계산한다. 배치 동안 P와 teacher는 고정한다. R_l은 4096×B 행렬이며 모든 층에서 0으로 초기화한다.

각 후보의 weight는 **W_entry + (R.double() @ P.T).float()**로 정의한다. 모델의 모든 token에서 이 weight가 작용하며, 상위층의 실제 activation과 key는 forward마다 변한다. P 고정은 실제 activation 고정이 아니다. Native z를 먼저 따로 최적화하거나 subject 위치의 가상 activation만 최적화한 뒤 writer로 변환하지 않는다.

공통 forward의 NLL/KL에서 모든 R에 gradient를 전달한다. 모델 weight는 후보 평가 중 영구 변경하지 않는다. 마지막 accepted 후보 또는 초기점을 원래 FP32 materialization으로 재평가한 뒤 같은 tensor를 commit한다. 모든 층 write 이후 post-key로 H를 층별 한 번씩 append한다.

원 실험의 entry NLL<.05인 요청에 대한 R column 0 규칙은 유지한다. 이는 layer mask가 아니다. 이 요청도 다른 column의 write 영향을 받을 수 있으므로, 아래 공통 보존 항으로 이미 잘 맞던 현재 요청의 손실을 보완한다.

## 두 arm의 목적함수

기존 sum 단위를 유지한다. NLL은 요청별 target-token 평균, rewrite context 6개 평균이다. 활성 요청 집합을 I, 일반 reference를 G, 과거 편집 reference를 E라고 한다.

F_native = Σ_(r∈I) [NLL_r + .0625 KL(p_R,r ‖ p_entry,r)]
           + Σ_(l,r) .5 ||R_l,r||₂ / ||h_l,r||₂².

F_common = F_native + B βG mean_(g∈G) KL(p_R,g ‖ p_W0,g)
                         + B βE mean_(e∈E) NLL_desired,e
                         + Σ_(r∉I) [NLL_r + .0625 KL(p_R,r ‖ p_entry,r)].

마지막 항은 이미 계산한 inactive current request의 NLL/KL를 사용한다. R column의 zero-step 규칙은 유지하면서 다른 요청의 write가 해당 요청을 손상시키는 효과를 공통 손실에서 본다. 별도 forward가 필요 없으며, 원 실행의 active-only loss와 달라지는 **두 arm 공통 방법 변경**이다.

일반 reference의 추가 KL도 current‖teacher 방향이다. 기존 native KL 방향을 변경하지 않는다. 전체 목적을 B로 나누어 설명해도 되지만 구현의 scalar와 gradient는 기존 sum 단위로 맞춘다. reference 개수에 따라 목적의 가중치가 자동으로 변하지 않도록 그룹별 mean을 사용한다. E가 비어 있으면 replay 항은 0이며 G 가중치를 키우지 않는다.

βE=1이면 과거 reference 집합 전체의 평균 NLL은 현재 B개 요청 전체의 평균 NLL과 같은 가중치를 받는다. B100/E16일 때 과거 reference 하나의 직접 계수는6.25다. 작은 replay 집합에 상당한 비중을 주는 새 가설이며, 단지 입력 row가 적다는 이유로 목적 영향도 작다고 해석하지 않는다.

M_l=P_lᵀ A_l P_l, s_l²=mean_r ||h_l,r||₂²,
Ω(R)=1/2 Σ_l tr(R_l M_l R_lᵀ)/s_l².

| 버전 | 목적 | 허용되는 결과 |
| --- | --- | --- |
| A | F_common | 한 층 전담, 다층 분담, zero layer 모두 허용 |
| B | F_common + η Ω | 같은 허용영역에서 추가 write 비용까지 고려한 배분 |

기존 group norm을 B에서 제거하거나 교체하지 않는다. 추가 smooth gradient는 η R_l M_l/s_l²이다. 따라서 기존 group-norm prox와 block clamp를 공통으로 사용할 수 있다. Ω는 균등 사용이나 정해진 층 수를 강제하지 않는다. 한 층이 훨씬 효과적이면 B에서도 그 층에 집중할 수 있다. B의 정확한 의미는 **분산 강제가 아니라 write 비용을 추가한 배분**이다. 편집 자체를 약하게 만들어 비용을 낮출 수도 있으므로, 같은 edit strength에서 locality와 층별 부담을 비교해야 한다.

목적의 Ω는 ΔW_ideal=R_l P_lᵀ의 이상적 FP64 에너지다. 실제 ΔW_actual=materialized_weight−W_entry에는 FP32 cast와 weight addition 반올림이 포함된다. 따라서 ideal 에너지, commit 후 actual 에너지 tr(ΔW_actual A_l ΔW_actualᵀ), 각 정규화 비용, R norm, D_l=R_l(P_lᵀK_l)의 norm을 구분해 기록한다. Actual 에너지는 최종 후보에서만 별도 진단 비용으로 계산해 매 oracle 비용을 키우지 않는다. s_l 정규화는 activation scale에 대한 선택이며 모든 층의 기능 민감도를 같게 만들지는 않는다. 두 arm의 서로 다른 total objective를 직접 비교해 승패를 판단하지 않는다.

M은 B×B 행렬로 배치당 한 번 계산한다. S=sym(KᵀP)이면 이상적인 산술에서 M=S−S²이다. 이 작은 행렬 경로를 사용하되, 처음 geometry 검산 및 불안정한 경우 PᵀAP의 직접값과 비교한다. 유한한 작은 불일치는 기록하고 직접식을 사용하며, eigenvalue를 조용히 잘라 목적을 바꾸지 않는다. A에서도 M을 계산해 동일 진단과 비용 결속에 사용한다.

## 평가와 분리한 reference

### 일반 사실 보존

기존 로컬 전체 CounterFact 21,919개를 후보로 사용한다. fixed10k에 속하는 case, 편집 stream의 동일 claim 또는 동일 subject, fixed10k R/P/N 평가와 정규화된 canonical prompt가 일치하는 항목을 제외한다. 이 필터는 문자열·데이터 식별자 기준이며 의미적 중복이 전혀 없다는 보장은 아니다. 최종 NS 문항 자체를 학습 reference로 넣지 않는다.

이번 CPU 점검에서 후보는 11,919 → 동일 claim 제외 11,544 → 평가 prompt 제외 11,134 → 편집 subject 제외 **10,715개, 34 relation**으로 남았다. 모델 평가나 점수에 따른 선별은 하지 않았다. 원자료 SHA와 재현 코드는 evidence.json 및 inspect_design.py에 기록한다.

전체 후보 목록에서 seed20261002와 고정 hash로 배치별16개를 뽑는다. 가능한 경우 현재 배치 relation에 대응하는 후보8개와 전체 pool의 후보8개를 사용하고, 관계 후보가 부족하면 전체 pool에서 중복 없이 채운다. 단일 배치 안에서는 고정한다. 두 arm은 같은 ID·순서·tokenization을 사용한다.

요청당 canonical prompt 한 개와 전체 true target token prefix를 사용한다. target prediction 위치 전체에서 full-vocabulary KL을 구하고 token 평균 후 prompt 평균한다. 길이 자르기나 어휘 근사는 하지 않는다. Teacher는 W0에서 해당 입력으로 계산한 FP32 log probability이며 CPU에 보관한다. 필요한 reference가 정해진 뒤 W0 상태에서만 teacher를 준비하며, arm B에서 arm A의 편집 모델을 teacher로 사용하지 않는다.

### 과거 편집 보존

W0의 원래 답을 보존하면 이미 바꾼 사실과 충돌하므로 past reference는 **현재 유효한 desired-new target의 NLL replay**를 사용한다. 이는 단순한 출력 유지뿐 아니라 망각된 편집을 다시 맞추는 효과도 포함한다.

과거 committed 요청에서 (정규화 subject, relation_id)의 최신 target만 유지한다. 현재 배치에서 다시 편집할 claim과 superseded target은 replay에서 제외한다. 성공 여부나 NS 결과로 표본을 골라내지 않는다. 시간 사분위별 고정 hash로 최대4개씩, 총16개를 고르며 부족한 경우 남은 pool에서 결정적으로 채운다. 16개 미만이면 전부 사용한다. Canonical prompt 하나와 전체 desired target token을 사용한다.

H에는 기존처럼 모든 committed occurrence와 multiplicity를 유지한다. Reference의 latest-claim membership과 H의 occurrence membership을 혼동하지 않는다. 두 arm의 요청 ledger가 같은 prefix까지 진행했다면 past reference ID는 같지만, 각 arm의 실제 모델·H·native teacher·P는 자신의 trajectory를 따른다.

## 계산 경로와 예산

두 arm 모두 L0–L3 prefix 재사용, 후보당 materialization 공유, 필요한 prediction 위치의 full-vocabulary head, microbatch별 오른쪽 padding 제거, 단일 down_proj 계산, key 추출 L8 조기 종료, endpoint 평가 재사용을 적용한다. 현재 batch의 모든 loss 성분을 같은 후보 weight로 평가한다.

직접 R 역전파는 이미 별도 효율화 코드에 있는 경로를 공통 기술 비교한다. 먼저 원 materialized forward를 유지하며 큰 dW 생성만 제거한다. 계산 중 dX도 유지한다. Reference route와 같은 고정 후보에서 loss/gradient/finite를 기록하고 실제 wall·메모리로 공통 route를 정한다. 유한한 오차의 기존 threshold 초과만으로 제외하지 않는다. 사용한 route와 수치 warning은 결과에 남긴다.

선택된 E123_MB4의 이전 B100 median47.026→37.597초는 native-only 기술 endpoint 측정이며, direct-R의 결과나 reference 추가 후 속도는 아니다. Factorized forward는 FP64 token contraction과 rounding 차이 때문에 자동 가속을 보장하지 않는다. 이번 두-arm 차이에 추가하지 않는다. 저차원 방향 고정, activation-only proxy, layer subset도 도입하지 않는다.

한 후보의 대략적인 입력 row는 현재700 + 일반16 + 과거최대16이다. 최대 row 증가4.57%는 FLOPs·시간 증가율이 아니다. Reference 길이, KL head 위치 수, microbatch packing에 따라 실제 비용이 달라진다. W0 teacher 준비, prefix 준비, geometry, 모델 F/B, reference, observer 시간을 별도 집계한다.

첫 비교의 solver는 기존 SPG/BB/prox/line search를 공통 사용한다. Solver 교체는 이번 두-arm 차이에 섞지 않는다. Science cap에는 초기, accepted/rejected trial, trial 재확인, 마지막 원본 경로 재평가를 모두 포함한다. 별도의 숨은 full oracle 호출을 하지 않는다. 마지막1회를 반드시 예약한다. 원본 no-grad commit 재평가와 R/P/N observer는 별도 비용으로 기록한다.

## Gate와 진단의 역할

| 항목 | 새 설계의 제안 처리 | 배분에 대한 의미 |
| --- | --- | --- |
| 사용 층 수, 한 층100% 집중, zero layer, 최대 점유율 | record-only | 실패·재배분·mask 근거로 사용하지 않음 |
| 낮은 cosine, 실현률, 상대 전달오차 | record-only | JLZ의 학습 경로와 목적이 판단할 문제 |
| RS/PS/NS 또는 reference loss 악화 | 결과와 목적 성분으로 기록 | 별도 hard ceiling·자동 η 변경 없음 |
| BUDGET_STOP, STALLED, 유한한 LINESEARCH_FAILED | 마지막 accepted/초기점을 fresh 평가 후 채택 가능 | 수렴을 강제하는 gate가 아님 |
| 유한한 loss/gradient/prox 오차의 기존 기준 초과 | 원 판정과 실제값 보존, CONTINUE_WITH_WARNING | 기준을 완화해 PASS로 고쳐 쓰지 않음 |
| reference/optimized solver의 경로 차이 | 기록 | 같은 궤적이라는 주장만 하지 않음 |
| 서로 다른 arm의 accept/reject·층 배분 차이 | 정상적인 비교 결과 | arm 간 parity를 요구하지 않음 |
| 잘못된 input/teacher/token/KL, dX 누락·잘못된 VJP/prox/Ω 미분, 다른 weight commit, 원치 않은 상태 변경 | 중단·rollback | 유한한 값이어도 계산 의미·실험 무결성 문제 |
| 최종 후보/원본 최종 loss·gradient·weight NaN/Inf | 채택 금지·rollback | 실제로 평가 가능한 결과가 아님 |
| 기존 .75 clamp 또는 zero-step 제약 위반 | 채택 금지 | 진단이 아니라 명시한 허용영역 위반 |

**Commit weight의 bitwise 일치와 commit loss의 수치 parity를 구분한다.** 최종 원본 oracle이 평가한 weight tensor를 그대로 복사하는 것은 강하게 확인한다. 독립적인 forward의 유한한 loss 차이는 경고로 기록하고 실제 materialized 평가값을 최종 보고값으로 사용한다. Native KL teacher/input 의미가 다르거나 intended tensor가 다르면 수치 warning으로 넘기지 않는다.

유한 오차의 warning 처리는 반올림·유효한 실행 순서 차이에 대한 규칙이다. 미분 경로 누락 등 확인된 구현 결함을 유한하다는 이유로 허용하는 규칙이 아니다. 수치 차이 하나만으로 결함을 단정하지도 않는다. 원 판정과 오차를 보존하고, 실제 결함이 확인되면 공통 경로를 수정한다.

**논의가 필요한 변경 1: trial의 비유한값.** 원 policy는 어느 trial에서든 NaN/Inf가 나오면 전체 batch를 막는다. 새 설계에서는 영구 weight를 바꾸지 않은 trial의 비유한 loss/gradient만 발생한 경우 그 trial을 거절하고 step을 절반으로 줄여 예산 내에서 계속하는 안을 제안한다. 마지막 accepted 상태는 유지한다. 초기점 또는 최종 원본 평가가 비유한이면 중단한다. CUDA 오류·state 손상은 단순 trial 거절로 처리하지 않는다. 이는 원 gate와 다른 **공통 정책 변경 제안**이며 원 job에는 적용하지 않았다.

**논의가 필요한 변경 2: .75 clamp.** 현재는 두 arm 모두 유지한다. 단층 집중이 허용된다는 것은 이 상한을 넘는 단층 해까지 허용한다는 뜻은 아니다. 층별 clamp hit 비율·KKT boundary 정보를 기록해 실제 제약이 배분을 막는지 판단한다. 새 layer별 energy cap이나 최대 점유율 cap은 추가하지 않는다. Clamp를 바꾸려면 두 arm 공통 조건으로 별도 버전 결정이 필요하다.

**논의가 필요한 변경 3: 초기 계수.** η=1, βG=.0625, βE=1은 초안값이다. Pilot에서 각 loss와 gradient 크기, edit strength·locality 곡선을 함께 보고 해석한다. Ratio가 크다는 이유로 arm을 탈락시키거나 η를 자동 변경하지 않는다. 재설정이 필요하면 고정된 새 설정으로 다시 명시하며 기존 결과도 남긴다.

## 비교와 단계별 실행안

두 arm은 동일한 W0/H0, 요청 순서, reference 선택, runtime·route·초기값·예산으로 독립 chain을 시작한다. B1 이후 weight·H·P·teacher가 달라지는 것은 정상이며 매 배치를 억지로 같은 상태로 reset하지 않는다. 일반 W0 teacher만 공통이다. 이전 JLZ에는 새 reference와 inactive 보호가 없으므로 새 A를 원 job56684와 같은 baseline으로 부르지 않는다.

1. CPU에서 수식·reference 분리·정규화·판정표를 점검한다. 실제 모델 성능 PASS로 해석하지 않는다.
2. 초기 작동 검증 제안: fixed order 첫8개를 BS4×2로, 두 arm 각각 cap32/batch. Science 상한 총128회. 두 번째 배치로 past replay와 H 연결을 확인한다. 이는 원 cap120 실험과 다른 작은 pilot이다.
3. B100 기술·방향 검증 제안: 같은 첫100개에서 두 arm 각각 cap32 한 배치, science 상한총64회. Same-candidate route 검사는 공통이며 별도 기술예산으로 공개한다. 기술예산의 신규 모델 oracle는 최대6회로 제한하고 teacher/cache 준비도 시간·token 장부에 남긴다.
4. 이후 BS100×10 비교는 별도 실행 단계로 두 arm 모두 cap120을 유지하는 안이다. 이번 요청에서는 설계를 구체화하며 새 job을 제출하지 않는다. 작은 pilot의 좋은 점수를 자동 확장 gate로 삼지 않는다.

8/16/32 시점은 단일 cap32 실행의 accepted incumbent trace로 기록한다. 독립 cap8/16 실험과 동일하다고 주장하지 않는다. Trace에는 objective 성분·gradient/잔차·층별 norm/energy·경과시간을 남긴다. 대규모 R/W/H tensor 또는 복원 checkpoint는 저장하지 않는다. 필요한 중간 모델 관측은 RAM에서 수행하고 추가 F/B·시간을 명시한다. 기본 checkpoint 미저장 정책을 유지한다.

핵심 결과는 같은 호출 수와 실제 wall time에서의 target NLL·RS/PS/NS, 동일 edit strength에서의 locality와 최대 층 부담이다. Reference 성적과 reference에 사용하지 않은 R/P/N 평가를 분리한다. 한 층 집중률 자체나 B의 낮은 energy를 승리 조건으로 삼지 않는다. First1000 final 분모 R1000/P2000/N10000 및 기존 at-write→최종/first500→최종 retention 정의를 유지한다.

NS는 preference뿐 아니라 true/new NLL과 TF strict, lost/gained로 분해한다. 일반 보호와 과거 편집 replay의 성능도 별도로 보고한다. 과거 epoch별 누적 간섭과 현재 batch의 즉시 손실을 구분한다. 층별 D norm을 기능 기여도라고 부르지 않으며, 진단 cosine은 동일 prompt/key 기준으로 측정한다. CPU 저장 scalar만으로 층별 인과 기여를 인증하지 않는다.

## Cross effect를 목적과 진단에 반영하는 방식

학습에서는 다섯 weight를 동시에 적용한 실제 모델에서 공통 손실을 계산한다. 하층 변경이 상위층 activation을 바꾸는 효과와 서로 다른 요청이 같은 weight를 공유하는 영향은 해당 forward/backward에 포함된다. 별도 Jacobian·Hessian을 만들 필요는 없다. 다만 general16/past16 밖의 모든 이웃과 과거 사실의 locality를 직접 최적화하는 것은 아니며, 그 범위의 일반화는 독립 평가로 확인해야 한다.

Cosine 하나로 가법성 붕괴를 단정하지 않도록 작은 pilot에 한정해 다음 진단을 제안한다. 각 배치 최종 후보를 commit하기 전에 같은 entry·같은 R·같은 입력으로 entry, 각 층 하나의 write를 적용한 가상 상태5개, 전체 write 상태1개를 no-grad 평가한다. 이는 **배분 후보층을 줄이거나 별도 편집 chain을 만드는 실험이 아니라**, 이미 결정된 전체 후보의 상호작용을 분해하는 관측이다. 단일층 관측의 품질로 해당 층을 제외하지 않는다.

진단 입력은 현재 요청 최대4개, relation-matched 일반 reference 최대4개, 과거 replay 최대4개의 canonical prompt다. 모두 실행 전 ID 순서로 고정하고 성공·실패 점수로 고르지 않는다. 동일 입력·subject 위치의 L8 block output으로 d_l=h8(W_entry+ΔW_l)−h8(W_entry), d_all=h8(W_final)−h8(W_entry)를 측정한다. Cross residual은 d_all−Σ_l d_l이며 norm과 d_all norm을 함께 보고한다. 분모가 작은 normalized ratio는 NA 또는 절대량 병기하며 gate로 쓰지 않는다. Local key와 local down_proj 변위도 동일 prompt·subject 위치에서 측정해 context pooling 차이를 제거한다.

이 진단은 동일 entry에서 독립 write들의 합과 joint 결과의 차이를 본다. 단순 순차 차이의 망원합과 구분한다. Cross residual은 비가법성 관측이며 특정 층이 NS 손실의 원인이라는 증명은 아니다. 일반 reference 진단도 평가용 neighborhood 전체의 대리 성적은 아니다.

BS4×2의 두 arm 전체에서 최대 7 states×12 prompts×4 endpoints=336 prompt-state no-grad 평가다. 재사용 가능한 entry/joint 결과는 재사용하되 logical states와 실제 forward·token·wall을 따로 기록한다. Science gradient cap 밖의 명시적 진단 예산이며 전체 시간에는 포함한다. B100 반복에서는 기본적으로 끈다. R/W/H checkpoint는 저장하지 않는다.

## 구현할 함수와 산출물

새 namespace `project/run_scripts/jlz_two_arm/`에 공통 reference 준비·teacher cache·reference loss, burden geometry, common oracle, 정책 분류, 두-arm runner·reducer를 구현하는 안이다. 기존 frozen jlz_sequential 및 원자료를 덮어쓰지 않는다.

- Reference: general/past ID 선택, 충돌·평가 overlap 검사, teacher/token identity, loss 그룹 평균.
- Burden: M/s의 고정, Ω 및 gradient, η=0에서 정확한 공통 목적 복원.
- Oracle: 실제 materialization 유지, reference를 포함한 direct-R/참조 경로, component/call/time counters.
- Policy: error가 diagnostic인지 무결성인지 구분, 최종 원본 평가 예약, 유한한 미수렴 채택, trial-NaN 정책의 명시적 구현.
- Runner: 독립 W0 시작, own H/teacher/P, 전체층 eligibility, post-all-layer append, 현재·과거·미편집 관측 분리.
- Reducer: gate 때문에 특정 층/요청/불리한 결과가 사라지지 않았는지 검산하고 각 비용·분모를 보존.

동반 JSON은 실행 허가서가 아닌 machine-readable draft다. 현재 산출물은 설계 문서·설계 정합 CPU 검산·근거 목록이며, 구현 완료나 새 GPU 검증을 뜻하지 않는다.
