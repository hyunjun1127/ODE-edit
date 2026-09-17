# 순차 local-z의 품질 제약 allocation — 실험 사양 v2

2026-09-17 · 상태: 설계 및 CPU 제어 reference 작성. 실제 model runner 미구현, GPU 제출 없음.

사용자 결정: **L4를 기준으로 한 순차 local-z 경로에서, 실제 최종 품질을 제약으로 두고 layer별 강도와 사용 layer를 선택한다.** 이번 버전은 pre-edit 출발, 별도 paraphrase target 없음, 조건부 local-z 유지, 고정 .75 배분 대체, 두 층 이상 허용을 함께 반영한다. 과거 repair 설계와 실행 중인 다른 실험은 수정하지 않는다.

기계 계약: [contract](2026-09-17-sequential-local-z-allocation-contract-v2.json). 실행 셀: [cells](2026-09-17-sequential-local-z-allocation-cells-v2.csv). 수렴 분석: [근거 설명](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-17-sequential-local-z-allocation/convergence-evidence-ko.md), [수치·출처](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-17-sequential-local-z-allocation/convergence-evidence.json).

## 1. 검증할 주장과 제외할 주장

주 질문은 동일 진입 상태의 native L4-only 편집 품질을 충족하면서, layer별 native write 강도를 연속적으로 결정해 W0-output preservation 비용을 줄일 수 있는가이다. 추가층은 target editing을 보완한다. 이미 full L4를 쓴 뒤의 preservation-only repair가 아니다.

구분할 효과는 (i) L4 강도만 조절하는 효과, (ii) L8 보완을 허용하는 효과, (iii) 기존 고정/grid보다 연속 탐색이 제공하는 효과, (iv) L5–7까지 허용하는 효과다. Layer 수 증가 자체, 균등 norm 분담, gate 합1, static보다 dynamic이 반드시 우월함을 전제하지 않는다.

이번 first1000은 기존 연구에서 사용한 개발 구간이다. 신규 cold 재실행은 warm-state 교란을 제거하지만 unseen 검증이나 모든 noise 제거를 뜻하지 않는다. 이 버전은 제한된 탐색 예산에서 얻은 controller의 개발 실험이며 저비용·전역 최적 allocation이라는 주장을 하지 않는다.

## 2. 출발 모델과 데이터

- 모델: Llama-3-8B-Instruct, revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`.
- 모든 arm은 동일 pretrained W0, zero edit history M4–M8에서 시작한다. 원지식 covariance/projector P4–P8은 유지한다. 5000-edit checkpoint는 사용하지 않는다.
- physical 0-based layers 4–8, weight는 `model.layers.{layer}.mlp.down_proj.weight`다.
- FP32, eager attention, TF32 off, 동일 tokenizer/padding/microbatch/context token bytes. Seed 20260916과 기존 cold 입력 순서를 유지한다.
- CounterFact fixed10k ordinal `[0,1000)`, B100 × 10. 각 arm은 자신의 최종 W/M/RNG를 다음 batch로 전달한다.
- Context는 기존 native context를 공통 capsule에서 재사용한다. 새 paraphrase set, paraphrase 생성, paraphrase guard는 없다.
- Current: 현재 100개 요청의 native rewrite contexts 및 canonical rewrite. Canonical의 target-new/target-old 점수는 사용 가능하다.
- Past64: 수신한 과거 fact의 latest-active canonical rewrite. 현재 overwrite되는 fact 제외. 기존 `LZ-ALLOC-v1|20260916|past|stable_event_id` SHA256 우선순위를 재사용하고 성공률로 표본을 고르지 않는다.
- Base S64: 고정 C4 token의 KL(p_W0 || p_candidate). 문서별 scored-position 평균 뒤 문서 평균. Teacher를 batch-entry 모델로 교체하지 않는다.
- Official P/N과 Dev128은 선택 봉인 후 observer만 사용한다. 미래 요청을 controller·Past 표본·conflict 필터에 넣지 않는다.

기존 Teacher192 manifest SHA `f81b798f44ce626ac1e2e402ca7438363b1dd60f5681ec0ac17b9e92d096761a`는 재사용 후보다. 실제 model/token/수치 연결은 기술 단계에서 다시 검증한다. 모델 출력이 아니라 단순 파일명만 맞는 teacher는 재사용하지 않는다.

## 3. 순차 local-z 경로

Batch entry를 We라 한다. 먼저 같은 We에서 original native L4 fit을 한 번 수행해 D4와 기준 endpoint WN을 얻는다. WN은 해당 arm의 own-entry 기준이며 독립 N4 chain의 상태가 아니다.

```text
S = exact batch entry We
D4 = native local-z4 + native solve at We  # batch 내 1회
S.W4 = native_materialize(We.W4, full_L4_endpoint, a4)
for layer in ascending enabled layers after 4:
    if a[layer] == 0 exactly:
        skip target and solve; preserve entry weight at this layer
    else:
        full_endpoint = native local-z + native solve at current S
        S.W[layer] = native_materialize(S.W[layer], full_endpoint, a[layer])
score completed S; restore We
```

Gate는 native endpoint 대비 강도이며 각 값은 `[0,1]`이다. 합1 제약은 없다. `a4=0`도 탐색 경계로 허용해 L4 필요성을 강제하지 않는다. L4를 첫 proposal 기준으로 삼는다는 뜻이며, L4가 정확히0인 후보가 선택되면 그 사실을 별도로 보고한다. L4-only 기준점은 항상 후보에 남는다.

`g=0/1`은 endpoint exact copy, 나머지는 FP32 `U + g*(V-U)`의 기존 native materialization 규약을 유지한다. 계수·FP32 materialized weight SHA를 모두 기록한다. 계수 rounding으로 새 고정 grid를 만들지 않는다.

L8 target을 batch entry에서 미리 계산하지 않는다. 앞선 gate가 달라지면 해당 prefix 이후 target/key/solve는 다시 현재 상태에서 계산한다. 동일 prefix의 같은 layer fit은 그 layer gate를 여러 값으로 바꿀 때 공유 가능하다. Cache identity는 실제 model state, physical layer, M/P, request/token/context, hparams/source/RNG를 포함한다. 계수 배열이나 case ID만으로 target을 공유하지 않는다.

L5를 추가하면 이후 L6–L8 target도 새 prefix에 맞춰야 한다. 기존 L8 write 이후 L5를 삽입하는 순서로 대체하지 않는다. 중간 prefix가 품질조건을 못 넘는다는 이유로 중단하지 않는다. 조건은 **완성된 후보**에 적용한다.

## 4. Native target solver는 이번 비교에서 고정

기존 native source의 local-z 목적·Adam·clamp·early-stop·direct solve를 유지한다. 실제 설정은 lr .1, v_weight_decay .5, clamp_norm_factor .75, kl_factor .0625, v_num_grad_steps 25, AlphaEdit ridge L2=1이다. 25는 최대 loss 평가 횟수이며 Adam은 최대24회다.

기존 LD에서 선택된 L8 active160개의 마지막 target NLL은 약 .00048, decay 약 .06214였다. 156/160은 현재 delta의 decay만으로 .05를 넘었고 144/160은 최종 norm 경계에 있었다. 따라서 cap 도달을 target 예측 부족이나 수렴 증거로 읽지 않는다. 세부 NLL/KL은 반올림 로그이며 projected-gradient/KKT 값은 미저장이다.

이번 주 비교에서 25→50/100 변경, early-stop threshold 변경, warm-start, replay target, target와 gate의 공동학습은 하지 않는다. 추가 z 계산량 정책과 allocation 효과를 섞지 않기 위함이다. 이는 25회가 최적이라는 가정이 아니다.

새 runner는 source의 수치 연산을 바꾸지 않고 request별 loss 구성·Adam 횟수·delta/radius·가능한 per-step 정보를 계측한다. 최종 gradient 계산이 추가된다면 별도 진단 비용으로 기록하고 controller에 사용하지 않는다. 후속 z 반복 연장 연구는 별도 protocol이다.

## 5. 최종 품질 제약

각 batch에서 WN을 같은 evaluator로 평가한다.

- E: native rewriting context의 target-new token→context→request 평균 NLL.
- H: Past64 canonical target-new 평균 NLL; B1에서는 비활성.
- C_strict/H_strict: canonical target-new teacher-forced 전체 token argmax 성공 ID.
- C_pair/H_pair: target-new NLL < target-old NLL인 canonical 성공 ID. Old target이 없는 요청은 해당 pair 항만 제외하고 그 수를 기록한다.
- B: S64의 W0-reference KL.

Feasible은 다음 모든 조건을 만족한 실제 endpoint다.

1. `E(candidate) <= E(WN) + 1e-4`. **max(E,.05) plateau 없음.**
2. Current strict와 pair에서 WN 성공 ID를 모두 유지.
3. Past가 있으면 `H(candidate) <= H(WN) + 1e-4`, Past strict/pair WN 성공 ID 유지.
4. 모든 수치가 finite이고 계수가 정확히 `[0,1]` 내부.

Canonical current 평균 NLL, 각 요청 NLL 무악화, paraphrase는 온라인 제약이 아니다. 이 분리는 의도적이다. 기존 후보에 모든 canonical NLL 무악화를 요구하면 N4만 남았으며, 평균 native-context 조건도 PS를 보장하지 않는다. Current canonical NLL 분포·tail·악화 ID는 반드시 보고한다.

Past 기준도 We가 아닌 WN이다. Native 자체가 잃은 과거 편집을 완전히 막는 보장은 없다. Entry→candidate의 과거 손상은 별도 측정한다.

반복 오차는 E/H 5e-5 이하, B 5e-7 이하, strict/pair ID exact 일치여야 한다. 실패하면 수치/평가 구현을 고친다. 결과를 보고 허용폭을 늘리지 않는다.

실제 feasible endpoint 중 B 최솟값 +1e-6 이내를 tie set으로 한다. 우선순위는 exact own N4, 더 적은 실제 nonzero edited layer 수, 더 작은 전체 delta Frobenius norm, gate tuple 사전순, endpoint SHA다. Tie set은 **전체 측정된 feasible 후보의 최소 B**에 대해 매번 계산하여 epsilon 손실을 반복 누적하지 않는다.

## 6. 연속 계수 탐색의 명시적 알고리즘

Optimizer는 **SciPy 1.15.3의 COBYLA**로 고정한다. 원격 기존 환경은 이번 확인에서 Python3.12.3/SciPy1.15.3이었다. PRIMA 구현으로 명명하지 않는다. 현재 환경에 설치·업그레이드를 수행하지 않았다. 실행 source/solver/dependency hash는 기술 단계에서 봉인한다.

변수는 gate 자체 대신 `u_l = 1-a_l`이다. 초기점 `u=0`은 모든 허용 layer의 full sequential native write다. 별도로 N4 incumbent를 이미 확보한다. `rhobeg=.25`의 초기 양의 좌표 probe는 유효한 감쇠 방향이므로 N4 corner에서 후속 direction≈0인 문제를 완화한다. 이 .25는 **수치 탐색 초기 반경**이고 final gate menu가 아니다. 이후 계수는 연속적으로 제안된다. 완화할 뿐 유용한 방향 발견을 보장하지 않는다.

설정: `rhobeg=.25`, `tol=.01`, `catol=1e-8`, `maxiter=64`, `disp=False`. SciPy의 maxiter는 objective 평가 상한이며 native fit 상한이 아니다. tol은 trust-region 종료 규모이며 edit 품질 허용폭이 아니다. [해당 버전 공식 문서](https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.minimize-cobyla.html).

목적은 `B / max(B(WN),1e-6)`다. Proposal용 평균 제약은 `(E(WN)+1e-4-E)/.05`, Past도 동일한 고정 scale .05를 쓴다. 여기 .05는 optimizer의 단위 스케일일 뿐 품질 plateau가 아니다. Strict는 WN 성공 ID의 최소 target-vs-best-other logit margin, pair는 최소 old-NLL minus new-NLL margin을 .05로 나눈 제약 scalar로 사용한다. Strict logit margin의 scale은1이다. 빈 집합은 +1로 비활성화한다. Strict tie·pair strict inequality는 최종 실제 ID 검사로 판정한다. Native target optimizer 안으로 미분하지 않는다.

같은 x에서 objective와 여러 constraint callback은 같은 점수를 공유한다. u→a 변환 전 raw u가 bounds 밖인지 먼저 검사한다. Raw u가 bounds 밖이면 GPU 호출 없이 finite dummy objective=1과 위반 box constraints를 반환한다. 그 점은 실험 후보/품질실패/feasible incumbent에 넣지 않는다. clipping해 다른 gate를 평가한 것처럼 기록하지 않는다.

COBYLA 최종 반환값을 곧바로 commit하지 않는다. 탐색 중 실제 평가한 feasible incumbent가 최종 선택 기준이다. Optimizer 성공 flag와 scientific 성공은 다르다. 불연속 early-stop 및 비볼록 출력 때문에 전역 최적성을 주장하지 않는다.

### 정확한 layer 사용 여부 결정

연속 최적화가 작은 양수 gate를 낼 수 있으므로 임의 threshold로0에 snap하지 않는다. 검색 후 현재 best feasible 후보에서 추가층을 **8→7→6→5 순서로 한 번씩** 정확히0으로 만든 후보를 생성하고, 필요한 suffix를 fresh 재계산해 평가한다. L4는 이 pruning 단계에서 제거하지 않는다; continuous search의 경계0은 별개다.

모든 평가 후보를 global feasible pool에 넣고 같은 selector로 incumbent를 다시 선택한다. 따라서 제거가 품질을 해치거나 global-best B tie 범위를 벗어나면 채택하지 않는다. Budget이 부족하면 완료하지 못한 제거 검사를 표시하고 현재 incumbent를 유지한다. 한 번의 greedy pass이므로 모든 support 조합의 최적 선택은 아니다. Removal 검사로 실패한 layer가 모든 다른 조합에서 필수라는 주장도 하지 않는다.

## 7. 계산 예산 — 동일 상한, 동일 실측 비용이라는 뜻 아님

연속 arm 공통 batch 예산은 아래와 같다. Native N4 fit1회/score1회는 별도 필수 비용이다.

| 항목 | Search | Pruning reserve | 합계 상한 |
|---|---:|---:|---:|
| New suffix native fits | 32 | 8 | 40 |
| New scored endpoints | 24 | 4 | 28 |
| Extra Adam updates | phase 간 공유 | phase 간 공유 | 9,600 |
| COBYLA objective calls | 64 | 해당 없음 | search64 |

한 native fit(B100)을 시작하기 전, extra Adam 잔여가 최악의 2,400회 이상인지 확인해 예약한다. 종료 후 실제 사용량만 차감하고 미사용 예약을 반환한다. Mid-target truncation으로 target 정책을 바꾸지 않는다. Candidate의 다음 suffix fit을 시작할 예산이 없으면 incomplete-budget으로 표시하고 We로 복원한다. 이미 쓴 비용과 유효한 prefix cache는 유지하지만 incomplete candidate를 점수 있는 후보로 처리하지 않는다.

Search는 pruning reserve를 사용할 수 없다. Pruning은 남은 총 예산을 사용한다. 별도 extra Adam reserve는 없으므로 target 예산 소진으로 pruning이 미완료일 수 있다. 이 경우 layer subset 탐색이 완료됐다고 쓰지 않는다.

최대 target request-calls는 batch당4,100(100 native +40×100 suffix), 최대 Adam12,000(2,400+9,600), loss평가16,100 이하, solve41, full score29다. C4는 suffixfit/추가Adam0이며 동일 scoring 상한만 적용된다. 실제 cached/zero-step 덕분에 더 작을 수 있지만 사전에 낮은 runtime 배수를 약속하지 않는다.

이 상한은 **가능성 및 계산-성능 곡선을 보는 연구용 상한**이다. Production-efficient method로 포장하지 않는다. Cost는 호출 수, 실제 Adam/loss, scored tokens, solve, history, target 시간, teacher I/O, scoring, state I/O, 전체 시간으로 분리한다. 포함 관계가 있는 timer를 합산하지 않는다. 2층·5층의 같은 상한은 cost-matched 실제 FLOPs가 아니다.

평가/fit/시간이 증가하면서 얻은 best-feasible B를 ledger로 공개한다. CPU search 횟수만으로 계산 효율을 비교하지 않는다. 거절·미완료 후보 및 N4 reference 비용도 포함한다.

Budget이 초기점 계산만 허용할 가능성을 숨기지 않는다. N4 anchor를 제외한 완료된 서로 다른 search gate vector가 d+2개 이상이고 서로 다른 a4가 포함되는지 기록한다(d는 search 변수 수). 동일 physical endpoint의 점수를 재사용한 완료 gate도 포함되므로 실제 endpoint 다양성은 별도 보고한다. 이는 initial simplex와 추가 탐색점 확보 여부를 보는 운영상의 coverage proxy이며 정확한 solver 내부 simplex 추적은 아니다. 미달이면 `INSUFFICIENT_SEARCH`를 부가 표시한다. Feasible incumbent의 실제 정책 결과는 보고하되 연속 탐색/다층 가능성을 충분히 조사했다고 쓰지 않는다. W0 기술 단계에서 C45678의 이 coverage를 확인한 뒤 과학 실행에 들어간다. 미달하면 성능 튜닝 없이 계산 상한·탐색 구현의 새 버전을 먼저 봉인한다. Lifelong 중 미달한 batch는 fallback/선택을 그대로 기록하며 같은 실행에서 예산을 늘리지 않는다.

## 8. 여섯 cold 개발 arm

| ID | 허용 layer / 정책 | 핵심 비교 |
|---|---|---|
| N4 | local4 full | 가장 강한 native 기준 |
| F48 | local4 .75→조건부 local8 .5 고정 | 실제 고정 배분 기준; 품질조건은 관측만 수행 |
| G48 | {4,8}, 기존 a4∈{.75,1}, a8∈{0,.5,1} 6후보 | v2와 같은 guard/selector로 재실행한 grid 기준 |
| C4 | local4, 연속 a4 | 여러 layer 없이 강도 조절만의 효과 |
| C48 | {4,8}, 연속 gate + exact-zero pruning | 정량적 2층 배분 |
| C45678 | {4,5,6,7,8}, 연속 gate + exact-zero pruning | 3–5층 경로까지 허용하는 주 확장 |

F48은 raw fixed endpoint를 항상 사용한다. E/H/strict/pair 조건은 observer로 기록하고 실패해도 N4로 바꾸지 않는다. 동일 진입 상태의 N4 reference는 비교용으로 계산하며 그 비용도 포함한다. 이를 통해 과거 cold7에 없었던 raw fixed(.75,.5) chain을 확보한다. C48−F48은 품질 feedback·선택 전체의 효과다. §5의 feasible commit 제약은 F48에는 적용하지 않는 명시적 baseline 예외다.

G48은 기존 LD와 target/gate family는 같지만 .05 plateau 제거와 pair guard 추가로 controller가 다르다. 기존 LD 결과를 이번 paired 수치로 재사용하지 않는다.

각 arm10batch, 총60 committed batches/6,000 arm-request 관측/unique1,000요청이다. 기술 준비와 CPU synthetic은 별도다. 기존 cold7은 motivation이며 모든 여섯 arm을 같은 capsule에서 새로 실행한다.

주 대조: C4−N4, C48−C4, C48−F48/G48, C45678−C48. C45678의 이득이 있고 실제3층 이상 nonzero 선택이 있는지를 함께 확인한다. 허용 layer가5개라는 사실만으로 5층 기여가 입증되지 않는다. C48보다 높은 실측 비용에 따른 이득은 성능-비용 곡선으로 표시한다.

이번 결과로 calibration-selected best-static보다 우월하다고 주장하지 않는다. 그 주장을 위해서는 calibration에서 고정 벡터를 봉인한 뒤 다른 요청 구간에서 모든 비교를 다시 W0부터 시작해야 한다. 이는 후속 범위이며 현재 cells에 자동 제출하지 않는다.

C48−G48에는 연속성뿐 아니라 a4의 허용 범위가 [.75,1]에서 [0,1]로 넓어진 효과도 들어 있다. 연속성만의 인과효과로 명명하지 않는다. G48보다 탐색 예산도 크므로 추가 계산의 효과를 포함하며, 동일 비용에서의 우월성은 별도 cold 비교 없이는 주장하지 않는다. 또한 C45678에서 support≥3이 관측돼도 pruning이 미완료이면 그 층들이 이득에 필요했다는 증거가 아니다. 완료된 제거 검사도 현재 조합에서의 국소 필요성만 다룬다.

## 9. Transaction·history·실패 처리

모든 후보는 같은 batch entry의 W/M/P/RNG에서 격리한다. Candidate commit 전 inner history append는0이다. 최종 selected endpoint에서만 전체 batch key로 M4–M8를 각1회 append한다. 모든 arm에서 동일하게5개의 history를 유지해 나중에 켜는 layer의 과거 보호 통계를 비워 두지 않는다. 미사용층 history 계산은 N4에도 부과되는 식별용 overhead이며 별도 비용으로 보고한다.

Gate0 요청/zero-step 요청도 history에서 제거하지 않는다. M을 gate로 가중하지 않는다. 과거 전체 key를 재인코딩하는 refresh는 도입하지 않는다. History가 새 prefix에 대해 완벽히 갱신됐다는 뜻은 아니다.

L4-only fallback은 We에 full L4 write를 적용한 WN이다. 과거 batch의 다른 layer update는 그대로 포함한다. 독립 N4 chain의 weight로 돌아가는 것이 아니다.

Budget 소진/optimizer 미수렴은 이미 평가한 feasible incumbent 선택으로 끝나는 정상 controller 상태다. NaN/Inf/OOM/shape mismatch/잘못된 cache/hash는 기술 실패이며 N4로 덮지 않는다. 실패한 batch는 미완료로 기록하고 exact entry 복원 후 원인을 수정한다. 정책/품질이 나쁘다는 이유로 학습 중 arm을 삭제하지 않는다.

## 10. 기술 단계와 산출물

GPU 실행 전 CPU reference에서 다음을 확인한다: prefix cache의 상태 의존성, extra fit·score·Adam budget 사전예약, incomplete candidate 배제와 rollback, strict/pair subset, global-best tie 기준, exact-zero pruning, commit1회. Synthetic 함수 검증은 Llama 성능 검증이 아니다.

실제 모델 기술 단계는 모두 W0에서 시작한다. (a) local4(1)↔native N4, (b) local[4,8](1,1)↔same-entry BLUE, (c) local[4..8] all1↔동일 layers의 native BLUE 분기, (d) zero gate와 state cache, (e) fixed candidate replay 순서, (f) 반복 score/teacher 연결, (g) generalized M/P physical-layer mapping을 확인한다. Adaptive optimizer의 호출 순서까지 임의로 바꿔도 같은 결과여야 한다는 잘못된 요구는 하지 않는다.

기존 NativeSingletonFitter는 layer4/8만 허용한다. 원 helper/source를 수정하지 않고 새 versioned adapter에서 4–8을 지원한다. 기존 wrapper의 finite-Adam/teacher/frame-capture semantics를 유지하고 새로운 참조 구현과 실제 runner를 혼동하지 않는다.

기술 셀 완료와 source/config/dependency/input fingerprint 봉인 이후 여섯 과학 chain을 시작한다. 이번 문서 작성은 job 제출을 포함하지 않는다.

CPU reference는 실제 SciPy1.15.3 환경에서 synthetic 목표를 대상으로 연속 계수·budget 중단·pruning을 검증했다. F2PY가 typed BudgetExceeded 전파 시 callback 오류 형태의 stderr를 출력할 수 있으므로, 해당 예외와 진짜 native 오류를 ledger에서 구별한다. 나머지 예외는 정상 종료로 삼지 않는다. 실제 모델 성능·cache parity는 아직 검증하지 않았다.

## 11. 평가·성공 기준·한계

매 batch current R/P/N을 entry와 selected endpoint에서 관측하고 문항ID로 연결한다. B5/B10에는 full-seen 및 W0-success-conditioned N retention, at-write→later forgetting, old-edit 성능을 계산한다. S64와 Dev128은 분리한다. RS/PS/NS의 두-target 선호와 TF strict를 구분한다.

Primary: native L4-only 대비 RS·PS 저하 없이 NS/Dev preservation 개선 여부. 과학적 목표의 RS/PS 허용 손실은0pp이며, 유한 개발 stream 관측으로 일반적 noninferiority를 증명했다고 쓰지 않는다. PS가 낮으면 rewrite-only/locality tradeoff로 제한해 보고한다. 공식 P/N으로 gate·guard·solver를 재튜닝하지 않는다.

Selected gate, exact support, fit/score/Adam budget 소진원인, optimizer status, pruning 미완료, feasible 비율, N4 선택원인, 요청별 E/canonical NLL/margin, layer별 delta/path/net/energy, history 진단을 남긴다. Norm 감소나 gate 감소를 남은 edit capacity의 절대량으로 해석하지 않는다.

후보별 공식 P/N은 과학 B1/B5/B10에 **선택 및 다음 상태 결정을 봉인한 뒤** 실제로 점수화한 모든 completed endpoint에 관해 observer로 기록한다. Budget 미완료/미평가 상태는 재생성해 채우지 않는다. 이 observer 비용은 controller 비용과 분리한다. Immutable candidate branch에서 평가하고 W/M/RNG를 복원하며 controller가 결과를 읽지 못하게 한다.

최소 산출물은 capsule/source manifests, endpoint별 gate와 prefix state/target/fit hash, exact E/H/B와 success IDs, native target loss 구성/counters, solver raw-call cache ledger, 비용 ledger, selected transaction/history receipts, 평가 원문항 점수, paired loss/gain 및 비용-품질 곡선이다. 로그의 반올림 값을 원수치처럼 저장하지 않는다.

추가층/연속 탐색의 이득이 관측되지 않으면 **이 품질조건·native 방향·계산 예산·탐색기에서 추가 이득을 찾지 못했다**고 판정한다. 모든 다층 방법의 불가능성이나 intrinsic capacity 고갈로 확대하지 않는다.
