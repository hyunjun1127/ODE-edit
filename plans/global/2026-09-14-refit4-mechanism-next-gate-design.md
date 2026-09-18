# Six-arm 이후 REFIT4 기전 대조와 후속 순차 실험 설계

작성일: 2026-09-14. 상태: 리뷰에 근거한 구체 설계; 이 문서 작성으로 실험 실행·서버 지시 전송·GPU 제출을 수행하지 않았다.

**후속 변경:** 최신 사용자 요구에 따라 방법 선택의 주 실험은 [write-refresh 1,000요청 순차 확정 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-14-refit4-write-refresh-seq1000-final-design.md)로 옮겼다. 아래 G1의 단일 batch 대조는 기술 검사·보조 진단으로 보존하며 성능에 따른 선별 gate로 적용하지 않는다.

현재 판단은 **범위를 한정해 허용 — ALLOW_WITH_LIMITED_CLAIM**이다. 공통 Middle W50에서 B51–B60을 처리한 조건에서, REFIT4는 N4보다 신규 PS와 NS가 높고 계측 포함 online 비용은 1.223배였다. 신규 paraphrase strict 성능과 일부 old active R/P의 손실이 있으므로 전 지표 우월성·일반능력 보존·full10k 안정성까지 주장하지 않는다.

사용자의 최신 의견에 따라 **Audit128/MMLU68을 다음 기전 실험의 선행조건에서 제외한다.** 미측정 상태를 그대로 남기고 다음 단계로 진행하는 설계다. 이전 2026-09-13 문서의 audit→suffix 순서는 이 후속 범위에 적용하지 않는다. “비슷할 것”을 측정된 동등성으로 기록하지 않는다. 수치 참고선의 일괄 통과를 요구하지 않는 기존 사용자 원칙도 유지한다.

## 1. 무엇을 먼저 가를 것인가

현재의 가장 직접적인 질문은 다음 두 가지다.

1. REFIT4의 개선은 동일 entry의 native update 크기만 맞추면 설명되는가?
2. 부분 write 뒤 **첫 absolute target을 다시 fitting하는 것**으로 충분한가, 아니면 **현재 상태에서 target을 새로 최적화하는 것**이 필요한가?

이를 먼저 해결하고 다른 entry의 짧은 순차 실험으로 옮긴다. 새로운 writer를 만들기 전의 광범위한 4-cell 진단, E01 20-cell 완결, L4–L8 전수 sweep, full-functional PCG 보정은 이 단계의 선행조건이 아니다.

현재 REFIT4의 PS +0.60%p는 canonical 두 후보 간 선호의 개선이다. 신규 paraphrase의 new-target TF strict는 N4보다 18/2000 적고, paired new-NLL 악화량 p95는 1.480 nats다. 이 차이를 설명하는 실험이어야 하며 PS 한 값만을 최적화하는 설계로 바꾸지 않는다.

## 2. 근거와 재사용 경계

검토 main: `7e67befa1c73b67768d16ab98029468c945f2ac0`.
원 six-arm 실행: `5e96dcb3745977b1f273e3f5afbee61167248d49`.
원 보고: `experiment-reports/servers/server4/low-cost-write-donor-seq10-2026-09-13-v1/completed-review-v1/`.

- [이번 상세 리뷰](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-14-lowcost-seq10-review-ko.md)
- [읽은 보고·CSV·소스의 commit/SHA manifest](/mnt/raid5/janghj/ODE-edit/local/reviews/refit4-seq10-review-2026-09-14/source-manifest.json)
- [기전 대조 cell 목록](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-14-refit4-mechanism-cells.csv)
- [이전 사용자 판정 원칙](/mnt/raid5/janghj/ODE-edit/project/proposals/2026-09-13-low-cost-write-donor-pilot-gh-instruction.md)

저장된 것은 batch별 first/second target-key-readout capture, 실제 final−entry net increment, B51/B55/B60 checkpoint다. 첫 native 전체 weight, partial 전체 weight, 두 번째 subwrite delta 자체가 모두 저장된 것은 아니다. 따라서 캡처만 보고 모든 batch의 방향 분석이 즉시 가능하다고 단정하지 않는다.

우선 봉인된 writer solve를 capture의 Z/K/readout, entry P/M 및 실제 dtype으로 재구성한다. 기록된 native/partial/final hash·norm과 대조한다. 최초 capture는 완료 감사에서 현재 SHA를 봉인한 범위이며 최초 실행 시점의 이전 file SHA가 있었다고 쓰지 않는다. CPU snapshot 검증과 GPU 재개 동등성도 구분한다.

정확한 entry가 있는 B51과 B56을 먼저 사용한다. 다른 batch의 increment를 더해 만든 state는 FP32 반올림 때문에 자동으로 exact checkpoint가 되지 않는다. 재구성에 실패한 경우 그 비교만 같은 entry에서 재실행하고, 다른 유효 비교의 진행을 막지 않는다. Native first target을 다른 정책의 미래 target으로 대체하지 않는다.

## 3. G1: 두 동일 entry에서 다섯 endpoint — 다음에 수행할 주 실험

### Entry 고정

| Entry | 시작 state | 처리 batch | 용도 |
|---|---|---|---|
| E51 | 기존 공통 L4 W50/M50/prepared context/RNG | B51, ordinal [5000,5100) | 원 관측과의 연결, 공통 최초 상태 |
| E56-R | REFIT4가 B55까지 만든 W55/M55/context/RNG | B56, ordinal [5500,5600) | 반복 정책 안의 상태에서도 같은 분해가 성립하는지 |

E56-R의 N4는 **REFIT4 W55에서 시작한 native 첫 fitting**이다. 기존 N4 chain의 B56 결과는 다른 entry이므로 이 대조의 N4를 대신할 수 없다. E56-R은 결과가 좋았던 batch를 사후 선택하는 것이 아니라 보존된 B55 checkpoint를 쓰기 위한 선택이다. 두 entry는 하나의 Middle trajectory에서 나왔으므로 독립 entry/order 일반화의 증거로 세지 않는다.

### 각 entry의 다섯 endpoint

entry weight를 \(W_e\), 첫 native endpoint를 \(W_N\), native actual increment를 \(D_1=\operatorname{FP32}(W_N-W_e)\), partial endpoint를 \(S=\operatorname{FP32}(W_e+0.75D_1)\)로 둔다.

| Arm | 실제 구성 | 이 대조로 확인할 것 |
|---|---|---|
| N4 | 첫 native endpoint를 exact copy | 같은 entry의 기준 |
| S75 | S에서 종료 | 축소만 했을 때의 기준 |
| REFIT4 | S에서 fresh Z2 최적화 후 같은 L4 fitting | 현재 유망 정책의 reference |
| NM4 | 같은 D1의 크기를 REFIT4 net update와 맞춘 scalar endpoint | 단순 크기 차이로 설명 가능한가 |
| FZ4 | S에서 첫 absolute Z1을 목표로 residual 재계산·L4 fitting | fresh target 없이 repeated fitting으로 충분한가 |

총 10 endpoints다. 기존 B51의 N4/S75/REFIT4 평가와 REFIT4 B56 평가 등은 state·source·평가 identity가 맞는 범위에서 재사용한다. **새로운 10개 순차 chain이나 10회 full target optimization을 의미하지 않는다.**

각 entry 안에서는 첫 native fitting을 한 번만 계산해 다섯 분기에 공유한다. 공유 대상은 같은 entry의 첫 target/update뿐이다. E51과 E56-R 사이, 미래 batch 사이에는 공유하지 않는다. 두 fitting 사이 current history append는 0회, endpoint 확정 뒤 L4 history append는 1회다. M8 준비나 L8 수정은 필요 없다.

각 endpoint 분기마다 W/M/context/RNG를 명시적으로 복원한다. Fresh second fit은 공유 first fit 직후의 RNG와 공통 partial state에서 시작한다. 다른 endpoint가 finalize하면서 append한 history를 다음 분기의 solve에 넣지 않는다.

### NM4의 정확한 정의

REFIT4 endpoint를 \(W_R\), 그 actual net increment를 \(D_R\)라 두고 FP64 reduction으로

\[
\beta_{\rm norm}=\frac{\|D_R\|_F}{\|D_1\|_F},\qquad
W_{NM}=\operatorname{FP32}(W_e+\beta_{\rm norm}D_1)
\]

를 계산한다. materialization 뒤 실제 norm 오차를 별도로 기록한다. \(D_1=0\)이면 계수를 정의하지 않고 해당 비교를 typed invalid로 남긴다. 계수를 임의로 0.75–1 범위에 자르지 않는다.

10-batch path 비율 96.960/108.593≈0.893을 그대로 NM4 계수로 사용하지 않는다. 이 비율은 서로 다른 정책의 entry·update를 합한 값이다. 두 subwrite norm의 합을 분자로 사용하지도 않는다.

NM4는 REFIT4 결과를 읽는 **진단용 대조**다. REFIT4 비용을 뺀 online 배포 정책으로 주장할 수 없다. NM4가 REFIT4의 결과를 설명하면, 별도 개발 단계에서 scalar 하나를 고정한 뒤 새로운 구간에서 확인한다. 그때도 고정 scalar와 매 batch oracle norm-match는 서로 다른 정책이다.

보조 geometry는 다음과 같다.

\[
\beta_\parallel=\frac{\langle D_R,D_1\rangle_F}{\|D_1\|_F^2},\quad
D_\perp=D_R-\beta_\parallel D_1.
\]

batch별 \(\beta_{\rm norm}\), \(\beta_\parallel\), cosine, \(\|D_\perp\|/\|D_R\|\), 두 subwrite 및 net norm을 남긴다. \(\beta_\parallel\)와 \(\beta_{\rm norm}\)은 다르다. 비평행 성분의 존재만으로 보존에 유용한 방향이라고 결론 내리지 않고 실제 paired response와 함께 해석한다.

### FZ4의 정확한 정의

Z1은 첫 fitting이 만든 **absolute L4 block-output target**이다. 최초 delta나 최초 residual이 아니다. partial 모델의 canonical readout을 \(Y(S)\)라 하면

\[
R_{2,F}=Z_1-Y(S),\qquad
[P_4(K_SK_S^\top+M_{4,e})+I]D_{2,F}^\top=P_4K_SR_{2,F}^\top.
\]

실제 transpose/shape·dtype·solve는 실행 source를 따른다. 최종 endpoint는 \(\operatorname{FP32}(S+D_{2,F})\)다. Partial state에서 구한 K와 readout은 REFIT4 second-fit capture와 identity가 맞으면 재사용할 수 있다. 공식 P/N, Historical, audit 정답을 fitting에 넣지 않는다.

최초 residual \(Z_1-Y(W_e)\)을 두 번째 fitting에 그대로 넣는 것은 다른 실험이다. FZ4의 목적은 현재 상태에서 아직 남은 **첫 target의 residual**을 처리하는 것이다. Writer의 K는 context 평균이고 readout은 canonical prompt block output이며 regularized solve를 사용하므로, FZ4가 자동으로 0.25D1을 더하는 scalar와 같다고 가정하지 않는다.

FZ4는 두 번째 target optimization 호출이 0이다. 명시적인 frozen absolute target 입력을 사용하고 기존 case-ID z cache로 fresh/frozen mode를 혼합하지 않는다. REFIT4와 FZ4의 차이는 partial 모델에서 target·KL reference·target-init regularizer/clamp 기준과 optimizer를 다시 시작하는 효과를 포함한다. 이를 “Adam 몇 step 더 돌린 효과” 또는 “KL reset 하나의 효과”로 좁히지 않는다.

### 추가 optimization 없이 먼저 읽을 수 있는 단서

REFIT4의 second-fit 1000요청 중 Adam update 0회는 853개다. Zero-step target은 target_init이지만, target-init과 canonical readout forward의 tokenization·batch shape를 확인하기 전 residual 0을 단정하지 않는다. 또한 batch solve는 모든 K를 함께 쓰므로 zero-step 요청도 최종 write의 영향을 받는다.

기존 Z2/readout·optimizer 로그를 연결해 zero-step/positive-step 집단의 residual norm과 RHS를 기술량으로 남길 수 있다. 같은 solve matrix를 고정한 RHS 분해는 가능하지만, 그 norm 비율을 PS/NS 인과 기여율이라고 부르지 않는다. 0-step arm이나 첫 optimizer의 step sweep를 별도 필수 실험으로 늘리지 않는다.

## 4. G1 평가: 현재 품질, 기존 active 유지, 분포를 함께 읽는다

각 entry에서 N4/S75/REFIT4/NM4/FZ4를 **같은 prompt identity·분모·evaluator**로 비교한다. G1은 이미 관측된 요청의 기전 분석이므로 독립 audit라고 부르지 않는다.

| 평가 대상 | 필수 보고 |
|---|---|
| Current100 | R100/P200/N1000, 요청별 strict secondary, 양쪽 target NLL·desired margin |
| 기존 Historical128 | 고정 표본 R/P/N; active/superseded/unknown 원분모와 paired lost/gained |
| Old active 전체의 쟁점 | 기존 W60 raw에서 N4↔REFIT4 active-only 전이를 CPU 재집계; 새 G1 endpoint의 전체 prefix 평가는 쟁점이 남을 때 추가 |
| NLL 분포 | 평균·중앙값 및 paired desired-target NLL 악화량 p95/p99; 신규와 old를 분리 |
| 실제 비용 | 첫 target, 두 번째 target, 각 solve, materialization/finalize, 준비, 진단, 평가, 저장을 구분 |

NS에서 true NLL 하락과 competing-new NLL 상승을 각각 보인다. R/P에서도 new NLL·true NLL·TF strict를 함께 제시한다. “Full-seen 평균은 비슷함”을 신규 품질 유지의 증거로 대체하지 않는다. W50 또는 원 base 대비 복구는 해당 기준과의 직접 측정이 있을 때만 주장한다.

기존 six-arm에서 새 GPU 평가 없이 추가할 분석은 다음 셋이다.

ACTIVE_TARGET은 six-arm과 동일하게 관측 prefix의 마지막 raw `(subject, relation)` event와 `target_new` 문자열이 같은 요청을 뜻하며 same-target 재발행도 포함한다. E01의 정규화/latest-event 정의와 혼합하지 않는다.

1. **Old active endpoint 전이:** old4910의 N4↔REFIT4 R/P/N lost/gained. 현재 active 총점은 이미 있으며 R−1/P−6다. ALL의 R−4/P−9와 혼합하지 않는다.
2. **고정 suffix500:** N4와 REFIT4 양쪽 W55에서 성공한 동일 prompt 집합의 W60 손실·margin 변화와 전체500 요청의 변화. 공통성공 집합에서는 anchor 실패→성공이 정의되지 않으므로 회복은 전체 집합 또는 공통실패 집합에서 별도로 보고한다.
3. **노출 길이:** B51–B59의 자기 at-write→W60 변화와 미래 노출이 0인 B60을 구분한다. 같은 cohort끼리 비교한다. Conditional margin matching은 post-treatment 기술통계로만 표시한다.

기존 W55→W60 고정 suffix500 NS는 N4 3616→3578(lost140/gained102), REFIT4 3626→3626(lost107/gained107)이다. 순감소 0은 손실 0이 아니다. 추가 CPU 분석이 raw 접근 때문에 지연돼도 독립적으로 가능한 G1 비교를 막지 않는다.

불확실성을 계산한다면 두 paraphrase와 열 neighborhood를 request cluster로 묶고 batch별 차이도 보여준다. 한 fixed order의 문항 재표집을 여러 order 반복의 불확실성으로 해석하지 않는다. CI 하한 양수를 자동 허용 조건으로 만들지 않는다.

## 5. G1 결과별 방법 선택

| 관측 | 허용되는 해석과 다음 행동 |
|---|---|
| NM4가 REFIT4의 품질·보존 trade-off를 대부분 설명 | 크기 조절 설명이 강해짐. 새 방향의 고유 기여 claim을 보류하고 고정 scalar 하나의 실제 정책을 검증 |
| FZ4가 REFIT4의 주요 이득을 남김 | 첫 target의 residual 재피팅이 유용하며 fresh target의 필요성은 미확인. FZ4를 더 싼 구현 후보로 올리고 실측 비용 확인 |
| REFIT4가 NM4/FZ4보다 의미 있는 이득을 남김 | 현재 상태에서 새 target을 찾는 구성에 추가 가치가 있음. REFIT4를 다음 순차 구간의 주 후보로 유지 |
| PS/NS 이득이 strict/NLL/old 손실과 갈림 | 어느 지표·문항군의 trade-off인지 좁혀 보고하고 제한적 claim으로 다음 실험 가능 |
| 같은 entry 재현에서 핵심 차이가 사라짐 | 모든 연구를 정지하지 않고 sequential trajectory 효과 또는 재구성 차이를 가르는 비교만 추가 |

“대부분 설명”, “의미 있는 이득”을 소수문항 단일 cutoff로 자동화하지 않는다. N4·S75 대비 무엇이 회복되고 무엇이 악화됐는지 paired count·NLL·비용을 적어 GH가 판단한다. 다섯 기존 판정(ALLOW, ALLOW_WITH_LIMITED_CLAIM, NEEDS_TARGETED_CHECK, NOT_SUPPORTED, INVALID_COMPARISON)을 유지한다. 무손실·CI 양수·online 1.5배 이하를 AND gate로 복원하지 않는다.

FZ4와 REFIT4의 norm이 다르면 그 비교는 두 정책 구성의 효과를 보여준다. Fresh target의 **방향만**의 효과까지 분리했다고 쓰지 않는다. 이 더 좁은 claim이 꼭 필요할 때만 후속 norm control을 추가한다.

## 6. G2: 다른 누적 상태의 짧은 순차 검증

G1 뒤 구현 가능한 정책을 고정하고 **Late 공통 L4 W90/M90 → B91–B100**을 우선한다. 현재 우려인 old active 손실과 많은 history 아래의 행동을 직접 확인하기 위한 선택이다. 실제 W90/M90/context/RNG asset closure를 같은 host에서 확인하고, E01 replay 또는 다른 host의 native를 정확한 대조로 자동 대체하지 않는다.

- REFIT4를 유지하면 N4/REFIT4의 2 policies×10 batches=20 batch executions.
- FZ4 또는 고정 scalar를 새 단순 후보로 선택하면 N4/REFIT4/단순 후보 3 policies×10 batches=30 executions. 단순 후보는 최대 하나다.
- 모든 policy는 같은 W90에서 시작해 이후 자기 state에서 fresh 첫 z와 history를 계산한다. FZ4는 자기 batch의 첫 Z1만 두 번째 fit에 고정한다.
- 새 B100 terminal은 old9000/new1000, active/superseded/unknown, strict와 NLL을 나눈다. At-write→terminal R/P/N은 신규1000의 각 branch 자체 at-write에 대해 계산한다. Old9000는 최종 endpoint 비교로 보고하고, 직접 시계열 loss는 공통 W90 평가가 있거나 명시적으로 구분한 과거 reference가 있을 때만 계산한다.
- 이것은 **warm-entry 마지막1000개 정책 실험**이다. Terminal 분모가10000이어도 후보를 처음부터10000회 적용한 full10k 정책 실험은 아니다.

Late 자산 준비가 막히면 그 이유를 기록하고 독립적으로 준비된 Middle W60→B61–B70의 N4/REFIT4 연장을 먼저 할 수 있다. 이 경우 두 W60은 이미 서로 다른 정책 상태이므로 동일 entry의 새 개입이 아니라 기존 정책 trajectory의 연장으로 보고한다. 선택을 점수 조회 이후 유리한 구간으로 바꾸지 않는다. Early 또는 다른 order는 이후 일반화 질문에 맞춰 추가하며 여섯 arm 전부를 자동 확대하지 않는다.

Audit128은 별도 base/locality 일반화 확인이고 MMLU68은 일반능력 패널이므로 목적이 같지 않다. 둘 다 현재 미측정으로 남긴다. 최종 정책 선택 후 관련 보존·일반능력 claim을 작성할 때 평가를 배치하거나, 미측정 claim을 제외한다. Audit를 보고 재튜닝하면 그 자료는 개발 자료로 재분류한다. 미측정 audit의 예상 결과를 다음 단계의 성공으로 선기록하지 않는다.

Audit128을 실제 사용할 때는 full-seen 평가로 이미 본 문항과의 중복도 확인한다. 원 seal은 fixed10k 안에서 최초 Current B51과 Historical128의 case/known key를 제외하는 구성이다. 따라서 전용 audit 점수를 아직 집계하지 않았다는 사실만으로 이후 FullSeen6000과 모든 문항이 분리됐다고 보장할 수 없다. 이 검토에서는 실제 중복 수를 계산하지 않았으며, audit를 새 corpus의 blind test라고 부르지 않는다. 해당 확인도 G1의 진행 조건으로 추가하지 않는다.

## 7. 조건부 G-retention: 시작 상태와 미래 updater를 구분할 때만

“REFIT4로 형성된 상태가 이후 편집에 더 강하다”는 claim을 발전시키려면 기존 B55 state와 B56–60을 이용한다. G1/G2의 필수 gate는 아니다.

| B55 state origin | 이후 N4 updater | 이후 REFIT4 updater |
|---|---|---|
| N4 | 기존 NN 경로 | 새 NR 5 batches |
| REFIT4 | 새 RN 5 batches | 기존 RR 경로 |

기존 diagonal을 유효하게 재사용하면 추가 10 batches/15 fitting calls/1500 request-z다. 두 policy를 고정했다는 것은 알고리즘·정보·설정을 같게 했다는 뜻이다. Future ΔW/z를 동일 cache로 강제하지 않고 각 branch가 자기 state에서 다시 계산한다.

주 cohort는 B51–55의 동일500 요청이다. 시작 상태와 updater의 효과를 같은 cohort의 NLL/margin 변화 및 양쪽 anchor 공통성공 집합과 전체 집합에서 보고한다. 이 설계로도 단일 첫 write의 내재적 안정성을 완전히 분리했다고 쓰지 않는다. B55 old5000 전체 baseline은 미측정이므로 직접 old5000 시계열 loss가 필요할 때만 anchor 평가를 더한다.

원 checkpoint는 CPU hash/reload 검증이며 GPU continuation은 NOT_TESTED다. 동일 환경의 가벼운 재개 대조로 기존 diagonal 재사용 가능성을 확인한다. 재개가 비동등하면 영향을 받은 비교를 명시하고 필요 시 같은 재개 환경의 네 경로로 수행한다. 이를 Audit/MMLU 성능 통과 gate와 혼합하지 않는다.

## 8. 실행량·구현 범위·필수 산출물

G1에서 모든 target을 새로 계산해도 entry당 첫100+REFIT4 second100, 두 entry 합 **400 request-z**다. FZ4 second target optimization은 0이며 writer solve는 별도다. 저장 target이 검증되면 필요한 신규 optimization은 그보다 줄어든다. 보존된 endpoint의 평가를 재사용할 때도 실제 입력·state identity를 결속한다.

G1의 공통 native+fresh refit 경로만 기존 REFIT4 평균으로 환산하면 약 2×372.75=746초다. 이는 같은 host에서의 거친 warm online 참고값이며 **FZ4 solve·추가 endpoint materialization·새 평가·복원·진단·저장 비용을 포함한 전체 예산이 아니다.** FZ4가 저렴할 것으로 예상해도 아직 실측 runtime으로 쓰지 않는다. Norm oracle 계산을 이용한 연구 비용과 각 배포 정책의 비용을 따로 보고한다.

후속 구현 범위는 기존 `project/run_scripts/low_cost_write_donor_pilot/`의 adapter·runner·집계 확장이다. Frozen absolute target 입력, 임의 scalar 진단 endpoint, first/partial/final actual weights 또는 명시적 subwrite delta 보존, structured loss/Adam/early-stop counters를 추가한다. Passive 기록은 기존 수치 동작을 바꾸지 않게 하고 logger 검증을 과학 실험 반복과 구분한다. 필요한 fixture는 frozen-target residual과 실제 materialization 의미를 검증하는 데 집중한다.

GH 설계와 SH 사실 보고는 분리한다. SH에는 입력·상태·분모·source identity·실측 비용·오류·미측정 사실을 남기게 하고, 과학적 해석과 허용 claim은 global report에서 판단한다.

필수 산출물:

1. entry/source/capture 재사용 manifest와 affected comparison별 복원 수준.
2. 10 endpoint의 actual geometry, 같은 prompt의 성능·strict·NLL·paired 전이.
3. frozen-first target과 fresh target의 호출·residual·solve 기록.
4. old active 및 공통 suffix500 CPU 분석 또는 접근 불가 상태.
5. 준비/online/진단/평가/저장 비용의 분리와 claim별 판정.
6. 다음 순차 검증에 사용할 최대 한 단순 후보 또는 REFIT4의 policy lock. Audit/MMLU 상태는 DEFERRED_NOT_EVALUATED로 유지.
