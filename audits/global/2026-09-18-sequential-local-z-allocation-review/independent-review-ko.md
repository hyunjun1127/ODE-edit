# Sequential local-z allocation v2: 연구 질문, layer 분배, 품질, 장기 horizon과 비용

2026-09-18 독립 분석. 설계 v2, 봉인 실행 코드, 완료 보고서, 원문항 점수, 모든 원 과학 산출물을 대조했다. 신규 GPU 실험·재튜닝·원자료 수정은 하지 않았다.

**판정: L4를 약하게 쓰고 추가층이 부족한 편집을 보완하는 경로는 실제로 작동했다. 그러나 그 결과는 eff/gen/loc의 일괄 개선이 아니라 지표별 교환이다. 연속 탐색의 grid 대비 우월성, 3층 이상 분배의 일반적 필요성, 긴 horizon에서의 capacity 개선은 입증되지 않았다.** C45678은 이 개발 stream에서 사전 지정된 preference RS/PS와 NS/Dev 방향을 충족했지만, paraphrase strict/NLL와 계산량까지 보면 품질 무손실·효율적 다층 방법이라는 결론에는 이르지 못한다.

## 1. 무엇을 알아내려던 실험인가

설계의 핵심은 동일 진입 모델에서 native full-L4의 편집 품질을 유지하면서 W0 대비 출력 변화(S64 KL)를 줄일 수 있는가다. 현재 batch를 다룬 뒤 각 arm 자신의 W/M을 다음 batch로 넘긴다. 그러므로 최종 chain 간 비교는 정책 전체의 효과이며, B2 이후 layer 하나의 순수한 인과효과를 직접 분리하지 않는다.

| 연구 질문 | 주 비교 | 이번 관측이 답한 범위 |
|---|---|---|
| L4 강도만 낮춰도 보존 이득이 생기는가? | C4−N4 | 선택된 강도 변화는 2/10batch, 약 0.6–0.75% 감쇠에 그쳤다. 실용 이득은 작고, 탐색이 좁은 feasible 구간을 놓치는 실제 예도 있다. |
| L4를 더 낮추고 L8으로 편집을 보완할 수 있는가? | C48−C4; 같은-entry L8 제거 | 가능하다. L8을 선택한 7개 batch 모두 해당 a4에서 L8 제거 시 E 제약을 위반했다. 다만 최종 gen preference가 −0.8pp다. |
| 고정 (.75,.5)보다 품질 feedback 선택이 나은가? | C48−F48 | PS +0.35pp를 얻으면서 NS −1.54pp, RS −0.1pp, 비용 약3배. 일괄 우월성은 없다. |
| 작은 grid보다 연속 탐색이 유리한가? | C48−G48 | NS +0.82pp, PS −1.10pp, RS −0.1pp, 비용 2.45배. 연속성만의 효과는 허용 a4 범위·예산 차이 때문에 분리되지 않는다. |
| L5–L7까지 허용하면 추가 이득이 생기는가? | C45678−C48 | PS +1.0pp, NS −0.58pp, Dev KL +10.27%, 비용 +18.0%. 실제 선택은 L4/L5 중심, L6는 2회, L7/L8은 0회다. |
| 3층 이상이 필요한가? | C45678의 완료된 제거 검사 | B2는 KL 선택 때문이고, B5는 현재 a4/a5에서 L6 제거가 E를 위반했다. B5의 L5 제거는 미완료이므로 3층 전체의 필수성을 입증하지 않는다. |

F48은 품질 guard를 만족할 의무가 없는 raw fixed baseline이다. 10개 중 7개 batch가 guard를 실패한 것은 승인된 대조군 동작이다. G48은 과거 LD 결과 재사용이 아니라 이번 guard로 다시 실행한 chain이다. 모든 arm은 고정 first1000 개발 구간을 사용했으며 unseen/noninferiority 검증이 아니다.

## 2. Edit 강도와 layer allocation을 어떻게 읽어야 하는가

층별 동작은 `W_l = U_l + a_l(V_l(prefix) − U_l)`다. `V_l`은 앞층의 gate가 적용된 현재 prefix에서 새 local-z/key/native solve를 수행한 endpoint다. a=0은 fit 자체를 건너뛰고 a=1은 exact native endpoint를 쓴다. 중간값은 FP32로 materialize한다.

따라서 gate는 **각 native 방향에 대한 배율**이다. 합1의 분담률이나 layer 중요도, 남은 capacity가 아니다. a4를 바꾸면 L8 또는 L5–L8의 native 방향 자체도 바뀐다. 예를 들어 C45678 B2의 gate는 약 (.750,.751,.688)이지만 저장 full-native norm에 gate를 곱한 proxy는 L4=5.783, L5=3.272, L6=.372다. 비슷한 gate가 비슷한 weight 변경량을 뜻하지 않는다. 이 proxy도 선택된 FP32 weight의 층별 norm을 직접 재계산한 값은 아니다.

실행 코드에서는 다음을 확인했다.

- 동일 own-entry N4를 batch당 1회 fit하고 L4 anchor로 재사용한다.
- physical 4→5→6→7→8 순서 및 P4–P8 mapping이 맞다. 뒤층 target은 변경된 prefix에 조건부다.
- cache identity는 실제 state/history와 source/model/request/context/hparams namespace에 결속된다.
- raw u bounds를 먼저 검사하며 clipping된 다른 후보로 기록하지 않는다.
- E/H +1e−4, strict/pair 성공 ID subset, global feasible 최소 B+1e−6 tie selector가 설계와 일치한다.
- 최종 선택 후 모든 M4–M8에 현재 100개 key를 한 번씩 append한다. gate로 history를 가중하지 않는다.
- 공식 P/N/Dev는 선택 봉인 이후 observer다.

핵심 구현의 명확한 사양 위반은 찾지 못했다. 관련 frozen 코드: [controller.py](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-sequential-local-z-allocation-review/executed-source/controller.py:258), [runtime.py](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-sequential-local-z-allocation-review/executed-source/runtime.py:232), [metrics.py](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-sequential-local-z-allocation-review/executed-source/metrics.py:126).

## 3. 실제 선택된 분배

표는 소수 여섯 자리로 표시한다. exact 값은 [선택 CSV](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-sequential-local-z-allocation-review/report-snapshot/selected-gates.csv)에 있다. C45678의 L7/L8은 모든 batch에서 정확히0이다.

| Batch | C48 a4 | C48 a8 | C45678 a4 | a5 | a6 |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 0 | .998265 | 0 | 0 |
| 2 | .753714 | .700514 | .750000 | .750604 | .687503 |
| 3 | .979824 | .750815 | .985371 | .750428 | 0 |
| 4 | .885537 | .493961 | .911581 | .465500 | 0 |
| 5 | .822467 | .546331 | .637353 | .526817 | 1 |
| 6 | .995649 | 0 | 1 | 0 | 0 |
| 7 | .839162 | .645249 | .916008 | .763884 | 0 |
| 8 | .991890 | 0 | 1 | 0 | 0 |
| 9 | .777119 | .439650 | 1 | 0 | 0 |
| 10 | .631842 | .256924 | .711289 | .780693 | 0 |

C4는 B6=.993916, B8=.992543 외에는 full L4다. G48은 B2/B10에 (.75,.5), 나머지는 own N4다. C48의 실제 support는 1층3회/2층7회, C45678은 1층4회/2층4회/3층2회다.

### L8은 실제 편집 보완을 했는가

C48에서 L8을 선택한 B2/B3/B4/B5/B7/B9/B10 모두 해당 a4를 고정하고 L8만0으로 만들면 native E가 기준을 넘었다. B2에서는 E가 .01608185에서 .48476936으로 올라간다. B10도 선택 E=.03466130에 비해 제거 후 .25523640이다. 이는 현재 prefix와 강도에서 L8이 편집을 실제로 보완한다는 직접 관측이다. a4를 다시 올리거나 다른 gate를 탐색해도 L8이 반드시 필요하다는 뜻은 아니다.

### C45678의 3층 선택 두 건은 같은 이유가 아니다

**B2:** 선택 (.75,.750604,.687503,0,0)의 E=.01247019, B=.00313756769다. L6를 제거해도 E=.01296284로 own-N4 기준 .02939645+1e−4를 넉넉히 통과하고 성공 ID도 유지한다. 다만 B가 .00313940855로 +1.84087e−6 증가하여 tie1e−6 밖이다. L5 제거 후 L6를 새로 계산한 후보도 품질은 통과하지만 B가 +5.13367e−5 커진다. **B2의 세 층은 품질 달성의 필수 support가 아니라, 측정된 pool에서 S64를 더 줄인 선택이다.**

**B5:** 선택 (.637353,.526817,1,0,0)의 E=.01364934, B=.00830351931다. L6 제거 후보는 B가 오히려 .00828731158로 작아지지만 E=.01553756으로 own-N4 .01467522+1e−4를 위반한다. 현재 a4/a5에서 L6의 편집 보완 역할은 확인된다. 이어 L5를 제거하고 L6를 재계산하려는 검사는 extra Adam 잔여 2,329회가 사전예약 2,400회에 못 미쳐 실행하지 못했다. **현재 조합의 L6 필요성은 관측됐지만, 3층 전체의 필수성 및 최적성은 미확정이다.**

### L7/L8 선택0을 layer 불필요성으로 일반화하면 안 되는 이유

C45678의 search 완료 후보 105개는 전부 5층 gate>0이며 actual support도5였다. 작은 support는 마지막 8→7→6→5 제거 과정에서 나왔다. 연속 search가 여러 작은 support를 폭넓게 비교한 것이 아니다. 완료 search의 gate 범위도 L6 .658–1, L7 .750–1, L8 .750–1에 머물렀다.

| C45678 physical layer | Native B100 fit | target 호출 | zero-Adam target | Adam 총회수 | B100 전체 zero-Adam fit |
|---|---:|---:|---:|---:|---:|
| L4 | 10 | 1,000 | 1 | 23,976 | 0 |
| L5 | 68 | 6,800 | 5,951 (87.51%) | 20,360 | 6 |
| L6 | 78 | 7,800 | 7,740 (99.23%) | 1,431 | 55 |
| L7 | 82 | 8,200 | 8,199 (99.99%) | 24 | 81 |
| L8 | 93 | 9,300 | 9,300 (100%) | 0 | 93 |

앞층의 편집 이후 뒤층 local-z는 대개 이미 early-stop 조건을 만족한다. 이 조건부 경로에서 후반층 최적화가 거의 활성화되지 않았다는 강한 증거다. 그러나 층 순서를 바꾸거나 앞층을 더 낮췄을 때도 같은지는 측정하지 않았다.

Zero Adam이 exact zero weight write를 뜻하지도 않는다. C45678 B1 첫 L8 tensor에서 target=anchor는 exact지만, native가 별도 forward로 얻은 canonical readout과의 잔차 norm은 2.08023e−4였다. 이를 native solve가 처리해 실제 ΔW norm 5.50809e−5가 생겼다. 요청별 context forward와 B100 canonical forward의 layout 차이에서 발생한 FP32 수치차가 유력한 설명이며, 새 GPU로 원인을 분리한 것은 아니다. SHA 변화만으로 support를 세므로 이런 작은 write도 1층으로 집계된다.

L8의 모든 native fit norm은 4.38e−5–7.89e−5였다. 완료된 L7/L8 제거 14건에서 E 변화 절댓값 최대는 2.97e−8, B 변화는 1.82e−8로 매우 작았다. 따라서 이번 prefix들에서 후반 두 층의 효과가 미미했다는 해석은 가능하다. 전역적으로 L7/L8이 불필요하다거나 모델 전체의 layer 중요도를 밝혔다는 해석은 불가능하다.

### C4의 부진에는 탐색 해상도도 섞인다

B1 C4의 추가 평가 gate는 .75, .96875, .995 세 개뿐이며 모두 E 조건을 위반한다. 반면 동일 W0 진입의 C45678은 pruning으로 L4-only .9982652821을 실제 평가해 E=.00346141724, B=.00171474040을 얻었다. 이는 같은 N4의 E=.00340461437/B=.00172090799에 대해 feasible이고 KL도 6.16760e−6 낮다. **같은 문제의 L4-only 개선점이 실제로 존재했으나 C4가 찾지 못했다.** tol=.01과 엄격한 E 허용폭이 결합된 이번 탐색 결과를 강도 조절 자체의 한계로 확대하면 안 된다.

![선택, 실제 계산, 누적 drift, 동일 cohort locality](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-sequential-local-z-allocation-review/allocation-and-horizon.png)

## 4. Eff / Gen / Loc 관계

여기서 eff=RS, gen=PS, loc=NS이며 모두 두 target의 teacher-forced NLL 선호 지표다. TF-strict와 자유 생성 정확도는 별개다. Dev128은 W0 대비 generic KL의 observer이며 작을수록 좋다.

| Arm | Eff RS% | Gen PS% | Loc NS% | P TF-strict% | Dev128 KL | 전체 실행시간 /N4 |
|---|---:|---:|---:|---:|---:|---:|
| N4 | 99.9 | 96.7 | 80.26 | 67.70 | .02265189 | 1.00× |
| F48 | 100.0 | 95.45 | 83.24 | 64.00 | .01364659 | 1.22× |
| G48 | 100.0 | 96.9 | 80.88 | 66.85 | .01933609 | 1.50× |
| C4 | 99.8 | 96.6 | 80.46 | 67.45 | .02255923 | 1.16× |
| C48 | 99.9 | 95.8 | 81.70 | 65.65 | .01758247 | 3.68× |
| C45678 | 100.0 | 96.8 | 81.12 | 65.90 | .01938770 | 4.35× |

**관측된 관계는 L4를 낮추고 추가층으로 rewrite를 보완하면 RS를 유지하면서 출력 변화와 일부 locality 손상을 줄일 수 있지만, paraphrase까지 같은 강도로 보존되지는 않는다는 것이다.** F48은 locality/Dev가 가장 좋고 paraphrase가 가장 약하다. C48은 더 엄격한 current/Past 보호로 F48보다 P를 일부 회복하지만, F48의 locality 이득도 상당 부분 잃는다. C45678은 C48보다 P preference를 회복하는 대신 locality/Dev가 나빠진다. 총 변화량이나 a4의 단일 축으로 모든 성능을 설명할 수 없다.

C45678의 PS +.1pp는 N4에서 P30개를 잃고32개를 얻은 순증가다. P strict는120개를 잃고84개를 얻어 −1.8pp이고, 두 P가 모두 strict인 요청은513→481로 −3.2pp다. P target-new NLL도 평균 +.074610, p99 +4.152845 악화한다. G48도 preference PS는 +.2pp지만 P strict는 −.85pp다. **Primary로 정한 preference 기준의 개발 관측 성공과 더 강한 의미의 generalization 보존을 구분해야 한다.**

현재 선택 제약이 보호하는 것은 native-context 평균 E, canonical 성공 ID, Past64의 H/성공 ID다. 각 요청 canonical NLL, paraphrase, 공식 N, 전체 과거 요청은 직접 보호하지 않는다. 이는 실제로 다음 결과를 허용했다.

| 같은-entry own N4→selected | ΔE | ΔS64 KL | canonical NLL >1e−4 악화 | PS 변화 | NS 변화 |
|---|---:|---:|---:|---:|---:|
| C48 B5 | −.00514683 | −.00037600 | 91/100 | −1.0pp | −.8pp |
| C48 B10 | −.00060285 | −.00096355 | 85/100 | −4.0pp | +1.1pp |
| C45678 B5 | −.00102587 | −.00063353 | 82/100 | −1.5pp | −.1pp |

특히 C48 B5는 E와 S64가 함께 좋아지고 guard를 통과하지만 P와 N이 함께 나빠진다. **S64 최소화가 공식 locality까지 일관되게 개선한다는 가정은 저장된 같은-entry 반례로 성립하지 않는다.** 이 결과를 보고 온라인 정책을 다시 조정한 흔적은 없다.

C48−G48의 PS −1.10pp 기술적 case-cluster CI는 [−1.85,−.35], NS +.82pp는 [+.39,+1.23]다. C45678−G48의 NS +.24pp CI는 [−.22,+.68]다. 다른 seed/order 반복이 없는 한 이러한 행 재표집 CI는 단일 stream 내부 요약이며 일반적 우월성 인증이 아니다. 상세 원시 재검산은 [성능 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-sequential-local-z-allocation-review/metrics-audit-ko.md)에 있다.

## 5. Horizon이 길어지면 무엇을 예상할 수 있는가

### 실제 관측: 편집 성공 유지와 locality 누적 손상이 분리된다

처음500개라는 동일 집단을 W5와 W10에서 비교하면 다음과 같다. 다른 cohort의 난이도를 섞지 않은 비교다.

| Arm | 동일 first500 NS: W5→W10 | 변화 | 동일 first500 PS: W5→W10 | 전체 at-write→W10 NS 순손실 |
|---|---|---:|---|---:|
| N4 | 84.06→79.62 | −4.44pp | 96.1→95.8 | −3.53pp |
| F48 | 85.62→82.82 | −2.80pp | 93.9→94.0 | −2.05pp |
| G48 | 84.04→80.30 | −3.74pp | 95.6→95.9 | −2.99pp |
| C4 | 84.06→79.74 | −4.32pp | 96.1→95.9 | −3.36pp |
| C48 | 84.34→81.02 | −3.32pp | 95.0→95.1 | −2.51pp |
| C45678 | 84.04→80.48 | −3.56pp | 95.9→96.3 | −2.84pp |

C48의 최종 NS +1.44pp 중 at-write부터 있던 격차는 +.42pp, 이후 순손실 감소분은 +1.02pp다. C45678의 +.86pp는 각각 +.17/+.69pp다. 이는 관측 차이의 산술 분해이며 두 원인의 독립 인과효과 추정은 아니다. 추가층은 현재 요청 neighborhood의 손상뿐 아니라 이후 누적 손상을 줄이는 방향도 보여줬다. 하지만 모든 arm에서 locality는 계속 떨어졌다.

C48 PS는 at-write95.8→W10 95.8이다. 따라서 이 구간의 낮은 gen을 대부분 장기 forgetting으로 설명하기 어렵고, 작성 시점의 일반화 손실이 먼저 발생한 것으로 보인다. RS는 포화에 가까워 1,000회에서 장기 차이를 구별하는 감도가 낮다. 현재 high RS만으로 수천·수만 회까지 안정적이라고 예측할 수 없다.

S64는 모든 arm에서 매 batch 단조 증가했다. 후반500회 증가량은 N4 +.01096647, F48 +.00579001, G48 +.01002002, C4 +.01107367, C48 +.00749482, C45678 +.00933899다. 출력 drift가 멈추거나 일정 수준에서 포화된 증거는 없다. C45678은 B5까지 C48보다 S64가 작았으나 B6에 순위가 바뀌었고, B10에는 더 크다. Dev에서도 C45678이 G48보다 좋던 B5 순위가 B10에 뒤집힌다.

### 장기 보장을 약하게 만드는 세 구조

1. **Greedy 기준은 own-entry N4다.** 매번 그 reference보다 좋다고 독립 N4 chain보다 최종 상태가 좋다는 보장은 없다. C4는 B6/B8에서 own-N4의 B를 줄였지만 W10 S64는 독립 N4보다 .00010720 높아졌다. 미래 효과를 최적화한 policy가 아니다.
2. **Past64는 과거 전체의 출력 guard가 아니다.** B2–B10의576 sample slots는185개의 고유 요청이다. eligible로 등장한900개 중715개는 이 직접 guard에 한 번도 들어오지 않았다. B10은 현재 overwrite 제외 eligible899개 중64개(7.12%)를 검사한다. 고정 SHA priority bottom64여서 매번 독립 표본을 뽑는 것도 아니다. 5,000/10,000 규모에서 표본 수를 그대로 두면 한 batch 직접 보호 비율은 대략1.3%/.65%로 더 작아진다.
3. **M 전체 누적은 의미적 성공 보장과 다르다.** 모든 요청 key의 보호 통계는 들어가지만 과거 key를 새 prefix에 맞춰 재인코딩하지 않는다. history 값은 계속 변하고 native solve의 방향/조건도 달라질 수 있다. H 역시 entry가 아닌 own-N4 기준이므로 native가 이미 잃은 과거 성공을 되살리는 조건은 아니다.

### 가능한 예상과 아직 말할 수 없는 것

유사한 요청 분포와 지금의 편집 동작이 유지된다면, F48/C48처럼 L4를 낮춘 경로가 N4보다 locality 손상을 늦추는 경향은 계속될 가능성이 있다. 현재 gen 손실도 함께 고려해야 한다. 이는 기전과500→1000 관측에 근거한 조건부 예상이며, 5k/10k 성능 수치는 계산할 수 없다.

Horizon 증가로 native reference 자체의 품질이 낮아지거나 과거 보호 조건이 더 자주 걸리면, feasible allocation이 줄거나 own-N4 선택이 늘 수 있다. 반대로 앞층만으로 target을 충분히 만족하지 못하게 되면 지금 거의 zero-step인 후반층이 활성화될 수도 있다. 현재10batch의 support는 증가 추세가 없고 C45678은 B6/B8/B9에 own-N4를 선택했다. 따라서 “오래 갈수록 더 많은 layer가 자동으로 필요해진다”는 예측은 지금 결과에서 나오지 않는다.

장기 검증에서는 새 cold 구간·순서 반복과 함께 같은 cohort의 RS/PS/NS, P strict/NLL tail, W0-conditioned NS, 전체 active 과거 성공, own-N4 품질, suffix zero-step 비율, 실패 guard 종류, support/비용 곡선을 같이 봐야 한다. 현재 개발 결과에 맞춰 P/N으로 guard를 조정한 뒤 같은1000에 재검증하면 독립 검증이 되지 않는다.

## 6. 연산량: 무엇을 더 계산했고, 얻은 이득에 비해 얼마나 비싼가

| Arm | target 호출 | Adam | loss 평가 | native solve | online full score | GPU 할당 시간 | program/N4 | controller/N4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N4 | 1,000 | 24,000 | 25,000 | 10 | 10 | 1.261h | 1.00× | 1.00× |
| F48 | 2,000 | 27,912 | 29,912 | 20 | 20 | 1.544h | 1.22× | 1.28× |
| G48 | 3,000 | 28,152 | 31,152 | 30 | 60 | 1.895h | 1.50× | 1.61× |
| C4 | 1,000 | 24,000 | 25,000 | 10 | 40 | 1.466h | 1.16× | 1.17× |
| C48 | 15,100 | 65,759 | 80,859 | 151 | 174 | 4.641h | 3.68× | 4.59× |
| C45678 | 33,100 | 45,791 | 78,891 | 331 | 142 | 5.477h | 4.35× | 5.47× |

여섯 과학 run 합계58,623 GPU초=16.284h다. 기술 준비3,379초=.939h, 과거 teacher98초는 별도다. 전체 program에는 observer/history/상태관리 등이 포함된다. controller도 pure model FLOP가 아니며 source에 기록된 타이머 경계를 따른다. N4에도 비교를 위해 미사용5층 history 통계를 유지하는 비용이 부과돼 있으므로 일반적인 단층 native editor의 production 속도로 읽을 수 없다.

### 5층 arm이 느린 이유

C45678은 C48보다 target 호출이2.19배인데 Adam은.696배, loss평가는.976배다. 그럼에도 전체시간은1.18배다. zero-step이어도 forward, key/readout, solve, state hash/cache, capture/기록, score가 남는다. Adam 횟수만으로 계산량을 평가할 수 없다.

L7/L8의175개 fit은 C45678 suffix321개 중 **54.5%**다. 두 층의 native-inclusive receipt 시간 합은3,032초로 전체 program의15.4%이며, 이 수치에 겹치는 target/solve 시간을 다시 더하지 않았다. 최종 선택은 두 층 모두0이다. 이번 설정에서 큰 탐색 예산이 기여가 작은 후반층 처리에 쓰인 셈이다.

Search32 fit 한도에서 a4가 달라지면5층 arm은 suffix최대4개를 재계산하고, a5 변화는3개, a6는2개, a7은1개, a8은0개다. C48의 a4 변화는1개다. 따라서 동일 fit 상한은 동일 탐색 기회나 실제 FLOP가 아니다. C45678은 전10batch가 SUFFIX_FIT_CAP로 끝났고 완료 search vector는10–12개뿐이다. count proxy를 통과했다고5차원 탐색이 충분하거나 수렴했다고 볼 수 없다.

Pruning용 fit8/score4 reserve는 있지만 Adam 전용 reserve는 없다. B5가 최종 layer 제거를 못 마친 이유가 이 구조다. 학습량이 적었던 다른 batch에서 L7/L8을 빨리 제거할 수 있었던 사실과 구분해야 한다.

### 실제 타이머가 알려주는 것

C48/C45678의 controller-inclusive는14,574/17,344초, finalize는904/1,067초, post-seal 후보 observer는595/669초다. 원시 fit receipt들의 inclusive 합은9,940/10,915초, online scoring은3,172/3,394초다. 서로 다른 계층의 inclusive 타이머를 합쳐 총시간이라고 부르면 중복된다.

Teacher read 시간은 C48 488초, C45678 1,223초로 scoring에 포함된다. C45678은 online score가 더 적어도 scoring wall이 더 길었다. 저장 경로/I/O 등 실행조건이 비용에 섞여 있으므로 wall 차이를 algorithm FLOP 차이와 동일시할 수 없다. Pure writer/state-I/O는 분리되지 않았고 official observer의 exact forward/token 계수도 누락돼 있다. 따라서 총 run 비용과 기록된 online 호출 수는 비교 가능하지만 완전한 FLOP-matched 비교는 불가능하다.

Online full-score1회도 값비싼 평가다. 본 데이터에서 B1은178회, 이후186회 forward이며, native-context E100회/current canonical14회/Past canonical8회/S64 64회가 포함된다. S64 teacher는 score마다64×128×128256 FP32로 약3.914GiB의 논리적 tensor 복사·처리를 요구한다. 실제 disk I/O는 OS cache에 따라 달라지므로 이 값을 물리 disk-read 실측으로 간주하지 않는다.

N4/C4 L4 target은 전부24Adam cap, 나머지4arm은 각각999/1000 cap+1zero-step다. L4 전체6,000개 중5,996개가 cap이다. 마지막 target NLL은 이미 약 .001 수준이지만 decay가 약 .12라 total-loss .05 early-stop을 넘는다. cap을 target 학습 부족/수렴/capacity 고갈로 해석할 수 없고, 이를 이유로 더 긴 target optimization이 필요하다고 바로 결론내릴 수 없다.

![실제 측정된 best-feasible KL 대 suffix fit](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-sequential-local-z-allocation-review/measured-search-frontier.png)

그림은 각 batch의 own-entry N4 대비 실제 feasible 최소B의 개선이다. 거절·미완료에 쓰인 마지막 fit 비용도 가로축에 포함한다. N4 필수 fit은 별도이며 arm들의 entry가 다르므로 곡선을 같은 목적함수의 solver 우열로 읽지 않는다. 많은 batch는 후반 fit에서 개선이 거의 없지만, B7/B10처럼 늦게 개선점이 나온 경우도 있어 단순 조기 종료의 무손실 효과를 입증하지 않는다.

### 긴 horizon의 계산 구조와 비용 시나리오

B100와 후보 상한을 고정하면 batch당 online 입력은 current100/Past64/S64로 고정된다. Cache는 batch마다 새로 만들고 폐기한다. M/P도 각 layer `[1,14336,14336]` 고정 shape이며5M=3.828GiB,5M+5P=7.656GiB다. 따라서 **horizon 때문에 dense solve 차원이나 기본 M 메모리가 선형 증가하지 않는다.** 값과 조건, 실제 fit 횟수/Adam 소모는 바뀔 수 있다. GPU peak allocated는 모든 arm33.48GiB였으며 model 전체, 기타 상태까지 포함한 관측값이다.

아래는 지금의1000회 평균 비용을 단순5배/10배 한 **예산 감각용 시나리오**다. 성능이나 실제 소요시간 예측, 상한 보장이 아니다.

| Arm | 관측1000 GPUh | 5000 선형 환산 | 10000 선형 환산 |
|---|---:|---:|---:|
| N4 | 1.261 | 6.307 | 12.614 |
| F48 | 1.544 | 7.722 | 15.444 |
| G48 | 1.895 | 9.474 | 18.947 |
| C4 | 1.466 | 7.332 | 14.664 |
| C48 | 4.641 | 23.203 | 46.406 |
| C45678 | 5.477 | 27.383 | 54.767 |
| 여섯 군 합 | 16.284 | 81.421 | 162.842 |

Full-seen을5batch마다 계속 평가하면 별도 observer는 선형이 아니다. 평가되는 누적 request 수 합은10batch에서500+1000=1500,50batch에서27,500,100batch에서105,000으로 각각18.3배/70배가 된다. 이 부분은 대략 horizon 제곱으로 증가하므로 위 단순 환산에 주의해야 한다. 최종 full-seen만 수행하거나 평가 시점을 고정하면 비용 구조가 달라지며 이는 새 protocol 결정이다.

연속 arm의 batch당 계약 상한은 필수N4를 포함해 target4,100/Adam12,000/loss16,100/solve41/full-score29다. 10,000회=100batch라면 arm당 target410,000/Adam1,200,000/solve4,100/full-score2,900 상한이다. 시간 상한은 하드웨어/I/O/문맥 길이 등에 의존해 이 계수만으로 정해지지 않는다. Raw9.92GB 역시 동일 기록량을 가정하면10배 약99.2GB 규모지만, 실제 fit 수와 full-seen 정책에 따라 달라진다.

현재 runner는1000/10batch와 B5/B10 observer를 고정한 코드다. 장기 검증은 실행 숫자만 몰래 늘리는 continuation이 아니라 horizon·평가시점·예산을 새로 봉인한 cold 실험으로 해야 비교가 성립한다. 세부 코드 근거는 [horizon/비용 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-sequential-local-z-allocation-review/horizon-cost-code-ko.md)에 정리한다.

## 7. 산출물 신뢰도와 남은 결손

이번 독립 감사에서 원 과학 output **9,696파일/9,922,229,924B 전량 SHA**, native **552파일/55,200targets 전량 CPU tensor 검산**, 원시 W10 **78,000 평가행**, online score **446개**, 선택60개, commit60개/인접link54개/history append300개를 확인했다. 재집계 불일치를 찾지 못했다. Frozen CPU 테스트77개도 통과했다. 이는 저장 증거와 source의 일관성 검증이며 새 model forward 재현은 아니다.

기술 단계에는 original BLUE5층과 singleton path의 exact weight parity 증거가 있고, 이번에 그 source/receipt를 검산했다. Teacher192는 manifest/schema/stat에 연결되어 있으며 실제 W0 numerical reproduction은 S64 64문서 범위다. Dev128까지 새 W0 forward로 검산됐다고 확대하지 않는다.

**실제 산출물 요구 미충족은 selected FP32의 layer별 ΔW, 누적 path/net/energy가 없다는 것이다.** Native full-fit norm과 전체 concat action_norm은 있으나, 이들로 실제 층별 에너지 분배나 남은 edit capacity를 확정할 수 없다. No W/M disk checkpoint는 동결 지시에서 허용된 정책이다. 다만 실행 중 층별 scalar를 기록할 수 있었으므로 checkpoint 미보존이 해당 진단 누락을 정당화하지는 않는다.

원 보고서는 숫자를 대체로 정확히 정리하고 한계도 공개했다. 보완할 부분은 '기술적 완료'와 '원래 연구 질문의 답'을 연결하는 해석이다. **이번 결과가 지지하는 것은 조건부 보완의 가능성과 rewrite/generalization/locality 사이의 교환, 그리고 상당한 탐색 비용이다.** 최적 layer 분배, 연속 탐색 우월성, 다층 capacity 증가, 장기 품질 무손실을 지지하는 수준은 아니다.

## 8. 이 결과를 다음 판단에 쓰는 순서

1. **가장 먼저 구분할 가설:** 추가층 이득인지, a4 약화 이득인지, 탐색기가 좁은 feasible 영역을 못 찾은 것인지. B1 C4 반례가 있으므로 C4를 완벽한 강도조절 oracle로 취급하지 않는다.
2. **다층 필요성:** C45678의 선택 형태는 실제로 L4/L5 중심이다. 고정된 예산의 {4,5}/{4,6}/{4,8} 및3층 비교나 동일-entry support 재최적화가 없어서 L5 자체의 우월성과 추가 계산의 효과를 분리할 수 없다. 현재 pruning은 gate를 다시 최적화하지 않는 국소 제거 검사다.
3. **목표 품질:** preference PS가 거의 유지돼도 strict/NLL가 낮아지는 것을 허용할지 먼저 명확히 해야 한다. 이번 공식 P/N을 보고 현재 실험 guard를 사후 고치는 것은 검증 결과를 바꾸는 일이므로 후속 독립 protocol로 분리한다.
4. **효율:** G48은 C45678보다 약2.9배 저렴하면서 PS/Dev는 조금 더 좋고 NS만 .24pp 낮다. 다층 확장을 주장하려면 이 비용 대비 차이를 넘어서는 반복·장기 근거가 필요하다. Zero-step suffix를 단순 생략하면 native exact semantics가 달라질 수 있어 별도의 parity/정책 검증이 필요하다.
5. **장기성:** 현재1000개에서 high RS보다 먼저 드러나는 문제는 locality 누적 하락과 gen의 즉시 약화다. 이를 같은 cohort에서 추적하고, Past64 밖의 전체 과거 성공과 비용 증가를 같이 측정해야 한다.

근거 묶음: [성능·guard 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-sequential-local-z-allocation-review/metrics-audit-ko.md), [전량 산출물 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-sequential-local-z-allocation-review/artifact-audit-ko.md), [raw layer ledger 재집계](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-sequential-local-z-allocation-review/layer-ledger-evidence.json), [원 완료 보고서 사본](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-sequential-local-z-allocation-review/report-snapshot/diagnostic-report-ko.md).
