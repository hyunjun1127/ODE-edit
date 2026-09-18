# F48–CAKE 후속 방향: 과거 완료 실험이 식별한 것

2026-09-18. 과거 완료 보고서·독립 감사·보존 source 설명을 읽어 현재 F48 해석에 연결했다. 새로운 GPU 실험, 모델 forward, 학습, 원자료 변경은 없다. 현재 v2 결과는 이미 완료한 독립 검산을 연결하되 여기서 중복 재집계하지 않았다.

## 핵심 판단

**F48의 locality 개선은 강한 연구 신호지만, “고정 heuristic이 좋은 layer allocation을 찾았다”만으로 설명되지 않는다.** 과거 결과는 (i) 단순한 L4 감쇠만으로 locality–generalization 교환이 생기며, (ii) 추가층 없이 같은 L4의 fresh refit으로도 preservation이 달라지고, (iii) 같은 gate라도 local target과 terminal target의 성능이 크게 갈리며, (iv) quality guard가 유지하는 지표의 선택이 결과를 바꾼다는 것을 각각 보여준다.

따라서 후속 연구의 가장 싼 식별 질문은 **“static인가 dynamic인가”보다, 같은 진입 상태에서 L4를 약하게 쓴 뒤 어떤 보완 방향이 동일 편집 품질을 가장 작은 부작용으로 복구하는가**다. F48은 강한 저비용 기준으로 남기고, 그보다 복잡한 방법에는 명확한 추가 효과를 요구하는 것이 맞다.

CAKE와 F48의 차이를 static 대 dynamic으로 설명하면 분류부터 틀린다. 둘 다 고정 배율을 사용할 수 있지만 CAKE는 entry의 terminal L8 target을 공유하고, F48은 partial-L4 뒤 L8 local target을 fresh 계산한다. CAKE도 앞층 write 이후 현재 residual/key를 다시 읽으므로 전체 계산이 상태에 무관한 static update인 것은 아니다.

## 1. 비교 단위와 혼합 금지

| 기록 | 실제 완료 범위 | 이번 해석에서의 역할 |
|---|---|---|
| 2026-09-13 low-cost six-arm | 공통 L4 W50/M50에서 ordinal5000–5999, B51–B60 | 강도만 줄인 S875/S75, 다른-layer FULL8/RES8, 같은-layer REFIT4 비교 |
| 2026-09-14 write-refresh | 같은 W50 suffix, 기존N4/REFIT4 재사용+신규4chain | fresh native target reset과 frozen/carried target 반복의 차이 |
| 2026-09-16 cold7 | W0/zeroM, first1000, 7개의10batch chain | warm 출발 없이 local/terminal 및 단순 강도/두층 선택 비교 |
| 2026-09-15 EP-TW-1 완료 | W0 first1000, 신규1chain | native endpoint 주변 작은 보존 보정의 국소 효과와 한계 |
| 2026-09-16 SL-ZFlow 완료 | W0 first1000, 신규1chain, 과거N4 재사용 | actual-write feedback objective/trajectory 변화의 결과 |
| B-OS Middle 부분 recall | 공통 Middle entry의 단일 correction endpoint | full-L4 후 L8 repair의 finite-step/비용 실패 사례 |
| CAKE native | W0 fixed10k, B100×100, 신규1chain | 실제 CAKE whole-policy 성능과 설정 차이 |

서로 다른 row의 성능을 한 matched-cold 표로 합치지 않는다. 특히 warm suffix의 N4 NS71.08%, cold7/v2 N4 NS80.26%, 과거N4 first1000 NS80.72%는 서로 다른 실행·출발점이다. W50→W60의 새1000 결과를 W0→W10 결과라고 쓰면 안 된다. 전체6000과 새1000의 분모도 분리했다.

## 2. L4 강도 감소만으로 locality 교환은 이미 나타났다

[Low-cost six-arm 독립 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-14-lowcost-seq10-review-ko.md)의 신규 suffix1000 비교다. ΔP의 분모는2000, ΔN은10000이다.

| 정책 | 구성 | 신규 ΔR/ΔP/ΔN count 대 N4 | 신규 ΔPS/ΔNS pp |
|---|---|---:|---:|
| S875 | L4 native write ×.875 | −2/−14/+26 | −0.70/+0.26 |
| S75 | L4 native write ×.75 | −4/−42/+80 | −2.10/+0.80 |
| FULL8 | full L4 뒤 fresh L8 | −1/+15/−1 | +0.75/−0.01 |
| RES8 | 감쇠 L4 뒤 L8 보완 | −1/+22/+12 | +1.10/+0.12 |
| REFIT4 | L4 .75 뒤 같은L4 fresh refit | −1/+12/+67 | +0.60/+0.67 |

이 기록은 **단순한 감쇠만으로도 N을 높이고 P를 낮출 수 있음**을 직접 보여준다. 따라서 현재 F48의 NS gain 전부를 “둘째 층으로 분배한 효과”로 해석할 수 없다. 추가 L8 없이 REFIT4도 NS를 높였으므로 추가층 자체가 유일한 보존 기전도 아니다.

반면 FULL8은 신규 NS가 사실상 변하지 않았다. 이 정책은 full-L4 뒤 다른층을 더하는 형태이며 L4 부담을 줄이는 정책이 아니다. “두 층이므로 locality가 좋아진다”는 일반화를 지지하지 않는다.

REFIT4 online은 N4 대비1.2226배였다. 하지만 P preference +0.60pp와 동시에 P strict는1423→1405/2000(−0.90pp), 두 P 모두 strict는569→542/1000이었다. 과거 active R/P도 net−1/−6이 남았다. 동일한 preference/strict 분리가 현재 F48/C45678만의 특이 현상은 아니다.

이 실험에서도 REFIT4의 batch norm 합/N4 비율 .8929는 서로 다른 trajectory 집계였다. 동일-entry scalar(.8929) 대조는 없었으므로 “새 방향이라서 이겼다”는 식별은 당시에도 남아 있었다.

## 3. Fresh target reset은 frozen target을 더 실현하는 것과 다르다

[Write-refresh 완료 보고서](/mnt/raid5/janghj/ODE-edit/local/reviews/bg-tw-method-review-2026-09-15/source/experiment-reports/servers/server4/refit4-write-refresh-seq1000-2026-09-14-v1/completed-review-v1/diagnostic-report-ko.md)는 공통 W50/M50에서 같은 L4만 수정한 비교다.

| 정책 | 신규 PS /2000 | 신규 P strict /2000 | 신규 NS /10000 | Online/N4 |
|---|---:|---:|---:|---:|
| N4 | 1938 (96.90%) | 1423 | 7108 (71.08%) | 1.000 |
| REFIT4 | 1950 (97.50%) | 1405 | 7175 (71.75%) | 1.223 |
| FROZEN2 | 1958 (97.90%) | 1467 | 7070 (70.70%) | 1.067 |
| I2 | 1961 (98.05%) | 1456 | 7086 (70.86%) | 1.142 |
| FROZEN4 | 1969 (98.45%) | 1487 | 7056 (70.56%) | 1.193 |
| I4 | 1966 (98.30%) | 1489 | 7056 (70.56%) | 1.276 |

FROZEN은 최초 absolute z를 고정하고 중간 write 이후 residual을 다시 fit한다. I2/I4는 초기 anchor/teacher/clamp와 optimizer state를 유지하면서 target 최적화를 나눈다. REFIT4는 변경된 모델에서 native target 최적화를 새로 시작한다. 이 비교에서 target을 반복적으로 실현하는 방법은 P를 높이고 N을 낮췄으며, fresh refit은 다른 균형을 보였다.

다만 REFIT4는 target/optimizer/anchor/teacher/clamp 기준을 함께 초기화한다. Fresh gradient 하나, teacher reset 하나, ODE 하나의 인과효과를 식별한 비교는 아니다.

**Key refresh가 이득의 원인이라는 설명은 지지되지 않는다.** 같은 L4 down-projection만 변경하는 이 비교에서 고정 token/context의 해당 층 입력 key는 구조적으로 변하지 않는다. [후속 source·공개 hash 정리](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-16-refit-feedback-barrier-write-proposal.md)에서도 6개 정책의 각 batch entry/endpoint M4 hash가 같았다. 반복 과정에서 readout·target·regularization 기준이 변하는 것과 key 정보가 갱신되는 것을 혼동하지 않아야 한다.

REFIT4의 두 번째 target 1,000개 중 853개는 zero-Adam, 146개는 24 Adam, 1개는 21 Adam이었다. Zero-step에서도 현재 activation을 absolute target으로 반환하므로 최초 z를 재사용하는 연산은 아니다. 이미 충분한 rewrite에 대해 추가 confidence를 추구하는 정도를 줄일 가능성이 있다. 다만 zero-step과 zero-weight-write는 같은 뜻이 아니다.

## 4. Cold7은 warm-state 설명을 줄이고 target family의 차이를 보였다

[Cold7 독립 리뷰](/mnt/raid5/janghj/.codex/worktrees/odeedit-local-z-independent-audit-20260917/experiment-reports/global/2026-09-17-local-z-seven-arm-independent-review.md)의 모든 arm은 새로운 W0/zeroM에서 시작한다.

| Arm | RS% | PS% | NS% | 주요 의미 |
|---|---:|---:|---:|---|
| N4 | 99.90 | 96.70 | 80.26 | full local L4 |
| REFIT4 | 99.90 | 96.45 | 81.87 | 같은 L4의 fresh refit에서도 NS +1.61pp |
| L75 | 100.00 | 95.95 | 82.27 | local(.75,1) |
| T75 | 99.70 | 97.10 | 71.61 | terminal(.75,1) |
| L4D | 99.90 | 96.10 | 80.63 | L4 .75 채택은 1/10회뿐 |
| LD | 99.90 | 95.45 | 83.10 | (.75,.5) 9회, N4 1회 |
| TD | 99.90 | 96.70 | 80.26 | terminal 후보 채택 0회, 10회 모두 N4 |

Warm REFIT4의 NS 이득은 cold에서도 방향이 일치했지만, PS는 warm +0.60pp에서 cold −0.25pp로 바뀌었다. 이를 warm-state 하나의 인과효과로 단정할 수는 없으나, 과거의 PS 이득이 cold에서도 보장되지는 않는다.

Local/terminal을 동일 gate (.75,1)에서 비교하면 T75−L75는 PS +1.15pp, NS −10.66pp라는 큰 교환을 보인다. CAKE와의 차이를 해석할 때 중요한 실측이다. **Gate를 고정해도 target policy가 바뀌면 결과가 크게 달라진다.** 다만 T75는 CAKE 전체의 재현이 아니며 5층 구성, causal score, L2, target hparams가 다르다.

LD가 선택한 L8의 동일 상태에서의 역할도 확인됐다. L8을 쓴 9개 batch 모두 partial-L4→L8에서 S64 KL이 증가했다. B1은 다음과 같다.

| 동일 W0에서 만든 후보 | Native E | S64 KL |
|---|---:|---:|
| L4 .75만 적용 | .182800 | .000951336 |
| L4 .75→conditional L8 .5 | .0124145 | .001252665 |
| full L4 N4 | .00340461 | .001720908 |

이 순서에서 L8은 locality 손상을 직접 repair하지 않았다. L4를 약하게 쓰면서 부족해진 edit 효과를 보완하여 조합 전체의 KL을 full-L4보다 작게 유지했다. Layer allocation 연구의 주장도 이러한 조건부 효율로 정의하는 편이 실측에 맞는다.

Cold7 LD에는 E≤max(E_N4,.05)+1e−4의 plateau가 있었고 pair guard는 없었다. B1의 E 증가는 이 범위 안에서 허용됐다. Current canonical NLL은 788/1000개에서 ownN4보다 높았으며, PS의 최종 차이 25개 중 18개는 at-write 시점에 이미 발생했다. **낮은 PS의 원인을 장기 forgetting만으로 설명할 수 없다.**

당시에는 raw fixed (.75,.5) chain이 없어 dynamic 선택의 필요성을 식별하지 못했다. 이번 v2 F48이 그 대조군을 추가했다. Cold7 LD의 N4 대비 PS −1.25/NS +2.84pp와 v2 F48의 PS −1.25/NS +2.98pp는 가깝다. 단순 fixed를 강한 기준으로 놓을 이유는 되지만, 별도 실행의 차이로부터 “dynamic 효과는 −.14pp”라고 인과 추정하면 안 된다. V2 G48/C48의 동일 capsule 비교를 우선해야 한다.

## 5. CAKE의 문제를 static score 하나에 귀속할 수 없다

[CAKE native 10000 완료 보고서](/mnt/raid5/janghj/.codex/worktrees/odeeditgh-sh4-cake-baseline-cap-report-split-20260916-v1/experiment-reports/servers/server4/cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v1/diagnostic-report-ko.md)는 R/P/N을 CAKE 98.40/88.775/62.935%, 과거 L4-only 99.39/95.68/65.348%로 보고했다. 다만 hparams/backend/target policy/층 구성이 동시에 다르다.

CAKE는 entry에서 요청마다 L8 target을 한 번 만들고 L4..L8을 순서대로 업데이트한다. 매층에서 현재 상태의 K와 L8 readout을 읽고, 잔차를 `w_i/sum(w_i..w_last)` 비율로 배분한다. 고정 weight는 약 (.24395,.22777,.20870,.16821,.15137)이지만 실제 remaining ratio는 (.24395,.30127,.39506,.52634,1)이다. 마지막 L8은 남은 residual 전체를 적용하려 한다. F48 gate (.75,.5)와 같은 의미의 계수가 아니다.

CAKE는 L2=10, decay=.4, clamp=.5, temperature=.1을 유지했다. 현재 local-z v2는 L2=1, decay=.5, clamp=.75다. 따라서 현재 F48과 CAKE의 수치 차이는 고정 배분만 변경한 비교에서 나온 것이 아니다.

CAKE의 pooled-at-write P는 17767/20000=88.835%, W100은 17755/20000=88.775%였다. 낮은 PS는 작성 직후부터 거의 같은 규모로 존재한다. Generalization 진단을 우선할 근거지만 static score가 원인임을 보여주지는 않는다. CAKE first1000의 W10은 R/P/N=989/1703/8118, 즉 98.9/85.15/81.18%지만 현재 F48과 matched run으로 취급하지 않는다.

후속 비교에서는 CAKE 원방식을 충실하게 재현한 baseline과, 동일 target/writer/hparams에서 배분만 바꾸는 요인 실험을 분리해야 한다. 후자의 수정된 CAKE를 원 CAKE의 공식 결과라고 부르면 안 된다.

## 6. 즉시 규모를 확대할 근거가 약한 과거 방향

### Native endpoint 주변의 작은 preservation correction

[EP-TW-1 완료 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-completed-artifact-review-ko.md)에서 R/P/N은 EP 99.8/97.10/80.56%, 가장 가까운 과거 N4 99.8/97.15/80.72%였다. 새로운 cold paired 재실행은 아니지만 B1 target100/key/1300문항 NLL의 일치는 확인했다. Correction 후보 30개 모두 own native보다 S64를 줄였으나, 선택된 보정 8건의 norm은 native의 0.00675–0.02355%였고 즉시 R/P/N 성공 bit 변화는 0이었다. Alpha cap=1이 모든 batch에서 작동했다.

작은 국소 KL 감소의 작동 증거지만 F48 규모의 functional 효과를 얻은 증거는 아니다. Correction을 키우면 개선된다는 실측도 없다. 현재 크기를 그대로 10k로 확장하기보다 실제 write 크기와 품질 반응을 먼저 식별하는 것이 낫다.

### Actual-write feedback/ODE 형태로 변경

[SL-ZFlow 실행 리뷰](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-16-single-layer-zflow-seq1000-review-ko.md)의 R/P/N은 SL 100/94.20/72.30%, 과거 N4 99.8/97.15/80.72%였다. RS +0.20pp에 비해 PS −2.95pp, NS −8.42pp였다. P strict는 +5.00pp, N true strict는 +5.93pp로, target NLL 개선과 ranking/선택성 악화가 함께 나타났다.

특히 N true NLL은 5.302156→4.693866으로 개선됐지만 competing-new NLL은 9.272972→7.621593으로 더 크게 개선되어 margin이 악화했다. **True likelihood만 보호해도 new target의 부적절한 침투를 막을 수 있는 것은 아니다.** W0 KL, N true NLL, new-vs-true margin은 서로 다른 목적이다.

10/10 batch가 25 oracle에서 resource stop했고, accepted154/rejected86 중 초기 η=1의 overshoot만으로 reject 58회가 발생했다. Fixed λ, entry teacher, 목적함수, solver가 함께 바뀐 시험이므로 “ODE가 나쁘다”는 결론은 아니다. 다만 fresh feedback이나 긴 optimization만으로 품질–보존 균형이 개선된다는 기대는 입증되지 않았다.

### Full-L4 이후 다른 layer에서 repair

[B-OS Middle 부분 보고](/mnt/raid5/janghj/.codex/worktrees/odeedit-local-z-independent-audit-20260917/experiment-reports/servers/server2/multilayer-damage-compensation-b-2026-09-11-v1/partial-recall-r1/diagnostic-report-ko.md)에서는 full-L4를 고정한 뒤 L8 correction으로 Current E가 .0474684→.0628788, Base risk가 .0348951→.1136606, Past risk가 .0103827→.0194400으로 증가했다. Current N은 711→706/1000, Past N은 686→697로 방향이 섞였다. 일차 Current equality residual이 약 −2.5e−11이어도 finite step 보호에는 실패했다. PCG는 두 RHS 모두 20회에서 미수렴했고 edit core는 약 13996초였다.

하나의 근사 endpoint에 대한 부분 결과이므로 repair 전반의 불가능성을 뜻하지 않는다. 또한 F48의 L8 보완 효과를 full-L4 이후 repair의 성공 증거로 바꿔 해석할 수 없다.

2026-09-17 L4-preserving-repair에는 다른 감사 agent가 확인한 **technical READY 및 B1에서 실제 채택된 R-QP/R-GD finite endpoint**가 있다. Full-L4를 유지한 채 S64 KL을 약 23.87% 낮추고 canonical rewrite 성공 ID를 유지했다. 세부 수치와 조건은 [CAKE·repair 결과 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-f48-cake-research-direction/cake-and-repair-results-ko.md)를 참고한다. 이는 full-L4 이후 repair가 항상 불가능하다는 주장을 반박하는 단일 B1 기술 근거다. 다만 scientific commits=0이고 lifelong main 출력과 공식 P/N·Dev 결과는 없어, 완료된 장기 방법이나 generalization/locality 개선으로 승격하지 않았다. 설계서의 R-GD/R-QP arm 계획도 완료 성과에 포함하지 않았다. 2026-09-15 EP repair47942는 FP32 index 생성의 기술 실패로 commit0이어서 방법의 부정적 성능 결과로 세지 않았다.

## 7. 가장 저렴한 다음 식별 실험: 같은 entry에서 보완을 비교한다

아래는 **새 제안이며 미실행**이다. 처음부터 다층 COBYLA나 장기 10k 실험을 늘릴 필요는 없다.

첫 단계는 공통 W0의 한 B100에서 같은 L4 native fit을 공유하고 다음 endpoint를 비교한다.

| 후보 | 새로 식별할 질문 | 추가 native fit |
|---|---|---:|
| full-L4 | anchor | 필수 L4 1회 |
| L4 scalar 여러 점(.75/.875/.95/.99/1 등을 사전 고정) | scalar의 품질–locality 곡선 | 0 |
| L4 .75→conditional L8 .5와 같은 fit의 L8=1 | F48형 보완의 추가 가치·L8 강도 | L8 1회 공유 |
| L4 .75→같은 L4 fresh refit | 다른 layer의 자유도인가, target reset인가 | L4 1회 |

기본 비교는 native fit 총 3회=target request-call 300회이며, gate마다 native target을 다시 만들 필요가 없다. 실제 forward/scoring 비용은 별도로 들며 기존 F48보다 저렴하다는 약속은 아니다. Scalar family와 비교할 때는 raw P/N뿐 아니라 동일 native E/strict/pair 품질조건을 만족하는 범위와 S64/Dev를 함께 기록한다. L4 .75가 품질조건을 실패한 채 F48보다 높은 NS를 얻어도 allocation 개선으로 판정하지 않는다.

같은 entry에서 ①partial-L4→selected와 ②full-L4→selected를 모두 기록한다. L8이 edit를 복구할 때 generic/locality 비용을 얼마나 늘렸는지 측정한다. 실제 층별 ΔW norm/angle, 요청별 canonical 및 독립 P/N의 new/true NLL·margin을 남긴다. 그러면 단순히 약하게 쓴 효과와, 다른 방향이 같은 품질을 더 낮은 비용으로 달성한 효과를 구분할 수 있다.

Conditional target의 기여를 더 식별하려면 추가 1fit으로 entry-L8 local target을 먼저 만들고, 동일 partial-L4의 key/readout에서 frozen-entry target을 실현하는 대조를 둔다. 이는 CAKE 원방식 자체가 아니라 조건부 target 재계산을 겨냥한 ablation이다. 첫 3fit 비교의 효과가 거의 scalar로 설명된다면 이 단계나 5층 탐색을 서두를 이유는 약하다.

Cold B1에만 한정된 결론을 피하려면, 저장된 **완전한** cold7 N4 W5 checkpoint를 재사용할 수 있다고 확인된 경우 같은 비교를 해당 entry에서 반복한다. 현재 v2는 W/M checkpoint를 저장하지 않았으므로 v2 B5/B10을 추가 복원 비용 없이 동일 상태에서 재평가할 수 있다고 가정하지 않는다. 과거 entry를 사용한 진단은 기전 확인이며 새로운 unseen 검증은 아니다.

두 번째 단계에서 식별된 최소 후보와 F48·N4·G48을 공통 cold stream으로 비교한다. Calibration에서 선택한 fixed vector와 학습 또는 함수에 기반한 adaptive 선택을 별도 구간에서 동일 후보 family와 실측 비용으로 비교한다. Official P/N은 사후 observer로 유지한다. 선택에 추가 generalization 신호가 필요하면 별도의 training/calibration 규약을 먼저 고정한다.

## 8. 후속 방법의 연구 기여

과거 결과가 지지하는 설계 방향은 layer 중요도 score를 동적으로 만드는 것에 그치지 않는다. 더 구체적으로는 **약한 L4에서 어떤 요청의 편집이 부족하며, 어떤 conditional native 방향이 어느 정도의 preservation 비용으로 이를 보완하는지 실측하고, 필요한 보완만 선택하는** 문제 설정이다.

고정 F48이 강한 기준인 만큼 개선 후보는 (1) 같은 PS/strict/NLL에서 NS/Dev를 더 개선하거나, (2) 같은 품질–보존 수준을 더 적은 fit/score로 얻거나, (3) 다른 entry/order/horizon에서 fixed보다 안정적임을 보여야 한다. 큰 탐색 공간, 비고정 계수, ODE라는 이름, layer 수 증가 자체가 연구 기여는 아니다.

현재 기록을 바탕으로 피해야 할 해석은 proxy 감소만으로 locality 전반의 성공을 선언하거나, P preference만으로 generalization 무손실을 주장하거나, warm의 이득을 cold의 이득으로 옮기거나, full-L4 이후 repair와 약한 L4의 edit 보완을 혼동하는 것이다.
