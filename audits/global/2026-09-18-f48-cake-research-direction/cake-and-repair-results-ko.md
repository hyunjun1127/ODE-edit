# CAKE 완료 실험과 preservation repair의 현재 저장 결과

2026-09-18. 지정된 CAKE publication과 server4 원자료를 read-only로 확인했다. 신규 GPU·모델 실행·scheduler 조회·job 제출은 없다. 이전 sequential-local-z 원자료 전량 검산을 반복하지 않았다.

## 핵심 판단

**기존 CAKE는 이미 10,000-request chain이 완료됐지만, F48의 locality 이득을 설명하는 통제된 비교는 아니다.** 같은 first1000에서도 CAKE의 actual W10은 RS/PS/NS=98.90/85.15/81.18이고 F48은 100.00/95.45/83.24다. 그러나 두 실험의 target 위치·정책뿐 아니라 실제 context 문자열, L2, target decay/clamp, seed와 cuDNN TF32 설정까지 다르다. 이 차이를 "CAKE의 layer allocation이 약하다" 또는 "F48이 tracing allocation보다 낫다"로 단독 귀속하면 안 된다.

**9월 17일 full-L4 이후 L8 preservation repair의 기술 pilot에는 실제 성공한 finite endpoint가 있다.** 이전 handoff의 PENDING 표기만 보면 이 결과를 놓친다. 현재 READY와 selection이 존재하며 B1에서 S64 KL을 약 23.87% 줄이고 canonical rewrite 성공 ID를 유지했다. 다만 scientific commits=0, 두 lifelong arm 출력 경로 없음, 공식 P/N·Dev 성능 없음이다. 성공한 기술 pilot을 완료된 lifelong 방법이나 최적 layer 증거로 승격할 수 없다.

## 1. CAKE 실험의 실제 완료와 성능

정본 [CAKE 완료 보고서](/mnt/raid5/janghj/.codex/worktrees/odeeditgh-sh4-cake-baseline-cap-report-split-20260916-v1/experiment-reports/servers/server4/cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v1/diagnostic-report-ko.md)의 SHA는 `7b14e523cbbe3bf814eaad9ca75b8703847385ec75af18b124125a40784ba7b3`이며 rooted receipt와 일치한다.

원자료 루트는 `/data/janghj/ODE-edit/local/cake-native-lifelong/20260915-v1/attempt-v1`이다. `output/main/terminal.json`은 `TERMINAL_VALID`, 100 batch, 10,000 request, 99 state links, 10,000 compute_z, 500 solve, 100 history pass/500 layer append, nonfinite=0을 기록한다. 이번에는 terminal metadata와 actual B010/B100의 `seen-full.json` 원문항 NLL을 읽고 성공 조건을 재계산했다. 아래 여섯 metric의 모든 row success와 저장 분자·분모가 일치했다.

| 모델 상태/집단 | RS | PS | NS |
| --- | ---: | ---: | ---: |
| CAKE W10 / first1000 | 989/1000 = 98.90% | 1703/2000 = 85.15% | 8118/10000 = 81.18% |
| CAKE W100 / full10000 | 9840/10000 = 98.40% | 17755/20000 = 88.775% | 62935/100000 = 62.935% |
| F48 v2 W10 / first1000 | 1000/1000 = 100.00% | 1909/2000 = 95.45% | 8324/10000 = 83.24% |

CAKE W100에서 같은 first1000의 성능은 별도로 RS937/1000, PS1675/2000, NS6087/10000이다. W10 first1000과 혼합하면 안 된다. 10k 결과가 1k보다 낮은 NS는 누적 horizon을 포함한 관측이며, 서로 다른 cohort 구성의 영향도 함께 있다.

기존 동일10k family 표에서는 AlphaEdit L4-only=99.39/95.68/65.348, BLUE L4+L8=98.88/95.775/63.726, five-layer BASE_ALPHAEDIT_NATIVE=73.43/62.885/55.285다. CAKE는 five-layer native baseline보다 높은 관측치를 보이지만 L4-only/BLUE보다 PS가 낮다. 보고서 자체도 hparam/환경 차이를 공개하므로, 이 표는 정책 전체의 개발 결과이지 causal weighting 하나의 ablation이 아니다.

CAKE 전체 할당은 32,194 GPU초(8.943 GPUh), program 32,189.62초다. 평가 시간이 17,949.8초, edit 시간이 12,478.57초로 평가 비중이 크다. F48 first1000의 5,560 GPU초와 10k CAKE 비용을 직접 나누어 효율 우열을 내릴 수 없다. 평가 빈도·horizon·target 정책을 맞춰야 한다.

## 2. 실행된 CAKE의 layer 분배는 무엇인가

원 source commit `c8243e1d7e43ca9cf64d552f96221fcb9561aac2`, wrapper commit `7884aeb6000f8343139172825ec6c4ca24357fc0`. 실행 native 파일은 unused notebooks import 한 줄 제거 이외 계산 source를 유지한 복사본이다. 이번에 execution lock, 실제 `Cake/Cake_main.py`, `hparams/Cake/Llama3-8B.json`을 확인했다.

실제 정책은 batch entry에서 **L8 target을 요청별 한 번** 만든 뒤, L4→L5→L6→L7→L8 순서로 현재 state의 key와 L8 readout을 갱신하고, 남은 residual에 remaining-weight ratio를 곱하여 native projected direct solve를 수행한다. 다음 layer마다 fresh local-z를 푸는 v2/F48과 다르다. `compute_optimal_deltas`라는 별도 함수가 source에 있어도 이 실행 경로는 `apply_Cake_to_model`이다.

| physical layer | 저장 causal score | normalized weight | 실제 remaining ratio |
| --- | ---: | ---: | ---: |
| L4 | 0.4812439084 | 0.2439513 | 0.2439513 |
| L5 | 0.4743820429 | 0.2277731 | 0.3012678 |
| L6 | 0.4656370878 | 0.2087006 | 0.3950600 |
| L7 | 0.4440660179 | 0.1682061 | 0.5263430 |
| L8 | 0.4335190654 | 0.1513689 | 1.0000000 |

Score→softmax temperature=.1이고, remaining ratio는 `w_i / sum(w_i:)`다. Source `Cake_main.py:137–146`에 직접 있다. **L8 weight가 .151이라고 L8가 남은 residual의 15.1%만 담당하는 것은 아니다. 마지막 L8는 남은 residual 전체를 담당한다.** 이 값은 final native endpoint에 .151 gate를 곱하는 것과도 다르다.

실제 [layer-actions.csv](/mnt/raid5/janghj/.codex/worktrees/odeeditgh-sh4-cake-baseline-cap-report-split-20260916-v1/experiment-reports/servers/server4/cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v1/layer-actions.csv)를 다시 합산하면:

| 누적 step norm 합 | L4 | L5 | L6 | L7 | L8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1–B10 | 11.9911 | 14.2032 | 17.2441 | 20.3673 | 31.7722 |
| B1–B100 | 132.8843 | 155.3918 | 185.0000 | 213.0715 | 334.4480 |

L8가 이 path-length 합의 각각 33.24%, 32.76%로 가장 크다. Score 순위가 높은 L4에 가장 큰 실제 weight 변경이 배분됐다는 해석은 성립하지 않는다. 이 비율은 서로 다른 층 norm의 설명용 비중이며 Fisher energy/누적 net delta/남은 capacity 비율이 아니다. 원 보고서에도 endpoint net delta는 NOT_RECORDED다.

## 3. Tracing 근거와 F48 비교의 교란

실행 config에는 physical4–8에 대응하는 index0–4의 causal score 다섯 개가 고정되어 있다. Source에 `experiments/causal_trace_llama3.py`와 tracing 구현은 있지만, 조사한 `execution/CAKE` tree에는 `.npz/.npy/.pt` tracing 결과 payload가 없다. 이번 실행에 연결된 request-level tracing sample ID/전체 layer score/재생산 receipt가 있음을 확인하지 못했다. 따라서 **공개 config의 고정 prior를 사용했다**고 말할 수 있고, 현재 edit stream에서 tracing을 다시 해서 adaptive allocation을 했다고 말할 수는 없다. 다른 위치에 tracing 결과가 전혀 없다고 단정한 전역 검색 결과는 아니다.

| 항목 | 실제 CAKE | F48/v2 | 영향 |
| --- | --- | --- | --- |
| horizon | B100×100, 중간 W10 저장 | B100×10 | W100 대 W10 비교 불가 |
| native target | entry에서 L8 target 1회/request | L4 target 후 적용된 prefix에서 L8 local target | target 좌표와 목적 경로가 다름 |
| layer policy | L4–8의 remaining residual 분배 | L4 .75 → L8 .5 native endpoint | coefficient 의미가 다름 |
| solve ridge | L2=10 | L2=1 | solve regularization 교란 |
| local-z decay/clamp | .4 / .5 | .5 / .75 | direction·norm·조기종료 교란 |
| seed | 20260907 | 20260916 | 공통 RNG capsule 아님 |
| actual context | fixed10k baseline의 context | cold7/v2 공통 context | 실제 문자열도 다름 확인 |
| cuDNN TF32 | true | false | 수치 규약 차이; 관측 효과 크기는 미측정 |
| selection | fixed original policy | F48 fixed; C*에는 S64/guard feedback | F48과 C*를 함께 해석할 때 추가 교란 |

Context 차이는 단순 manifest 이름 차이가 아니다. CAKE는 첫 추가 context가 `The 5th annual "Taste of. {}`이고 v2는 `The 2022 NFL Draft is in the. {}`다. 나머지 추가 context도 다르다. Model revision, FP32/eager, fixed10k whole-order root `5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729`는 공통이다.

CAKE source 파일 SHA `42599bb68675c0cf167ce676a0827a33827c2254cd7d6bd622553be5c914b96c`, config SHA `457b63f35bf12d6b1020b87b7d241684963f32a48db94eca1ba89c0a6729bd3e`, execution lock SHA `f2ade3ee9dbdabdc6a1ac00a9d36b0e902710a44cf5e2a6a49c6d861e18eb401`, terminal SHA `a0db1ff4c3d8b9e0581ec340f5c98cb800d6da2103f8bb4ab3a03eb28c2a2ac2`를 이번에 재계산했다. 대형 checkpoint/tensor 재검산을 반복하지 않았다.

## 4. 9월 17일 full-L4 preservation repair의 새 저장 증거

원자료 `/data/janghj/ODE-edit/local/l4-preserving-repair/20260917-v1`의 `resume-manifest.json`은 과거 handoff 시점의 PENDING/READY 미관측 기록을 보존한다. 그러나 현재 `technical/attempt-v1/READY.json`은 실제 존재한다. Scheduler를 새로 조회하지 않고 다음 저장 결과를 확인했다.

- status=`TECHNICAL_READY`, scientific_commits=0, native_target_calls=100, native_solve=1, L8_target=0.
- 소요 1,012.93초, source `a89350e2c2b2b0652f0a085b2b6c2e9c3accc89b`.
- READY가 참조한 32개 작은 JSON의 SHA를 이번에 다시 계산했고 불일치 0.
- `twoarms` 경로 자체가 없다. 따라서 R-GD/R-QP의 fresh lifelong main이 완료되었다고 보고할 자료는 없음.

Full native L4 B1 anchor는 S64 B=0.0017209079930182725, canonical Current E=0.0014505088787700516이다. 이 anchor의 L4 SHA `3d33f44b13688fb3c18db8eb4d8be5cc6509aea161c96bed254e89240f265015`는 v2 기술 native N4 B1과 같다. Repair는 W4를 유지하고 W8만 변경한다.

| pilot | anchor S64 B | selected S64 B | 감소율 | selected canonical E | Current strict/pair | 실제 ΔW8 norm |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| R-QP, 2방향 | .001720907993 | .001310120657 | 23.87% | .001450514951 | 100/100, 100/100 | .0198113334 |
| R-GD, 1방향 | .001720907993 | .001310109815 | 23.87% | .001452938002 | 100/100, 100/100 | .0198115333 |

둘 다 첫 probe에서 accepted=true, reasons=[], first acceptable selected다. QP 실제 감소=.000410787336, 예측 감소=.000379613486, agreement=1.08212다. GD는 같은 기술 anchor/gradient/curvature를 재사용한 비교이므로 독립 scientific run 두 번으로 세지 않는다.

수치 derivative gate는 B 및 canonical Current의 두 scale FD/AD에서 PASS다. 의미 있는 강점은 **native 추가 target fitting이 아니라 preservation objective로 만든 작은 L8 weight 방향에서 유한 KL 감소를 실제 확인했다**는 것이다. F48의 두 번째 local-z와 다른 종류의 방향이다.

범위 제한도 크다. B1이라 실제 Past는 없고, canonical Current 보호 조건은 v2의 native-context E guard와 동일하지 않다. Official PS/NS와 Dev128은 측정하지 않았다. L8만의 단일 first100 anchor이고 L5/L6/L7/L9/L12 비교는 없다. 따라서 "full L4 후에는 보존 복구가 불가능하다"는 강한 주장은 이 pilot과 맞지 않지만, "L8 repair가 일반적으로 성공한다", "L8가 최적이다", "PS/NS가 좋아진다"도 아직 증명되지 않았다.

핵심 JSON SHA:

- READY: `09a517d3fe3840ded031f779ef6c9901a59267cfa4bbac499ea9ef265ddfbf76`
- QP selection: `69eec81a510bb2f1b986d24c6883d6a5b390d008e170db12ea00c3a38545195b`
- QP decision: `a995483d786b638e8113e8fcb5e69403ab23d687f973f95ea03cacfa40d34e43`
- GD selection: `55bcaca8a6739c63c049bd2cd9afd950bb64b76cee88d30d24cd66ee0717971b`
- GD decision: `f5c51e50ab1beba86ecb86ed81a177a274d6a5f69972719598881ec3a78232d1`

## 5. 다른 repair 관측과 후속 해석

기존 [B-OS 부분 보고](/mnt/raid5/janghj/.codex/worktrees/odeedit-local-z-independent-audit-20260917/experiment-reports/servers/server2/multilayer-damage-compensation-b-2026-09-11-v1/partial-recall-r1/diagnostic-report-ko.md)는 Middle anchor에서 Base128 risk .034895→.113661, Current NS71.1→70.6으로 악화됐다. 두 PCG RHS가 20회 미수렴했고 finite acceptance가 없는 다른 correction 절차였으므로, 9/17의 작은 guarded finite repair와 동일 알고리즘 반복 실패로 합치면 안 된다.

[9/17 layer feasibility 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-17-repair-layer-feasibility-audit-v1.md)의 3anchor×6layer=18cell은 설계 문서다. 지정된 server4 local root에서 대응하는 완료 실행 root나 결과표를 찾지 못했다. CAKE의 band 안 tracing이나 singleton edit 성능으로 이 layer별 repair 순위를 대신할 수 없다.

이 자료가 지지하는 후속 방향은 두 가지를 구별하는 것이다. CAKE에서 가져올 수 있는 것은 **정해진 layer prior와 현재 residual을 연결하는 순차 분배 구조**다. F48에서 얻은 locality 관측은 **초기 writer 강도를 약화하고 후속 native local target으로 edit 품질을 보완하는 정책**의 결과다. Repair pilot은 **edit 방향을 새로 더하기보다 실제 preservation gradient 방향을 사용할 여지**를 보여준다. 세 개의 update family를 혼합하기 전에 동일 context·target solver·ridge·entry·guard에서 별도로 비교해야 각 원인의 효과가 식별된다.

## 6. 최신 9월 18일 single-layer EN-F 설계와의 관계

현재 [single-layer edit-preserving correction 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-18-single-layer-edit-preserving-correction-design-v1.md)는 **상세 설계와 CPU 수학 검증 단계**다. 첫 문장과 마지막 절 모두 실제 모델 runner 미구현, GPU 제출 없음, numerical preflight/teacher/source capsule 준비가 남았음을 명시한다. [validation receipt](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-single-layer-edit-preserving-correction/validation_receipt.json)는 설계·분모·예산·링크 일관성 1,430/1,430 PASS, `model_runner_validated=false`, `gpu_model_tests=0`이다. [geometry checks](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-single-layer-edit-preserving-correction/geometry_checks.json)는 NumPy FP64 toy 검사 11개 PASS이고 실제 model gradient/FP32 materialization은 NOT_RUN이다. 이번에 artifact manifest의 8개 작은 파일 SHA/크기를 확인했고 모두 일치했다. 이 문서의 T/M/S/R/L cells와 호출 상한은 완료된 실험·실측 비용이 아니다.

이번 보존 방향 논의와 **질문은 상당히 겹치지만 방법은 다르다.** EN-F는 full native L4 이후 **같은 L4**의 full-token edit key를 잠그는 `D K_E=0` 공간에서 W0 KL gradient로 보정한다. 추가 layer·추가 z·CAKE tracing prior를 쓰지 않는다. 앞서 확인한 9/17 pilot은 W4를 고정하고 **L8**에 guarded correction을 넣었다. CAKE는 L8 target residual을 **L4–8**에 나눈다. 따라서 9/17 L8 pilot의 성공이 9/18 L4 EN-F 공간에 실제 자유도나 functional gradient가 남는다는 H1의 검증을 대신하지 못한다.

이미 최신 설계에 SCALE/CA/KL-P/EN-S/EN-F/EN-COV 대조, 실제 FP32 response invariant, 동일 native endpoint 비교, 이후 독립 lifelong chain, rank/gradient/비용 진단이 들어 있다. "그저 edit를 약하게 해서 NS가 좋아진 것인가"와 "writer 밖 보존 방향이 필요한가"를 검증하는 틀은 재사용 가치가 있다. 반면 §1은 H1–H3 이전에 layer allocation sweep을 동시에 진행하지 않는다고 명시하므로, CAKE 배분 연구를 이 EN-F의 구현 세부 변경처럼 섞어 넣으면 실험 질문이 달라진다. 두 방향을 구별하고 공통 관측·수치검증 틀을 공유하는 것이 현재 자료에 맞는 정리다.
