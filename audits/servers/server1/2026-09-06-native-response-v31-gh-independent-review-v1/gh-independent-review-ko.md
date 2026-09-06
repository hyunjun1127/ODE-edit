# Native-response ODE v3.1 — GH 독립 코드·산출물 검토

검토일: 2026-09-06 KST. 범위: 종료된 SH1 B10 short-history mechanism pilot. SH4 진행 중 retention rerun은 접근·변경하지 않았다.

## 결론

**핵심 controller 구현과 기록된 실험 결과는 검토를 통과했다. 다만 요청한 모든 측정 항목의 구현 완료로 승인하지는 않는다.** 정규화 model error의 단위 오류, 일부 관측값의 미기록 및 상태 표기 보강이 남아 있다. 이것을 낮은 성능에 따른 scientific failure나 전체 결과 무효로 처리할 근거는 없다.

현재 인정할 결론은 **“joint response controller가 단순 ray rescaling과 다른 physical field를 선택하고 다른 경로를 만든다”**이다. **“유사한 신규 edit 품질에서 더 적은 native action으로 locality/retention이 개선된다”**는 결론은 입증되지 않았다. `scientific_promotion=false`를 유지한다.

새 실험·재실행·gate 강화는 하지 않았다. 기존 source/raw/report/manifest/receipt/main도 수정하지 않았다. 이 검토서와 독립 검증 출력만 별도 local 경로에 생성했다.

## 1. 인수한 source와 실제 검증

- 공개 main: `347a892a910317b606443cf53d9155ae6a460831`, tree `0672fd84a2e46efac1b528737534d8cf63476cdc`. 분석 worktree의 tracked clean 확인.
- Primary 실행: `29884f208bca5afc2367c67510779b4a674cbbb1`.
- Refinement 실행: `964453d9308c1f68848c08946b01a72a5820bc55`.
- Audit technical repair 실행: `2214387fba0f6338c09269004e6b305f576b733f`.
- Final report SHA: `8bd7956fe12fc82fab9cae4b0f92438c7a8dc3c8c502f511dcc219d99a5f2aff`.

| 독립 점검 | 결과 |
|---|---|
| CPU focused tests | 26/26 직접 재실행 PASS, CUDA 비노출 |
| 최종 package | manifest-listed 76개 전부 rehash/Git blob 일치; manifest·receipt 포함 총 78개 |
| Component package | primary 49개, diagnostics 19개 member·root·receipt identity 일치 |
| 외부/raw 입력 | 고유 495개, 31,568,869,157 bytes 전체 SHA 재검산; 불일치 0 |
| Primary D10B | 16 arms, RS 160 / PS 320 / NS 1600 prompt 분모; raw NLL pair·strict·평균 NLL·old-loss 직접 재집계 일치 |
| H10 audit | 16 arms, RS 160 / PS 320 / NS 1600 prompt 분모; 같은 재집계 일치 |
| D2 refinement | 실제 CPU tensor에서 8개 N2→4 / N4→8 endpoint 거리 직접 재계산, CSV 불일치 0 |
| GPU fidelity 기록 | G0 8 fixtures / 40 layer FD receipts 및 overlay/materialization·restore 기록 확인 |
| 예산 | scheduler 원기록 재조회, 누적 `[4064,3491,6325,4332]`초 일치; 총 5.058888889 GPUh |

160은 fixture별 request-arm endpoint 수이지 서로 다른 160개 신규 fact가 아니다. 네 cell은 같은 10개 D10B와 같은 10개 H10을 각각 사용한다. H10은 D10B 이후 state가 아니라 동일 D10A warm snapshot에서 독립 시작한다.

독립 수치 검증: [verification.json](/mnt/raid5/janghj/ODE-edit/local/native-response-v31-gh-review/20260906-r1/verification.json). 실제 refinement 거리 검증: [refinement-verification.json](/mnt/raid5/janghj/ODE-edit/local/native-response-v31-gh-review/20260906-r1/refinement-verification.json). 재검산기는 원자료를 읽고 JSON만 출력하며 원자료를 쓰지 않는다: [verify.py](/mnt/raid5/janghj/ODE-edit/local/native-response-v31-gh-review/20260906-r1/verify.py).

## 2. 코드·수학 계약 판정

다음을 소스에서 확인했다.

1. Native full-residual directions는 layer별 동일 node state version에서 모두 만든 뒤 joint update한다. Direction/JVP 생성 중 다음 layer state를 섞지 않는다. Writer RHS에는 objective normalization을 적용하지 않는다.
2. MEMIT의 native covariance·regularization 및 ephemeral FP64 solve를 유지한다. AlphaEdit의 solve에는 stock P-inside 구조를 사용하고 metric에는 별도의 symmetric `M_entry + L2 I`를 사용한다.
3. Entry `qN_ref`·`qF_ref`·normalization을 고정한다. Disjoint native-whitened coordinates의 `G=I`와 실제 Frobenius `G_F`를 구분한다. JV/ray가 같은 entry/fixed-z/reference를 소비함을 네 cell raw에서 대조했다.
4. NNLS active-set enumeration은 negative individual gain column을 미리 버리지 않으며 coefficient cap 1, ridge, post-hoc gain, h-dependent objective를 넣지 않는다. Ray는 raw-direction ORBFH rule을 native coordinates로 변환한 뒤 full-condition ray normalization한다.
5. JV와 ray는 각자 도달한 state에서 dictionary/JVP를 재계산한다. ORBFH comparator는 기존 ordered T1/N4/h.25, JV/ray는 T2/N4/h.5 계약을 유지한다.
6. Current-state old D10A prompt replay가 controller 입력에 없다. Old/new evaluation은 endpoint callback에 격리된다. 공통 warm entry와 fixed-z를 공유하고 comparator 간 W/cache restore한다.
7. Finalization 시 한 번의 materialized endpoint와 baseline-compatible AlphaEdit history 갱신을 사용한다. Cross-process repair는 content/shape/dtype 비교로 한정하고 process 내부 무결성 검사를 유지한다.

Raw shadows의 coefficient로 native turning을 별도 계산했을 때 32/32 양수, 최소 `R_turn=0.0885904172402297`이다. 동일 state에서 JV gain ≥ ray gain도 일치했다. Primary JV 16 nodes의 dissipation identity 오차 최대 `3.608224830031759e-16`, speed bound 위반 0이다. 이는 기록된 native coordinate 기하에서의 검증이며 locality 정리가 아니다.

## 3. 보완이 필요한 구현·기록

### R1 — 정규화 model error 대신 raw activation error가 저장됨 (P2)

[trajectory.py:97](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s06-native-response-v31-analysis-v1/project/run_scripts/native_response_ode_v31/trajectory.py:97)은 `||Phi_exit-Phi_entry-h*predicted_raw||`를 `model_error`로 저장한다. 사용자 계약은 `||Delta Phi_tilde_actual-h*Psi*c||`, 즉 N0 request weighting이 적용된 오차다. 같은 record의 `velocity_mismatch`는 정규화 공간이므로 두 수치를 같은 공간의 tracking/model error decomposition으로 비교하면 안 된다.

기존 `model_error`를 raw activation norm이라고 명시하고, 계약상의 normalized model-error norm은 `NOT_RECORDED`로 구분해야 한다. Scalar raw norm 하나를 평균 scale로 나누어 복원해서는 안 된다. 향후 실행 계측에는 request별 weighting 후 norm을 추가해야 하지만, 이 검토는 실행 코드나 원자료를 수정하지 않았다. 이미 저장된 normalized V/barrier 차이로 계산하는 finite barrier defect는 이 오류와 별개로 유효하다.

### R2 — Matched-progress 표의 ‘saved’는 weight state 보존을 뜻하지 않음 (P2, 표기/완전성)

[analysis.py:121](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s06-native-response-v31-analysis-v1/project/run_scripts/native_response_ode_v31/analysis.py:121)은 scalar V가 level ±.02 안에 들어오면 `MATCHED_SAVED_NODE`로 표시한다. Primary/H10 중간 weight나 low-rank journal은 파일로 보존되지 않았고, [trajectory.py:110](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s06-native-response-v31-analysis-v1/project/run_scripts/native_response_ode_v31/trajectory.py:110)의 tensor endpoint 저장은 D2 전용이다.

Primary에서 level 근처의 scalar 관측은 LA-JV node0, QM-ray node1, QA-JV node0의 3건이다. 모두 endpoint 평가가 없고, 같은 cell의 JV/ray 양쪽이 matched된 pair도 없다. 따라서 이 3건은 `OBSERVED_LEVEL_HIT_UNEVALUATED`에 해당한다. 어떤 grid node도 tolerance 안에 없다는 뜻과 trajectory가 level을 지나지 않았다는 뜻도 구분해야 한다. 기존 보고서가 matched-quality 우월성을 주장하지 않은 점은 적절하다.

### R3 — NNUM noise floor가 모든 pair가 아니라 한 reference에 대한 최대 오차임 (P3)

[trajectory.py:59](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s06-native-response-v31-analysis-v1/project/run_scripts/native_response_ode_v31/trajectory.py:59)은 `max_a ||Phi_a-Phi_reference||`를 사용한다. 계약은 `max_a,b ||Phi_a-Phi_b||`다. 예를 들어 관측 `[0,+1,-1]`이면 각각 1과 2로 다르다. 다음 구현에서는 이미 얻은 primal captures의 pairwise 차이로 계산하면 추가 forward 없이 교정 가능하다.

다만 **이번 pilot의 primary N0와 관측된 NNUM 결과에는 영향이 없다.** D10B/H10 JV/ray의 64 nodes에서 `s / recorded_nu`의 최솟값이 84,950.62다. 삼각부등식상 정확한 pairwise floor는 recorded floor의 2배 이하이므로, 올바른 식이어도 어느 request도 floor에 걸리지 않는다. 이를 이유로 이번 결과를 재실행할 필요는 없다.

### R4 — 필수 physical/compute 관측 일부는 여전히 미기록 (완전성 제한)

Frobenius cross-field cosine, cross-arm dense endpoint distance, intermediate matched-quality evaluation, Official/ORBFH의 continuous native path/work, setup backward 및 key/solve/observation별 세부 비용은 완성되어 있지 않다. 일부는 SH1 `missing_fields.csv`와 REVIEW_READY에 이미 정직하게 명시했다. 이 상태는 ‘네 cell pilot과 예산 집행 완료’와 양립하지만 ‘모든 측정 계약 완료’와는 다르다. Historical 자료가 없는 항목은 `HISTORICAL_STATE_UNAVAILABLE` 유지가 맞다.

추가 주의: AlphaEdit의 stored-projector operator-action residual은 실제 기록되어 있고 최대 `1.3160749961567988e-4`다. [fidelity.py:67](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s06-native-response-v31-analysis-v1/project/run_scripts/native_response_ode_v31/fidelity.py:67)의 값은 관측이지 별도 feasibility 합격 판정은 아니다. 새로운 임의 tolerance를 도입해 이 결과를 실패로 바꾸지 말되, projector가 수치적으로 exact하다는 주장도 하지 않아야 한다.

## 4. 과학적 해석

| 질문 | GH 판정 |
|---|---|
| 수학·구현 fidelity | 핵심 식·current-state 결합·stored output 검증 PASS, 위 계측 보완은 별도 |
| Historical near-stall 원인 | unavailable/unresolved; 신규 fixture로 B9 원인을 확정하지 않음 |
| Same-state physical turning | 있음, 32/32; 단순 scalar-only 설명과 불일치 |
| 실제 trajectory 변화 | V·endpoint action 차이 있음; cross-arm dense distance/angle 미기록 |
| Fixed-T refinement | 네 cell에서 N2→4보다 N4→8의 weight·activation 거리 감소; 단일 B1 예시의 관측, 일반 수렴 정리 아님 |
| Matched progress/efficacy usefulness | 유효한 matched-quality pair 없음; 미입증 |
| Short-history retention | Primary와 audit 모든 arm old new-failure 0/10; 차별적 개선 미입증 |
| Native vs Frobenius | 기하 구현·same-state shadow는 구분됨; actual trajectory metric ablation의 우월성은 미입증 |
| 추가 계산량 | 총 5.059 GPUh, 예산 준수; 세부 breakdown은 일부 미기록 |
| Retention/lifelong 확장 주장 | 승인하지 않음; SH4 rerun과 별도 실험 |

Primary JV는 ray 대비 native endpoint action이 1.84–2.52배로 크며 rephrase target-new 평균 NLL은 네 cell 모두 더 높다. NS 차이는 LM/LA/QM에서 0, QA에서 +3/100이고, PS는 LA +2/20, QM -1/20이다. 방향이 다르다는 사실만으로 유용한 경로라는 결론을 내릴 수 없다.

H10 audit의 JV−ray 결과도 혼합되어 있다. PS 차이는 LM 0, LA -1/20, QM -1/20, QA 0이며 NS 차이는 +1/100, 0, +4/100, 0이다. Rephrase 평균 NLL은 QM에서만 낮고 나머지 세 cell에서는 높다. 모든 arm에서 old-edit loss가 0이므로 audit도 retention 우월성을 입증하지 않는다.

## 5. 권고와 작업 경계

기록된 pilot은 보존·인수하고, 다음 보고서 교정에서는 R1/R2의 명칭과 미기록 경계를 먼저 반영하는 것이 맞다. R3는 다음 실행 버전에서 수정할 작은 관측 코드 문제이며 이번 N0/NNUM 결과를 바꾸지 않는다. 부족한 관측을 채우기 위한 GPU 재실행이나 확대 실험은 이 검토로 승인하지 않는다.

검토 중 actions: model instantiate 0, GPU 0, Slurm mutation/submit 0, experiment source/raw mutation 0, report regeneration 0, Git commit/push 0. Scheduler read-only 확인과 CPU tensor reduction만 수행했다. 기존 package의 GH review 필드는 immutable하게 그대로 두고, 본 별도 검토서를 검토 근거로 사용한다.

원 보고서: [SH1 최종 보고서](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s06-native-response-v31-analysis-v1/experiment-reports/servers/server1/native-response-v31-b10-warm-pilot-2026-09-06-v1/factual-report-ko.md).
