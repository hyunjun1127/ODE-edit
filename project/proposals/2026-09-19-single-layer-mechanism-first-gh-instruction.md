당신은 ODE-edit의 Global Head(GH)다.

Instruction ID:
GH-SL-MECHANISM-FIRST-W0-B100-S3-S10-20260919-V1

목표는 single-layer의 강한 편집 성능을 유지하면서 실제로 줄일 수 있는 locality 손상이 존재하는지 검증하고, 이를 정량적인 보정 방향 선택으로 줄이는 것이다.

아래 설계에 따라 구현·기술 검증·cold B100을 수행하라. 사전 gate를 통과하면 정책 변경 없이 SEQ300, 이후 SEQ1000까지 진행하라. 준비 문서 작성이나 제출 계획에서 종료하지 말고, 실제 완료 결과 또는 구체적인 미해결 의존성을 보고하라.

1. 기준 문서와 실행 상태를 먼저 결속하라

다음 세 파일을 함께 읽고 bytes/SHA256을 실행 manifest에 봉인하라.

- /mnt/raid5/janghj/ODE-edit/plans/global/2026-09-19-single-layer-mechanism-first-experiment-design-v1.md
- /mnt/raid5/janghj/ODE-edit/plans/global/2026-09-19-single-layer-mechanism-first-contract-v1.json
- /mnt/raid5/janghj/ODE-edit/plans/global/2026-09-19-single-layer-mechanism-first-cells-v1.csv

참조:
- /mnt/raid5/janghj/layer_allocation/single_layer_mechanism_and_method_review_20260919.md
- /mnt/raid5/janghj/layer_allocation/en_r512_g256_capacity_scope_audit_20260919.md
- /mnt/raid5/janghj/ODE-edit/PROTOCOL.md

원격 checkout에서는 실제 repo root로 경로를 매핑하되 동일 파일임을 확인하라. 이 문서가 이미 main에 반영됐다고 가정하지 말라.

현재 설계의 DESIGN_ONLY 상태를 구현 완료로 해석하지 말라. 실제 runner·runtime·재사용 자산의 존재와 검증 상태를 확인하라.

설계·JSON·cells 사이 충돌이 있으면 실행 전에 내용을 명시하고 정정본을 봉인하라. 과거 EN/BPCW의 reference 길이, Past sampling, GSS, 후보 수, 확장 gate를 자동 승계하지 말라.

2. GH/SH 소유권과 실행 범위를 명시하라

현재 PROTOCOL과 서버 상태를 확인하고 실행 담당 SH를 지정하라. 기존 Server4 EN 자산의 재사용 가능성을 먼저 확인하되, 과거 server/session/job identity를 현재 실행에 그대로 사용하지 말라.

GH는 SH envelope에 다음을 명시하라.

- Instruction ID, 실제 checkout/branch, source commit.
- 구현 허용 범위: 전용 project/run_scripts/single_layer_mechanism_first/ 아래 runner·oracle·solver·분석·검증 코드.
- 기존 native/EN/runtime dependency는 검토한 source로 결속.
- 공통 runtime 변경이 필요하면 해당 작업 소유자와 조정하고 정확한 추가 수정 범위 명시.
- 서버별 report/audit 경로, ignored local raw 경로.
- 실제 GPU/host-memory/disk/wall cap과 제출 범위.
- T0/B1 및 조건부 S3/S10 실행 권한, 중단 조건, 완료 보고 경로.

격리된 codex/ branch/worktree를 사용하고 다른 작업자의 변경을 되돌리지 말라. 현재 진행 중인 실행 재사용 최적화와 중복 구현하거나 running job을 중단하지 말라.

SH는 사실·수치·계약상 gate 결과를 보고한다. 과학적 해석은 GH가 별도 작성한다. Source 공유·push·scientific submission은 프로젝트의 GH→SH 절차로 처리하고, 이미 범위에 포함된 정상적인 다음 단계마다 사용자 재승인을 요구하지 말라.

3. 모델·데이터·native 조건을 고정하라

- 기존 pinned Llama-3-8B-Instruct와 실제 revision을 재대조.
- 유일한 편집 weight:
  model.layers.4.mlp.down_proj.weight
- W0 + zero native history에서 시작.
- 기존 fixed10k 봉인 순서 첫1000, B100.
- FP32/eager/TF32-off 및 기존 수치 계약 유지.
- Native local-L4 z/write algebra와 context weighting 유지.
- 각 arm의 자기 entry에서 request당 native z routine 1회.
- Correction용 추가 z 계산 0회.
- Reference는 기존 C4 Train512 전체.
- W0 생성 최대256, 실제 EOS 길이와 valid positions 사용.
- Dev128·공식 P/N·미래 요청은 selector에 제공하지 않음.
- 별도 paraphrase target set 생성 금지.
- GSS/recency/Past sampling은 이번 primary에서 off.
- History는 도착한 요청의 latest-valid canonical 전체를 사용.

C4 continuation은 W0 행동 기준이며 factual QA 정답 bank라고 부르지 말라.

4. T0를 통과한 뒤 scientific B1을 실행하라

다음을 고정된 소규모 기술 panel에서 확인하고 receipt를 남겨라.

- 실제 dense writer와 FP32 delta 재현.
- 고정 full-token key의 H0 + Delta K와 actual weight forward 일치.
- Q_E 공간 결속, Current response/logit invariant.
- Margin activation gradient와 계수 gradient의 direct AD/central FD 대조.
- Gradient basis 합의 원 방향 복원.
- 누적 교차항 계산과 B1의 STEP=CUM 일치.
- Full512 coverage, prefix/position, EOS/censor, argmax tie 처리.
- 동일 endpoint 반복 평가의 수치 변동과 계약 tolerance의 관계.

기술 panel 통과가 실제 candidate의 full512 검사를 대체하지 않는다. 성능을 보고 tolerance를 넓히지 말라.

과거 native/teacher/EN 결과는 W0·WN·order·context·P/history·tokenizer·runtime 수치 의미가 결속된 경우에만 재사용하라. 달라진 state의 gradient나 endpoint를 혼용하지 말라.

5. B1은 네 개의 고유 endpoint arm으로 수행하라

- N4: Native local-L4 baseline.
- EN-KL-Q: 기존 평균 KL, projected gradient 1회, actual 후보 최대4개.
- DEC-LINE: decision-risk gradient 한 방향.
- DEC-MODES-CUM: functional gradient 성분과 cumulative covariance 방향의 공동 계수 선택.

DEC-MODES-STEP은 B1에서 CUM의 alias다. 별도 gradient·후보 평가·endpoint 실행을 만들지 말라. Sequential용 state만 독립 복제하라.

같은 W0/B1 native와 Q_E, reference capsule을 공유하라. DEC-LINE과 MODES는 같은 endpoint의 reference derivative factors를 공유하라. 검증된 동일-state EN 산출물을 재사용할 수 있으면 EN을 다시 계산하지 말라.

EN↔DEC는 목적과 수용 정책의 결합 비교다. 순수 loss 하나만 바꾼 비교라고 주장하지 말라.

6. Decision method를 설계 그대로 구현하라

Reference의 고정 W0 prefix에서:

m_is = logit(y0_is) - max_{v != y0_is} logit(v)
mu_i = min_s m_is
Phi_R = mean_i max(0, -mu_i)^2

History는 entry 기준 safety slack으로 Phi_H를 정의한다.
방향 생성 위험은 Psi = Phi_R + Phi_H다.

- 모든512문서·모든 valid 생성 위치·full vocabulary를 검사.
- 문서별 최악 pair의 derivative A_i를 전체 valid input-token L4 output에 대해 계산.
- Dense weight gradient를 문서마다 저장하거나 CPU로 전송하지 않음.
- H = -grad(Psi) Q_E의 top3 성분과 residual로 최대4 functional 방향 구성.
- STEP 추가 방향: -Delta_native C_R Q_E.
- CUM 추가 방향: -(W_native-W0) C_R Q_E.
- 최종 Frobenius orthonormal basis rank는 최대5.
- Randomized/operator seed, span 복원 오차, 추가 방향 angle/rank를 기록.
- Dense full SVD와 불필요한 dense C_R 구축 금지.

J_ik = inner(A_i, D_k K_i)는 저장 factor contraction으로 계산하라. 이 Jacobian 때문에 두 번째 reference backward sweep을 실행하지 말라.

계수 문제는 계약의 two-phase convex QCQP와 lexicographic 최소 norm 조건을 적용하라. 반경·solver tolerance·zero-risk/zero-gradient·tie 분기는 계약대로 봉인하라.

중심점은 batch당1개다. 공동 해의 1, 1/2, 1/4, 1/8 scale에서 최대4 actual 후보만 검사한다. 실패를 보고 추가 gradient, 재선형화, 8후보 탐색을 자동 추가하지 말라.

7. Actual acceptance와 history 보호를 정확히 적용하라

각 후보는 immutable own-native에서 materialize하라.

수용에는 다음이 모두 필요하다.

- Current guard와 actual Q_E/response/logit invariant.
- 전체 active Past의 entry 기준 guard.
- Native에서 안전했던 모든 reference position의 신규 token-ID 불일치0.
- 각 문서 worst deficit의 계약상 비악화.
- Reference risk 비증가, combined risk의 의미 있는 감소.
- 전체 choice mismatch 수 비증가.
- Full512 및 실제 전체 valid positions 검사 완료.

Phi 감소와 실제 token-choice 복구는 분리하라. Flip 복구 없이 margin만 개선되면 MARGIN_ONLY로 기록하라.

History는 성공 여부와 무관하게 모든 도착을 registry에 기록하고, overwrite된 이전 version을 active guard에서 제외하라. Semantic registry와 native C_hist를 혼동하지 말라.

Past 기준을 post-native로 바꾸지 말라. Native가 이미 손상시킨 Past를 숨길 수 있다. Entry→native와 entry→selected를 모두 기록하라.

Finite search 실패와 증명된 local infeasibility를 구별하라. Native fallback이 Past 조건을 위반하면 FALLBACK_WITH_PAST_VIOLATION이며 보호 성공이 아니다.

8. 단계별 gate를 그대로 적용하라

B1→S3:

- Primary MODES의 유효한 비영 보정.
- Current invariant와 full512 coverage.
- 실제 reference choice 복구 또는 수치 오차를 넘는 Phi 상대감소≥5%.
- N4 대비 공식 PS/strict/joint 점추정 손실 없음.

Margin-only는 약한 신호로 표시하라. N 복구0도 명시하되 이것만으로 single-layer의 불가능성을 선언하지 말라.

통과하면 N4/STEP/CUM 세 chain을 각자의 B1 checkpoint에서 B3까지 이어라. 이후 각 arm은 자기 entry에서 z/write/history를 계산한다. 분기 뒤 다른 arm의 z/gradient를 재사용하지 말라.

B2에는 N4 own-native에서 STEP/CUM same-entry shadow probe 한 쌍을 수행하라. N4 chain에는 commit하지 않고 비용을 별도 집계하라.

S3→S10:

Primary CUM vs N4의 B3 all-seen 결과로 판정하라.

- Current/history/reference 조건 충족.
- PS/P-strict/R+두P joint 점추정 손실 없음.
- W0-correct N gross loss 감소 또는 실제 recovery 관측.

통과하면 STEP도 대조군으로 함께 B10까지 연장하라. STEP만 좋아진 것을 CUM 성공으로 대신하지 말라. Gate 실패 시 S10을 실행하지 말라.

정책이 같으면 기존 B1/B3를 반복하지 말고 이어서 실행한다. 방법·threshold를 바꿨으면 별도 version으로 W0부터 시작한다.

9. “NLL의 method gain”을 별도 필수 분석으로 추가하라

성공률이 같다는 이유만으로 NLL 개선0이라고 쓰지 말라. 반대로 일부 문항의 작은 개선을 전체 locality 개선으로 쓰지 말라.

Observer에서 다음을 분리 집계하라.

- 전체 N.
- W0에서 맞았지만 own-native에서 손상된 N.
- Entry에서 맞았지만 이번 native에서 손상된 N.
- 안정적으로 유지된 N.
- 이전 batch N의 at-write→현재 변화.

각 집합에서:

- 원래 정답 NLL의 entry/native/selected 값과 paired 변화.
- 새 편집 답변 NLL의 동일 변화.
- Margin = NLL(new) - NLL(true).
- NLL 개선/악화/동일 개수.
- 평균·중앙값·분위수, gross lost/recovered.
- 실패 경계까지 부족량과 실제 margin 이동량.
- Request-cluster paired uncertainty와 sample identity.

Arm 간 비교는 동일한 ID cohort로 수행하고, own-native 손상집합이 arm마다 다르면 분모 차이를 명시하라.

이 추가 분석은 observer 보고 항목이다. NLL 결과를 보고 candidate 선택·목적·gate를 변경하지 말라. 기존에 측정한 raw NLL을 재사용하고 불필요한 model forward를 추가하지 말라.

10. Single-batch 계산 비용을 독립적으로 관리하라

기존 EN의 약46.6분 correction을 그대로 반복하지 않도록 다음 경로를 확인하라.

- Decision candidate loop는 W0 token IDs를 사용한다. Full-vocab teacher 확률 파일을 읽지 않음.
- Full-vocab model logits와 competitor 검색은 계속 필요함.
- Immutable teacher/key/residual의 반복 SHA·finite·argmax·정규화 검증을 불필요하게 반복하지 않음.
- 검증 재사용은 검증한 bytes와 실제 소비 버퍼의 결속을 유지.
- 문서별 full weight gradient D2H 금지.
- A_i factor, Jacobian, basis의 같은-state 재사용.
- History의 실제 TF path·token·forward/backward·candidate guard 비용 별도 집계.
- Factor cache와 covariance action의 RAM/disk/I/O 예산 사전 계산.

진행 중인 runtime 최적화는 완료·parity 검증을 확인한 source로만 사용하라. 미완료 의존성이 있으면 구체적으로 보고하고 독립적인 CPU 구현·검증은 진행하라.

두 비용 회계를 함께 보고하라.

A. 전체 연구 실제 비용: 공통 계산은1회.
B. 각 method standalone 비용: 필요한 공통 derivative·geometry·factor/basis 계산을 전액 포함.

공통 비용을 arm 수로 나눠 method가 싸다고 보고하지 말라. 과거 EN wall-time과 새 runtime을 matched speedup으로 비교하지 말라.

S3의 median correction/own-native ≤2 기준을 별도 보고하라. 초과하면 QUALITY_SIGNAL_COST_UNRESOLVED로 표시한다. 이 기준을 임의의 실행 timeout으로 바꾸지 말라.

11. 기전 분석과 observer를 선택 이후에 수행하라

설계의 H1–H5 계측과 bounded intervention을 수행하라.

- 실제 writer metric의 spectrum과 target loading.
- Actual Delta 및 realization error.
- Reference response와 정확한 cumulative cross term.
- 제한된 global component intervention과 matched controls.
- 개입 시 Current 품질.
- B2 same-entry STEP/CUM과 실제 chain 차이.

Ridge writer의 gain을 단순1/sigma로 해석하지 말라. Activation norm 감소를 locality 개선과 동일시하지 말라.

Posthoc N panel은 mechanism-only다. 유리한 component를 이미 선택된 arm에 소급 적용하지 말라.

12. 필수 산출물과 완료 보고

Compact report와 metadata는 다음 계열 경로에 남겨라.

- experiment-reports/servers/<server>/single-layer-mechanism-first-20260919-v1/
- audits/servers/<server>/single-layer-mechanism-first-20260919-v1/
- experiment-reports/global/single-layer-mechanism-first-20260919-v1/

Raw/factor/weight/generation/대용량 로그는 ignored local 경로에 보관하고 manifest로 연결하라.

필수 산출물:

- execution-manifest.json
- preflight-and-parity.json
- stage-gates.json
- batch-metrics.csv
- neighborhood-nll-paired.csv
- reference-decision-summary.csv
- history-entry-native-selected.csv
- candidate-solver-ledger 및 compact index
- writer-mechanism-summary.csv
- same-entry-step-cum.csv
- compute.csv
- artifact-index.json
- terminal.json
- report-ko.md

NLL 문항별 raw는 local에 두고 공유본에는 집계·identity hash·재현 코드를 남겨도 된다.

최종 보고는 다음 순서로 작성하라.

실제 완료 범위 → 기술 유효성 → gate 결과 → RS/PS/NS·정답 NLL·복구량 → reference/Dev/history → standalone 비용 → 기전 해석과 한계.

설계의 성공을 미리 가정하지 말라. Reference만 개선됐는지, 독립 factual locality도 개선됐는지, 편집 성능을 유지했는지, 비용이 감당 가능한지를 각각 판정하라.

Gate 실패 결과도 숨기지 말고 완료 보고하라. Full10k, 추가 layer, GSS/recency, 별도 DEC-FUNCTIONAL arm은 이번 실행 범위에 자동 추가하지 말라.
