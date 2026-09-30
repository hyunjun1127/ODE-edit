# MEMIT-HJ: MEMIT-H의 원 목적을 전 층 공동으로 푸는 편집 method (v1)

작성 2026-09-30. 상태는 method 명세다. GPU 실행, job 제출, dispatch는 없다.
어느 버전을 먼저 실행할지, 무엇을 더 바꿀지는 실험 결과를 보고 정한다.

## 0. 요약

- **기반.** MEMIT-H를 쓴다. MEMIT에 과거 편집 key의 누적 history를 더한 형태이고 null space는 쓰지 않는다.
  편집 강도는 원 compute_z가, 실현은 MEMIT-H의 OLS solve가 그대로 정한다.
- **바꾸는 것.** 원 목적 밖에 붙어 있던 세 가지만 바꾼다.
  1. 층 target의 divisor 규칙 $R/(5-i)$를, 같은 OLS를 전 층 공동으로 푼 해로 대체한다(v1).
  2. clamp 경계에 고정되는 z 최적화를, 같은 compute_z 손실을 정확히 푸는 것으로 대체하고 clamp multiplier를 기록한다(v2).
  3. 위층 history에 남는 stale key를, drift가 key 자체의 해상도를 넘을 때 현재 key로 재구성한다(v3).
- **결과.** 층별 z(층별 실현 변위)가 원 목적의 공동해로 정해진다. 층 몫은 $\kappa_l=k_l^\top(\lambda C_0^{(l)}+H_l)^{-1}k_l$에 비례하는 명시적 규칙이 된다.
  MEMIT-H의 divisor는 이 규칙의 특수 극한이다.
- **검증.** 수식과 성질은 [`project/run_scripts/memit_hj/`](../../project/run_scripts/memit_hj/)의 NumPy 참조 구현과 CPU 검사 16개로 확인했다.

## 1. 근거

1. **논문.** [arXiv 2605.26670](https://arxiv.org/abs/2605.26670) (ICML 2026)은 AlphaEdit의 순차 안정성이 과거 편집 key를 누적해 one-time edit(OTE)과 순차 edit(SE)의 해를 일치시키는 데서 온다고 보인다.
   null-space 투영은 오히려 근사 오차를 누적시킨다. P = I로 두면 sequential MEMIT with history(Remark A.1)가 되며, 이것이 MEMIT-H다.
2. **repo 결과.** MEMIT-H(job 54007, fixed10k 10k)는 RS/PS/NS 95.52 / 85.17 / 61.55다.
   native AlphaEdit(73.43 / 62.89 / 55.29)보다 크게 높고, AlphaEdit-BLUE(98.88 / 95.78 / 63.73)와는 PS −10.6pp 차이가 남는다
   ([비교](../../experiment-reports/servers/server3/memit-history-fixed10k-20260928-v1/completion-review-r1/four-method-comparison-ko.md)).
3. **실현 부채.** Official MEMIT B10×10에서 L8 잔차는 pre-L5 0.950 → pre-L8 0.647 → post-L8 0.281이었다
   ([보고](../../experiment-reports/servers/server4/official-layer-realization-debt-sequential-b10x10-2026-09-01-v1/official-layer-realization-debt-sequential-b10x10-factual-ko.md)).
   층별로 받은 target의 25 / 24 / 30 / 39 / 57%만 실현됐다. MEMIT-H의 update 에너지 비중은 stream 내내 L8 46–48%, L4 약 10%다.
4. **clamp.** 09-17 L4 local-z fit 1000/1000에서 Adam step마다 clamp가 적중했다.
   최종 ‖δ‖/반지름은 1.000, NLL은 0.0011, decay는 0.121이었다.
5. **key drift.** B010 native BS1 100 edit 뒤 L8 key drift는 0.76이었고, 가장 낮은 쓰기 층인 L4는 0이었다.

## 2. 기준 절차: MEMIT-H (pinned)

[`memit_history_lifelong/method.py`](../../project/run_scripts/memit_history_lifelong/method.py)와 `hparams.json`에 결속된 설정은 다음과 같다.

- **하이퍼파라미터.** 층 L4–L8, λ(`mom2_update_weight`) 15000, C0는 Wikipedia mom2 100k다.
  clamp 0.75, lr 0.1, 25 step, `v_weight_decay` 0.5, `kl_factor` 0.0625, loss layer 31, `fact_token` subject_last다.
- **batch 절차.**
  1. $Z$ ← 요청별 compute_z(L8).
  2. $i=0..4$(L4..L8)마다 다음을 순서대로 한다.
     - $K_i$ ← compute_ks(현재 W)
     - $R \leftarrow Z - \text{cur\_zs}$ (L8, 현재 W)
     - resid ← $R/(5-i)$
     - adj ← $(\lambda C_0^{(i)}+H_i+K_iK_i^\top)^{-1}K_i$ (FP64)
     - $\Delta_i$ ← resid·adjᵀ
  3. 다섯 층을 쓴 뒤 post-write key로 $H_l \mathrel{+}= K_lK_l^\top$.

## 3. 원 목적의 전층 공동형

### 3.1 목적

MEMIT는 어느 층의 출력 변위든 L8 잔차에 그대로 더해진다고 가정한다. 모든 층이 L8에서 잰 R을 나눠 받는 이유가 이것이다.
같은 가정 아래 OLS를 층마다 따로 풀지 않고 전 층 공동으로 쓰면 다음과 같다.

$$\min_{\Delta_4..\Delta_8}\ \Big\|\sum_l \Delta_lK_l - R\Big\|_F^2+\sum_l \mathrm{tr}(\Delta_lA_l\Delta_l^\top),\qquad A_l=\lambda C_0^{(l)}+H_l$$

층 하나로 제한하면 MEMIT-H의 층별 OLS와 같은 항이며, 새 계수는 없다.

### 3.2 해

$G_l=K_l^\top A_l^{-1}K_l$(B×B, 양정치), $G=\sum_lG_l$로 두면 다음과 같다.

$$\Delta_l=R\,(I+G)^{-1}(A_l^{-1}K_l)^\top,\qquad D_l\equiv\Delta_lK_l=R\,(I+G)^{-1}G_l$$

미실현분 $R(I+G)^{-1}$은 보존과의 절충이며, 원 OLS와 같은 성격이다.

### 3.3 해석: 명시적 배분 규칙

- **층별 z.** $D_l$이 층 l의 z 변위이고, 공동으로 정해진다.
- **단일 요청.** $D_l=R\,\kappa_l/(1+\sum\kappa)$이다.
- **κ의 의미.** 층 l에서 변위 d를 실현하는 최소 보존 비용은 $\|d\|^2/\kappa_l$이다.
  그래서 κ_l은 그 key 방향이 보존 지식(C0)과 과거 편집(H) 대비 얼마나 비어 있는지를 뜻한다.
- **모든 층 사용.** $G_l\succ0$이므로 모든 층이 0이 아닌 몫을 받는다.
- **시간 의존성.** $H_l$이 커지면 κ_l이 줄고 몫이 다른 층으로 옮겨 간다. 배분의 시간 변화가 누적 history에서 나온다.
- **λ 의존성.** H = 0(W0)에서 층 몫의 비는 λ와 무관하다. λ는 전체 실현량에만 영향을 준다.

## 4. 구현형: 순차 재측정 (v1의 writer)

단계 i의 target만 다음으로 바꾸고, MEMIT-H의 solve는 그대로 호출한다.

$$\mathrm{resid}_i=R_{rem}\Big(I+\sum_{j\ge i}G_j\Big)^{-1}(I+G_i)\quad\Longrightarrow\quad \Delta_i=R_{rem}\Big(I+\sum_{j\ge i}G_j\Big)^{-1}(A_i^{-1}K_i)^\top$$

- **공동해와의 관계.** 가법성이 맞으면 §3.2의 공동해를 정확히 재현한다(귀납, CPU 검사).
  맞지 않더라도 각 단계는 "현재 측정으로 남은 층들의 공동 OLS를 풀고 현재 층 몫만 확정"하는 해와 같다. 재측정이 전파 오차를 흡수하는 방식은 MEMIT와 같다.
- **최상위층.** L8에서는 MEMIT-H와 같은 식이다.
- **divisor와의 관계.** MEMIT-H의 divisor는 $G_j=\kappa I$, κ→∞인 극한이다.
  층 i update의 MEMIT-H 대비 크기 비는 $(5-i)(1+\kappa)/(1+(5-i)\kappa)$이고, κ→∞에서 1이 된다. κ≈1이면 L4에서 약 1.7배다.
- **코드 변경.** pinned loop의 `resid = targets / (len(hparams.layers) - i)` 한 문장과 $G_j(j\ge i)$ 계산이 전부다.
  - $G_i$는 MEMIT-H가 이미 계산하는 adj에서 추가 solve 없이 얻는다: $F=K_i^\top\mathrm{adj}$, $G_i=F(I-F)^{-1}$.
  - $j>i$는 현재 상태에서 $K_j$를 compute_ks로 다시 계산한다. $A_j$는 batch 안에서 변하지 않으므로 Cholesky를 재사용해 $A_j^{-1}K_j$를 구한다.
  - B×B 역행렬은 FP64로 계산한다.
- **경량 변형(선택).** 위층 $G_j$를 batch 진입 key로 근사한다.
- **AST 확인.** pinned BLUE `memit_seq_main`의 해당 문장과 cov 항(`mom2_update_weight·cov + cache_c`)을 T0에서 AST로 확인한다.
  task-local adapter로만 교체하고 원본은 수정하지 않는다.

## 5. z 단계 (v2): 원 목적을 정확히 풀기

- **목적은 그대로.** 원 compute_z의 목적·제약·zero-step 규칙을 그대로 둔다.
  NLL(6 context) + 0.0625·KL + 0.5·‖δ‖/‖h‖², 제약 ‖δ‖ ≤ 0.75‖h‖, L(0) < 0.05면 δ = 0이다.
- **원 루프의 문제.** Adam step의 norm이 약 lr·√4096 ≈ 6.4로 반지름보다 크다.
  그래서 매 step 경계로 되돌려지고, 적힌 손실의 최소점이 아니라 경계점에서 멈춘다.
  반지름에서 decay 항만 0.12라 조기 종료도 걸리지 않고 항상 25회를 돈다.
- **참조 toy.** 로그와 같은 척도(NLL(0)≈6 nat, ‖h‖=3.15)로 만들었다. 결과는 다음과 같다.

  | | clamp 적중 | ‖δ‖/r | 손실 |
  |---|---|---|---|
  | 원 루프 | 24/24 | 1.000 | 0.120 |
  | 정확해 | – | 0.822 (내부) | 0.107 |

  (`test_native_loop_pins_to_clamp_while_optimum_is_interior`)
- **풀이.** spectral projected gradient(SPG)와 비단조 Armijo line search를 쓴다. 스케일에 무관하다.
  종료 조건은 projected-gradient norm ≤ 1e-8·max(1, ‖∇L(0)‖)이고, 최대 반복에 닿으면 `NOT_CONVERGED`로 기록한다.
- **손실 oracle.** [`z_hook.py`](../../project/run_scripts/single_layer_mechanism_first/z_hook.py)의 parity 검증된 native 손실을 L8에서 prefix 재사용으로 평가한다.
- **기록.** 요청마다 다음을 남긴다.
  - ‖δ‖/r, 경계 활성 여부
  - clamp multiplier $\nu_c=-\langle\nabla L,\delta/r\rangle$
  - NLL / KL / decay 값, 반복·평가 수
  - 같은 요청의 원 루프 해와의 손실 차
- **clamp 값.** 0.75는 원 hparam으로 유지한다. $\nu_c=0$이면 clamp는 결과에 영향이 없다. $\nu_c>0$이면 그 값이 clamp의 영향 크기다.

## 6. history 정합 (v3)

- **근거.** 논문의 OTE–SE 등가는 층의 key가 고정일 때 성립한다. 다층에서는 아래층 write가 위층 key를 움직이고, 가장 낮은 쓰기 층(L4)만 정확하다.
- **저장.**
  - batch마다 요청 5개(SHA 우선순위)의 post-write key를 층별로 보관한다. 10k에서 500 × 5층 × 14336 FP32 ≈ 143MB다.
  - 쓸 때의 context 간 분산 $\rho_{ctx}$도 저장한다. context별 key가 필요하므로, compute_ks가 평균 내기 전의 key를 돌려받는 helper를 둔다.
- **검사와 재구성.** 10 batch마다 저장 표본의 현재 key로 상대 drift를 잰다.
  층 l(L5–L8)에서 median drift > median $\rho_{ctx}$이면 $H_l$을 기존 요청 전부의 현재 key로 다시 만든다.
  구성원은 native대로 유지한다(덮어쓴 요청 포함). L4는 검사하지 않는다.
- **비용.** H200 key 계산이 약 5초/100요청/층이므로, 5k 요청이면 층당 약 4분이다.

## 7. 버전

| 버전 | MEMIT-H 대비 변경 | 확인 대상 |
|---|---|---|
| v1 | 층 target (§4) | 배분 규칙의 효과 |
| v2 | + z 정확 풀이 (§5) | clamp의 실제 영향과 편집 강도 |
| v3 | + history refresh (§6) | 다층 OTE 정합 |

모든 버전에서 다음은 MEMIT-H와 같다: 층, λ, clamp, KL·decay 계수, context, fact token, loss layer, FP64 solve, history append 시점.

## 8. 기록 항목

- **batch·층별.**
  - capacity $G_l$ 요약: 대각은 요청별 κ, 고유값은 min/median/max
  - capacity 몫 $\mathrm{tr}(G_l)/\mathrm{tr}(G)$
  - 실현 변위 $\|D_l\|$와 그 몫, update norm과 에너지 몫
  - L8 잔차 궤적 q(pre-L4 … post-L8, 실현 부채 연구와 같은 정의)
  - 층별 도달 이득 $\alpha_l=\langle\Delta R_{obs},D_l\rangle/\|D_l\|^2$(진단 전용)
- **요청별(v2).** §5의 항목.
- **층별(v3).** drift와 $\rho_{ctx}$의 median, refresh 사건.
- **표준 지표.** at-write RS/PS/NS, checkpoint all-seen RS/PS/NS(기존 evaluator).

## 9. 수치 gate

- **G1 parity.** `allocation=divisor` 모드가 같은 H200 환경에서 MEMIT-H job 54007 B1의 z·key·Δ·at-write bit SHA를 재현해야 한다. 다른 GPU라면 수치 허용 기준을 적용한다.
- **G2 identity.** batch마다 $\|\mathrm{adj}-A^{-1}K(I+G)^{-1}\|/\|\mathrm{adj}\|\le10^{-8}$이고, $G_l$의 최소 고유값이 0보다 커야 한다.
- **G3 최상위층.** L8의 resid가 $R_{rem}$과 비트 단위로 같아야 한다.
- **G4 유한성.** 모든 값이 유한해야 한다.
- **G5 (v2).** SPG 종료 조건을 충족하거나 `NOT_CONVERGED`로 기록한다. 손실 oracle은 z_hook parity 검사를 통과해야 한다.
- **G6 (v3).** refresh 직후 표본 key가 재구성에 쓴 key와 일치해야 한다.

## 10. 비용 (H200, MEMIT-H job 54007 실측 기준)

기준인 MEMIT-H 10k는 12.4 GPUh다. edit 7.0h(compute_z 5.3h, key 1.4h, solve 58초)와 평가 5.2h로 이루어진다.

| 버전 | 추가 비용 | 근거 |
|---|---|---|
| v1 | +약 1.4 GPUh (경량 변형 +약 0.6) | step마다 위층 key를 다시 계산해 batch당 key 호출 10 → 20, 약 +50초/batch |
| v2 | +1–10 GPUh | compute_z를 SPG로 대체. 반복 수에 따라 달라지며 smoke에서 실측 |
| v3 | +0.5–3.5 GPUh | 10 batch마다 4층 표본 검사 + refresh 빈도 |

## 11. 이 명세가 정하지 않는 것

아래는 실험 결과를 보고 정한다.

- **가법성 가정의 보정.** $\alpha_l$이 1에서 크게 벗어나면 몫을 $\alpha_l^2\kappa_l$로 보정하는 확장을 검토한다.
- **target 위치.** v1 뒤에도 BLUE와의 PS 차이가 남으면 L8 z냐 층별 z냐를 다룬다.
- **clamp 반지름.** $\nu_c$가 크면 반지름 처리를 다시 정한다.
- **λ의 값.**
- **refresh 설정.** 주기와 표본 크기, conflict 요청의 history 처리(논문 Prop 3.5).
- **실험 설계.** batch 크기(BS1, BS100)와 비교 대상.

## 12. 이전 논의와의 관계

- **B010 joint.** "층별 z를 공동으로 정한다"는 발상은 유지한다. 목적을 원 목적으로 되돌려, 약한 편집(NLL ≤ 1 경계)의 원인을 없앤다.
- **MJZ.** token 확률 조건, 잔존 margin μ(t), 측정 가격 π_l, AlphaEdit P/Q 기하는 뺀다. 가격 역할은 원 OLS에서 나오는 $G_l$이 맡는다.
- **routing(시간축 단층).** 층 선택 대신 전층 공동 배분을 쓴다. 명제 3(가장 낮은 쓰기 층의 key 정확성)은 v3에 반영된다.

## 13. 파일

| 파일 | 내용 |
|---|---|
| `project/run_scripts/memit_hj/allocation.py` | 공동 OLS 닫힌 해, 순차 target, divisor, MEMIT adj에서의 capacity 복원, 최소 비용 실현, 층 몫 |
| `project/run_scripts/memit_hj/zsolve.py` | 원 루프 재현(진단), SPG, zero-step 규칙 |
| `project/run_scripts/memit_hj/drift.py` | drift, context 분산, refresh 판정, history 재구성 |
| `project/run_scripts/memit_hj/test_memit_hj.py` | CPU 검사 16개 |

검사 실행은 `python3 -m unittest project.run_scripts.memit_hj.test_memit_hj -v`이다.
참조 구현은 NumPy로 쓴 수식 명세다. 실행 runner는 이를 torch(FP64)로 옮겨 pinned MEMIT-H 경로에 붙인다.
