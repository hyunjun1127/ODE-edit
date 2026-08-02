# Session 01 Motivation — cumulative capacity × canonical Alpha history closure

- 날짜: 2026-08-02
- 방법명: ODE-Edit
- 상태: locked execution spec
- 연구 단계: Motivation의 short-chain capacity/history 검증
- claim boundary: 4-edit common-policy directional diagnostic; lifelong 또는
  deployable method superiority가 아님

## 네 범주

### Proposal에서 온 내용

- 동일 direct-z를 실현하는 layer actuator 중 누적 capacity cost가 낮은 layer로
  write를 재배분할 수 있다.
- 현재 state에서 proposal과 layer responsibility를 다시 계산하되, rewrite-only
  information firewall를 지켜야 한다.
- AlphaEdit은 preserved-key null-space proposal로 확장할 후보이며, long-horizon
  claim 전에는 작은 sequential signal을 확인해야 한다.

### Repo/protocol에서 확인한 사실

- 기존 unconditional full-distance always-refresh 4-edit skeleton은 두 모델의
  current utility와 retention을 악화시켜 kill됐다.
- EasyEdit AlphaEdit의 canonical history는 accepted edit 뒤 다시 계산한 layer key
  `H_l`을 `cache_c[l] += H_l H_l^T`로 누적한다.
- Llama/Qwen용 Wikipedia covariance와 Alpha projector가 이미 존재하며 pinned
  read-only preflight가 두 모델 각각 25개 파일을 확인한다.
- server1 project cap은 4 GPU이고 이번 4-GPU/260000M 요청은 현재 cap check를
  통과한다.

### GH 추정

- full-distance refresh의 harm는 state refresh 자체보다 고정 전량 budget과
  cumulative load를 무시한 greedy allocation에서 생겼을 수 있다.
- 4 edits는 lifelong 증거가 아니지만, high-load layer 억제와 history-aware Alpha
  solve가 두 모델에서 같은 방향의 capacity/preservation signal을 만드는지 보는
  Motivation closure에는 충분하다.

### 사용자 확인 필요

- 없음. 사용자가 capacity-aware QP, canonical Alpha historical term, NFE 제거,
  두 모델 동시 실행을 직접 지시했다.

## 질문과 네 branch

동일 case order `['13899', '21196', '15624', '12487']`에서 다음 네 branch를
fresh process별로 실행한다.

1. `memit_native`: EasyEdit-compatible ordered MEMIT.
2. `memit_capacity_qp`: current-state MEMIT actuator + cumulative capacity QP.
3. `alpha_native_history`: canonical ordered AlphaEdit with branch-local history.
4. `alpha_capacity_qp_history`: current-state historical Alpha actuator + 같은 QP.

모델은 `llama3-8b-inst`, `qwen2.5-7b-inst`로 고정한다. Policy ID는
`capacity-qp-history-k3-trust-d4-v1`이며 모델별 분기나 rescue를 금지한다.

## Cumulative capacity QP

W0-normalized layer load는

\[
\Psi_l(S_l)=\frac{\lVert S_l\rVert_{C_l}^2}
{\operatorname{tr}(W_l^0C_l(W_l^0)^\top)}
\]

로 고정한다. Unit-C current actuator `b_l`과 non-negative coefficient `x_l`에
대해 다음 load는 정확히

\[
\Psi_l(S_l+x_lb_l)=\Psi_l(S_l)
+\frac{2\langle S_l,b_l\rangle_{C_l}}{D_l}x_l
+\frac{x_l^2}{D_l}
\]

이다. QP는 위 cumulative cross term을 포함한 next-load increment를 최소화하면서
rewrite-only central-probe slope `a_l`의 requested progress를 만족한다.

- 첫 trust radius: edit별 native C-distance의 `D/4`.
- 한 번의 retry: `D/8`.
- accepted round 최대 3회; 최대 누적 path는 `3D/4`.
- requested progress: 해당 trust/barrier에서 reachable maximum의 75%.
- accept: actual rewrite gain `>0` 및 trust ratio `>=0.05`.
- exact rewrite top-1 first-hit 뒤 즉시 종료.
- retry도 실패하면 이전 accepted endpoint에서 종료하되, 첫 round가 모두
  reject되면 fail-closed.

공통 frontier는 현재 five-layer `Psi` median에 equal-distance capacity increment
median 두 step의 headroom을 더해 만든다. Frontier보다 이미 찬 layer는 현재
load를 더 늘릴 수 없다. Candidate가 cumulative displacement와 anti-aligned라
load를 낮추는 경우에만 시작 load까지의 범위에서 허용한다. 따라서 단순히
작은 update를 쓰는 것이 아니라 **많이 찬 layer의 양의 추가 write를 직접
억제하고 feasible progress를 다른 layer로 넘긴다.**

## Canonical Alpha history

- upstream dense `cache_c=H H^T`와 정확히 같은 key-column `H`를 CPU float32
  branch-local bank로 유지한다.
- historical solve는 `G=[K,H]`, `U=P G`에 대해
  `U(lambda I + G^T U)^-1` Woodbury factor를 사용한다.
- 한 atomic edit의 ordered/QP round 동안 history는 고정한다.
- accepted endpoint가 실제 model에 commit된 뒤 각 layer의 current key를 한 번만
  append한다.
- native Alpha와 QP Alpha 모두 같은 history append policy를 사용한다.
- EasyEdit global `cache_c`, projector, source는 수정하지 않는다.

## 정보·artifact firewall

- controller 허용 정보: rewrite request, target, MEMIT five prefixes, current
  key/activation, pinned covariance/projector, branch-local past-key history.
- controller 금지 정보: paraphrase, neighborhood, generation, target-true outcome,
  evaluation metric.
- 모델별 네 controller가 모두 terminal/pass이고 16 action receipts가 검증된
  뒤에만 evaluator가 evaluation fields를 처음 decode한다.
- proposal tensor/raw log는 ignored `local/`에만 둔다. Git에는 code, spec,
  compact scalar analysis와 report만 남긴다.
- 새 experiment의 schema, gate, 분석과 결론에는 NFE metric을 사용하지 않는다.

## Metric과 lenient gate

각 edit checkpoint에서 다음 scalar만 평가한다.

- current edit utility, all-edit mean utility.
- prior-edit mean utility/success와 retention AUC.
- neighborhood/generation `KL(W0||Wt)`.
- target-true NLL drift.
- cumulative capacity sum, maximum layer share, layer-load Gini.
- cumulative Frobenius, accepted path distance, wall time, proposal/accept/reject count.

모델은 pooling하지 않는다. Family별 QP−native current utility mean delta가
`>= -0.10`이면 non-collapse로 본다. 그 상태에서 capacity/concentration 또는
preservation 축 중 하나가 양성이면 model signal이다. Pair gate는 Llama/Qwen이
동일 family에서 모두 non-collapse이고 최소 한 개 **동일 positive axis**를 가질
때만 통과한다.

- MEMIT과 Alpha-history 모두 통과: actuator-common signal.
- 한 family만 통과: family-conditioned signal.
- current utility가 한 모델이라도 floor 아래: harm signal.
- 공통 positive axis 없음: no-common-signal.

CI exclusion이나 lifelong effect size는 요구하지 않는다. 불리한 축도 전부
보고하며, gate를 본 뒤 model별 threshold/case/policy를 바꾸지 않는다.

## 실행·자원

- parent recovery job: `odeedit_capacity_history_pair_v3`.
- server1: 4 GPU, 32 CPU, 260000M, 12시간.
- controller phase: model × family 네 worker를 동시에 시작하고 worker당 native와
  QP controller를 순서대로 실행한다.
- global controller barrier 뒤 evaluator phase도 같은 네 worker를 동시에 시작한다.
- Llama 이후 Qwen을 제출하는 순차 pipeline은 금지한다.
- raw root: `local/results/raw/session01_motivation/`.
- log root: `local/logs/slurm/session01_motivation/`.

## 중단 조건

- session/Git/resource cap mismatch 또는 output collision.
- EasyEdit pinned file, covariance, projector hash mismatch나 recompute/write 시도.
- controller outcome leakage, receipt/barrier/state lineage failure.
- Alpha history partial/duplicate append 또는 edit count mismatch.
- QP non-finite/empty first action, exact replay mismatch, child nonzero.
- 한 model/family만 별도 재제출해야 하는 상황. Partial rescue는 하지 않는다.

## Job 15813 technical-recovery lock

- Original `c0_v1` job `15813`은 Qwen MEMIT worker가 exit code 2를 반환해
  parent fail-fast가 나머지 세 worker를 종료했으며 evaluator phase에 진입하지
  못했다. Partial artifact는 scientific evidence로 사용하지 않는다.
- Recovery는 model/family 네 worker를 다시 동시에 시작하는 full pair 한 번만
  허용한다. Case, order, model, branch, QP objective, policy hash, history semantics,
  metric, gate와 resource request는 바꾸지 않는다.
- `c0_v1` raw와 log는 삭제·덮어쓰기·resume하지 않는다. Recovery output과 marker는
  모두 `c0_v2`이며 job name은 `odeedit_capacity_history_pair_v2`다.
- Recovery-only 변경은 exception text를 저장하지 않는 sanitized code-location
  traceback과 worker별 local log 분리다. 이는 controller action이나 evaluation
  estimand를 바꾸지 않는다.

## Job 15817 probe-contract repair lock

- `c0_v2` job `15817`은 Llama MEMIT native가 끝난 뒤 첫 capacity-QP probe에서
  terminal `ContractError`를 냈다. Five layer-only actions를 넘긴 capacity path가
  quarter-step helper의 six-action default(layer five + uniform)를 명시적으로
  override하지 않은 integration mismatch다.
- Failure는 probe evaluation과 QP solve 전에 발생했으므로 scientific signal이
  아니다. v2 partial output도 evidence에서 제외하고 보존한다.
- v3의 유일한 scientific-code repair는 `_proposal_panel`에 optional
  `expected_action_ids`를 추가하고 capacity caller가 exact five layer IDs를 넘기는
  것이다. 기존 quarter-step six-action default와 QP layer-only policy는 그대로다.
- v3도 네 model×family worker를 동시에 시작하는 full pair 한 번만 허용한다.

## Job 15819 evaluator-only recovery lock

- v3 controller 8개는 모두 terminal/pass였으나, 실행 중 GH의 docs-only commit으로
  repository HEAD가 `9900a51`에서 바뀌어 evaluator가 첫 evaluation row decode 전에
  Git identity mismatch로 fail-closed했다.
- `9900a51`과 current HEAD의 capacity/history scientific code가 byte-identical이고
  evaluator output이 0개임을 확인한 경우에만 controller 재실행 없이 evaluator-only
  recovery를 허용한다.
- Recovery는 detached clean `9900a51` evaluator를 사용하고 두 모델×두 family를
  4 GPU에서 동시에 시작한다. Controller action, case/order, policy, history,
  metric, gate, run ID는 변경하지 않는다.
- `9900a51` evaluator의 local-path boundary 때문에 controller 8개는 detached
  worktree의 ignored `local/`로 exact-copy하며, symlink 부재와 recursive byte
  equality를 시작 전 검증한다. evaluator 결과는 표준 main-repo `local/`로
  exact-copy·재검증한 뒤 merge/analyze한다.
- 세부 근거와 중단 조건은
  `audits/global/2026-08-02-session01-capacity-history-job15819-evaluator-recovery.md`를
  canonical source로 삼는다.
