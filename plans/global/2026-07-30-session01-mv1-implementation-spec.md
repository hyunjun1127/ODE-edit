# Session 01 — MV-1 calibrated predictive heterogeneity 구현 명세

- 작성일: 2026-07-30
- 소유자: `head-server1-gh` (`global-head`)
- 상태: `implementation in progress; MV-0 two-model post-run PASS`
- 상위 plan: `plans/global/2026-07-30-session-01-motivation-validation.md`
- measurement lock: `audits/global/2026-07-30-motivation-expected-effect-forecast-audit.md`
- 목적: 작은 MEMIT probe로 routing opportunity와 outcome-blind controller의 회수 가능 효과를 분리 추정한다.

## 1. 네 범주와 claim boundary

### Proposal에서 온 내용

- fixed direct-z 아래 same-snapshot layer proposal을 local actuator로 비교한다.
- allowed rewrite context의 utility와 editor-native geometry만 controller에 허용한다.
- static allocation으로 충분하거나 utility가 actual progress를 예측하지 못하면 routing/ODE 방향을 kill 또는 pivot한다.

### Repo/protocol에서 확인한 사실

- canonical baseline은 local EasyEdit의 ascending-order Gauss–Seidel-style one-pass MEMIT이다.
- 고정 model은 아래 두 snapshot뿐이며 layer는 모두 `4,5,6,7,8`이다.
  - `meta-llama/Meta-Llama-3-8B-Instruct@8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`
  - `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28`
- CounterFact 100 case는 outcome-blind `case_id` hash로 `calibration 20 / confirmatory 60 / untouched 20`으로 고정된다.
- reusable direct-z, synchronous factor, factor-only `C` metric, rollback, exact stop, smooth utility가 있다.
- MV-1은 pinned moments만 read-only로 읽고 projector는 identity만 검증하며 deserialize하지 않는다.

### GH 추정

- MV-1의 최소 action set은 five single-layer와 one `C`-uniform arm이다. pairwise mixture는 MV-3 전 금지한다.
- primary utility는 안정적인 gradient를 위해 context/token mean으로 고정하고,
  exact worst margin은 별도 stop/safety metric으로 둔다.
- 현재 수치화 가능한 것은 `oracle ceiling`과 held-out `achievable allocation gain`뿐이다.

### 사용자 확인 필요

- 없음. GPU 시간과 effect를 합치지 않고 effect–NFE–wall-clock Pareto 표로 보고한다.

## 2. 데이터 lock과 sequential case ladder

동일 selection manifest를 양 model에 사용하고 row payload는 wave 시작 때만 four-field sanitizer로 읽는다.

| wave | exact case | 목적 | 허용 결정 |
| --- | --- | --- | --- |
| `C0` | calibration 20 | noise, metric, budget, action/static/controller lock | claim 금지 |
| `C1` | confirmatory `[0:20]` | 1차 futility look | early kill만 |
| `C2` | confirmatory `[20:40]` | 2차 futility look | early kill만 |
| `C3` | confirmatory `[40:60]` | primary final estimate | U 실행/kill/pivot |
| `U` | untouched 20 | exact locked replication | 최종 MV-1 판정 |

- C1/C2 positive 결과로 early GO하지 않는다.
- C0가 끝나면 code/policy hash, action order, tie/failure rule, bootstrap seed를 잠근다.
- confirmatory ID hash를 round-robin 배치해 5-fold 12 case씩 만들고 C0 전에 기록한다.
- U를 한 번 읽으면 더 이상 untouched로 부르지 않으며 retuning하지 않는다.

## 3. exact metric lock

각 context에서 `prefix + leading-space target` 전체를 tokenize하고 target token
suffix가 standalone target tokenization과 byte-identical ID sequence인지
검사한다. 입력은 마지막 target token을 제외하고, causal logit position
`prefix_len-1 ... prefix_len+T-2`를 target `0 ... T-1`에 맞춘다. right
padding, `[context,target_token,vocab]` layout, non-empty mask를 assert한다.
실패는 ITD `tokenization_contract_failure`이며 사후 제외하지 않는다.

`smooth_target_utility(..., temperature=1.0)`의 context/token별 값을 `u_cj`라
할 때 primary는 다음으로 고정한다.

```text
U(W) = mean_c mean_j u_cj(W)                 # higher is better
P_i(a) = U_after_i(a) - U_before_i
M_i(a) = min_cj(target_logit - best_other)   # detached exact margin
S_i(a) = all_cj(target is top-1 and margin >= 0)
```

- raw `P`는 model별 primary다.
- pooled sensitivity만 `s_m=max(MAD(C0 native-MEMIT P), C0 replay envelope)`로 표준화한다.
- target length/hash만 남기고 raw prompt, subject, target, full logits는 남기지 않는다.
- event가 statistical unit이며 layer/context/token을 sample로 세지 않는다.

## 4. exact C0 action set과 budget

각 event에서 frozen direct-z를 한 번 만들고 다음 두 factor set을 base state에서
독립 생성한다.

1. EasyEdit native ordered MEMIT factor: contextual baseline 및 budget scale
2. same-snapshot synchronous MEMIT factor `B_il`: MV-1 action source

Pinned covariance `C_l`로:

```text
E_native_i = sum_l ||Delta_native_il||^2_C
b_i         = q_m * E_native_i
Bhat_il     = B_il / ||B_il||_C
```

C0 candidate `q ∈ {1/256, 1/64, 1/16}`를 작은 것부터 평가한다. model별로
central finite difference sign concordance `>=0.80`, median relative derivative
error `<=0.25`, p90 `<=0.75`를 모두 만족하는 가장 큰 `q`를 고정한다.
near-zero denominator는 C0 replay envelope로 정하고 숨기지 않는다. 어떤
`q`도 통과하지 못하면 analytic utility calibration failure다.

동일 `b_i`의 locked set은 정확히:

```text
A_lock = {
  single_l: sqrt(b_i) * Bhat_il, l in {4,5,6,7,8},
  uniform:  sum_l sqrt(b_i/5) * Bhat_il
}
```

- `native MEMIT full`과 `global-alpha`는 contextual arm이며 oracle candidate가 아니다.
- fixed native delta `K`-step은 C0 sentinel에서만 endpoint equivalence를 재검증한다.
- `best static`은 C0 mean `P` 최대 arm이다. near-tie이면 single, 낮은 layer 순으로 고른다.

## 5. analytic utility와 controller

Qwen MV-0의 reserved peak가 약 46.8 GB로 A6000 여유가 작으므로 C0
primary는 backward graph를 만들지 않는 inference-only central finite
difference로 고정한다. 각 locked action의 unit-`C` direction을 `V_ia`,
operational distance를 `d_i=sqrt(b_i)`, probe distance를
`epsilon_i=d_i/4`라 하면:

```text
score_i(a) =
  [U(W + epsilon_i V_ia) - U(W - epsilon_i V_ia)] / (2 epsilon_i)
actual_i(a) = U(W + d_i V_ia) - U(W)
```

모든 `+/-/actual` branch는 `TemporaryLowRankApplication` 계열의 exact
rollback 아래 inference만 수행한다. `requires_grad`, model mode,
`use_cache`, RNG와 editable-weight hash가 branch 전후 동일해야 한다.
서로 다른 `q`에서 같은 signed distance가 나오면 evaluation을 hash-keyed
event-local cache로 한 번만 수행한다.

Primary frozen controller:

```text
action = argmax_{a in A_lock} score_i(a)
         # calibration near-tie이면 static fallback
```

Gradient scalar-gate는 C0 뒤 GPU headroom이 별도 검증될 때만 sensitivity로
열며 primary나 진입 조건이 아니다. 이 선택은 signed utility를 버리는 것이
아니라 finite-step local utility로 측정해 backward OOM confound를 제거한다.

confirmatory event는 feature/action JSON과 SHA256을 먼저 만들고 outcome을 연다.

별도 secondary `cross-fitted achievable`은 5-fold held-out 12-case씩 계산한다.
학습 rule은 shared non-negative utility slope와 six action intercept만
갖는 monotone least-squares calibrator로 고정한다. held-out fold outcome,
relation, eval field를 fit에 전달하지 않는다. U에서는 같은 rule을 누적
confirmatory 60 case에 한 번 fit한 뒤 action hash를 outcome보다 먼저 고정한다. frozen
primary와 cross-fitted estimate를 합치지 않는다.

## 6. estimand, CI, 기대효과

```text
O_mi       = near_tie_adjusted_max_{a in A_lock} P_mi(a)
G_oracle,m = mean_i [O_mi - P_mi(a_static,m)]
G_alloc,m  = mean_i [P_mi(f_m(x_i)) - P_mi(a_static,m)]
```

- `G_oracle`은 candidate-set retrospective upper bound일 뿐 method gain이
  아니다.
- `G_alloc`만 최초 achievable forecast다. cross-fitted 값은 별도 열이다.
- model별 mean/95% paired CI, median, sign fraction, 20% trimmed mean, reach-rate pp, denominator를 낸다.
- 여러 primary contrast는 Holm correction한다. pooled sensitivity는 같은
  case ID를 두 model에서 함께 resample한 equal-weight summary일 뿐이다.
- oracle captured fraction은 denominator CI가 replay floor에서 분리될 때만
  secondary로 낸다.
- MV-1에서 refresh gain, displacement→retention 변환, long-horizon 개선률,
  두 model 밖 backbone 일반화는 모두 금지한다.

## 7. early kill과 next-stage gate

C0에서 one-sided 99% paired-bootstrap futility boundary와 seed를 고정한다.
C1/C2 누적 look에서 **두 model 모두** 아래를 만족할 때만 조기 종료한다.

```text
UCB99(G_oracle) <= replay/near-tie envelope
AND UCB99(G_alloc) <= replay/near-tie envelope
```

최종 판정:

- `routing kill`: 두 model 모두 near-tie-adjusted oracle upper bound가 null
- `current controller kill/pivot`: oracle은 있으나 두 model `G_alloc`이 null
- `architecture-conditional`: 한 model만 positive; pooled 값으로 rescue 금지
- `U 승인`: C3에서 양 model `G_oracle`, frozen `G_alloc`의 Holm-adjusted
  lower CI가 replay envelope보다 큼
- `MV-2 advance`: U에서 양 model frozen `G_alloc` sign이 유지되고, C3+U
  locked estimate도 positive. 이때만 refresh incremental experiment를 연다.

Technical failure, rollback mismatch, source/artifact mismatch, recompute guard,
firewall violation은 research null로 바꾸지 않고 fail-closed `block`한다.
pre-satisfied, all-nonpositive, target-not-reached, arm failure는 ITD row에
남긴다.

## 8. 최소 구현 단위와 artifact

다음만 추가/확장한다.

1. `teacher_forcing.py`: causal target-position contract와 inference utility
2. `covariance_bundle.py`: run-start 1회 SHA/size/shape/count 검증, CPU read-only
   moment bundle; EasyEdit `COV_CACHE/layer_stats()` 우회 및 recompute 차단
3. `hooks.py`: scaled low-rank factor branch와 exact state restoration
4. `mv1_heterogeneity.py`: split-scoped runner, action precommit, ITD trace
5. `mv1_forecast.py`: oracle/frozen/cross-fit/bootstrap/futility CPU analysis

Raw output `local/results/raw/session01_motivation/<run_id>/` 아래
`manifest.json`, `features.jsonl`, `actions.jsonl`, `outcomes.jsonl`,
`summary.json`, `direct_z/`만 둔다. Git에는 policy/fold hash, compact metrics,
artifact checksum, 재현 command만 둔다.

CPU tests는 case-ID-only split/fold, multi-token position, forbidden field,
gated-hook/actual parity, finite difference, no-recompute bundle, decision-before-
outcome, cross-fit fold mask, failure denominator, exact rollback을 포함한다.

## 9. paired model submit envelope

각 wave는 Llama 종료 뒤 Qwen을 올리지 않고 하나의 paired allocation으로
동시에 시작한다.

```text
job: odeedit_mv1_<wave>_pair_v1
node/partition: devbox / gpu
allocation: --nodes=1 --ntasks=2 --cpus-per-task=8
            --gres=gpu:a6000:2 --mem=130000M --time=12:00:00 --export=NONE
child: srun --exclusive --exact -N1 -n1 -c8 --gres=gpu:1 --mem=65000M
run IDs: mv1_llama_<wave>_v1, mv1_qwen_<wave>_v1
```

- 제출 전 GH session `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`, exact CWD/repo를 검사한다.
- `scripts/check-slurm-resource-cap.sh server1 2 130000`, clean
  `origin/main`, CPU tests, `bash -n`, `sbatch --test-only`, wave별 red PASS가
  모두 필요하다.
- 허용 write는 위 local run/log/state path뿐이다. Slurm 제출은 사용자
  time-critical 지시에 따른 GH 최소 예외로 wave별 audit에 명령·영향·후속
  report를 기록한다.
- moments/projector 계산·download, EasyEdit 수정, eval field 전달, raw
  prompt/log/weight Git 유입, 다른 Codex session/repo 조작은 금지한다.
- raw artifact 생성 후 active peer가 없으므로 broadcast exception을 summary와
  report에 기록한다.
- 실행과 다른 analysis agent 및 red post-run이 동시 실행, cap, firewall, denominator, action hash를 확인한다.
