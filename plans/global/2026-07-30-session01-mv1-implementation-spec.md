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

## 10. Post-pilot v2 amendment — finite-action identifiability 교정

- amendment 일자: 2026-07-30
- 상태: **REVISE-before-full / v2 D0만 실행 허용**
- 근거 red:
  `audits/global/2026-07-30-mv1-finite-action-identifiability-postpilot.md`
- 우선순위: 이 절은 v2 실행에 한해 위 2, 4–7, 9절과 충돌하는 내용을
  대체한다. 위 절은 `mv1_*_c0p_v1`의 historical contract로 보존한다.

### 10.1 네 범주

#### Proposal에서 온 내용

- fixed direct-z와 same-snapshot layer proposal 아래 event별 allocation이
  static allocation보다 유리할 수 있는지를 작고 반증 가능한 Motivation
  diagnostic으로 검사한다.
- outcome을 보지 않는 rewrite-side signal만 controller에 허용하고, static
  policy가 충분하거나 signal이 actual progress를 예측하지 못하면
  routing/controller를 빠르게 kill 또는 pivot한다.

#### Repo/protocol에서 확인한 사실

- v1 3-case pair는 실행·artifact·rollback 측면에서는 PASS했다.
- v1에서 five single-layer unit directions를 `V_l`, 해당 central-difference
  slope를 `s_l`이라 하면 locked uniform direction의 local score는 선형성상
  `sum_l s_l / sqrt(5)`다.
- 양 model의 v1 세 case와 세 fraction에서 controller, retrospective
  best-static, six-arm event-wise winner가 모두 `uniform`이었다.
- Qwen은 세 fraction 모두 derivative p90 error gate에 실패했고, 특히
  finite-step magnitude calibration tail risk가 남았다.
- 기존 v1 pilot의 세 case는 v2 calibration의 첫 세 case와 겹치므로 v2
  estimand, action/static selection, uncertainty interval에서 모두 제외한다.

#### GH의 추정 및 확정 운영 결정

- positive slope가 여러 layer에 넓게 분포하면 layer별 heterogeneity가
  존재해도 `uniform`이 구조적으로 single-layer보다 높은 score를 갖는다.
  따라서 기존 six-arm oracle gap `0`은 “continuous routing opportunity가
  없다”를 식별하지 않는다.
- v2는 sparse single-layer winner를 찾는 대신, non-negative unit-`C`
  continuous mixture가 uniform보다 나은지를 직접 검사한다.
- 사용자의 claim-revealing Motivation diagnostic과 빠른 kill 지시에 따라
  pairwise/continuous mixture를 MV-1에서 열며, 이는 위 4절의 “MV-3 전
  pairwise mixture 금지”를 v2 D0/D1 범위에서 대체한다.

#### 사용자 확인 필요

- 없음. D0는 기존 calibration split 안의 다섯 case만 사용하고, positive
  결과도 GO나 MV-2를 승인하지 않는 제한된 diagnostic이다.

### 10.2 v2 action과 tie/fallback lock

각 event와 layer `l ∈ {4,5,6,7,8}`에 대해 unit-`C` direction `V_il`과
outcome 이전 central-difference slope `s_il`을 만든다. Event-adaptive
mixture weight는 다음으로 고정한다.

```text
if any_l(s_il > 0):
    w_il = relu(s_il) / ||relu(s_i)||_2
else:
    l* = argmax_l s_il
         # exact tie이면 더 낮은 layer
    w_il* = 1; other weights = 0

V_mix,i = sum_l w_il V_il
```

Layer parameter block은 서로 분리된 direct-sum `C` geometry이므로
`sum_l w_il^2=1`을 강제한다. Runner는 계산된 `V_mix`와 `uniform`의 실제
`C` energy가 같은 operational budget과 허용 오차 내 일치하는지
fail-closed assert한다. 임의 renormalization, negative coefficient,
outcome 기반 weight 수정은 금지한다.

Local-linear diagnostic은 다음이다.

```text
score_mix,i    = sum_l w_il s_il
score_uniform,i = sum_l s_il / sqrt(5)
```

Primary realized outcome contrast는 같은 event, snapshot, direct-z,
`q=1/256`, `C` budget에서:

```text
G_mix,i = P_i(V_mix,i) - P_i(V_uniform,i)
```

이다. `score_mix-score_uniform`은 prediction/identifiability diagnostic이고
`G_mix`를 대신하지 않는다. 기존
`{single_4,...,single_8,uniform}` panel은 sparse-vs-uniform sentinel로만
남기며 continuous routing 전체의 oracle 또는 kill estimand로 사용하지
않는다. `ordered_global_alpha`, `native MEMIT full`, `no-op replay`도
sentinel이며 그 contrast를 `G_mix`에 더하지 않는다.

### 10.3 D0/D1 sequential calibration

v2는 `q=1/256` 하나만 사용한다. Llama와 Qwen은 각 wave에서 하나의 paired
allocation으로 동시에 시작한다.

| wave | exact calibration slice | model별 case | 허용 결정 |
| --- | --- | ---: | --- |
| historical v1 | `[0:3]` | 3 | v2 estimand/selection/CI에서 제외 |
| `D0` | `[3:8]` | 5 | technical validation 및 continuous-routing early kill만 |
| `D1` | `[8:20]` | 12 | D0 생존 시 calibration 완료 및 policy freeze |

D0의 모든 planned event/arm을 ITD에 남긴다. Source/artifact mismatch,
teacher-forcing contract failure, non-finite, rollback mismatch, budget
mismatch, firewall violation, covariance recompute 시도, missing denominator는
research null로 바꾸지 않고 **technical BLOCK**한다.

Model별 replay/near-tie envelope는 결과를 열기 전에 다음으로 고정한다.

```text
e_m = max(1e-12, max_i abs(P_i(no_op_replay)))
```

`no_op_replay`의 exact logits hash와 모든 technical validity gate가 먼저
통과해야 하며, missing/non-finite replay를 `0`으로 대체하지 않는다. D0의
다섯 paired realized gain을 `G_mix,mi`라 할 때 아래가 동시에 성립하면:

```text
for both models m:
    all five D0 events satisfy G_mix,mi <= e_m
```

current continuous-routing direction을 early kill하고 D1, confirmatory,
MV-2를 열지 않는다. 한 model에서라도 한 event가 envelope를 넘으면 두
model을 같은 paired D1 wave로 확장한다. D0의 positive event는 오직 D1
실행을 허용할 뿐, positive claim, GO, confirmatory success 또는 MV-2
진입을 허용하지 않는다.

### 10.4 D0+D1 뒤 frozen static comparator

Confirmatory를 열기 전에 v2 calibration 17 case만으로 model별 static pooled
mix를 한 번 fit하고 hash와 함께 freeze한다. Pilot 세 case와 outcome은
사용하지 않는다. Outcome leakage를 막기 위해 fit 입력은 slope feature뿐이다.

```text
sbar_ml = mean_{i in D0 union D1} s_mil
w_static,m = relu(sbar_m) / ||relu(sbar_m)||_2
             # all-nonpositive이면 max-slope one-hot,
             # exact tie이면 더 낮은 layer
```

Raw slope scale를 model 간 pooling하지 않으며 model별 weight를 따로 고정한다.
Confirmatory primary는 event-adaptive `V_mix,i`와 frozen
`V_static,m=sum_l w_static,ml V_il`의 realized progress 차이다. 두 arm의
actual update `C` budget은 정확히 같아야 한다. Adaptive probe를 포함한
forward/solve NFE, wall-clock, GPU/host-memory peak를 arm별로 공개하고,
matched-NFE sensitivity를 별도 표기한다. Extra NFE를 숨기거나 effect와
latency를 임의 scalar로 합치지 않는다.

### 10.5 Pilot-only structural forecast와 금지된 해석

GH가 확정한 v1 slope-only pure-rotation 계산은 다음 descriptive reference로
만 남긴다.

| model | mean of casewise score ratios | ratio of mean scores |
| --- | ---: | ---: |
| Llama3-8B-Instruct | 약 `+0.71%` | 약 `+1.06%` |
| Qwen2.5-7B-Instruct | 약 `+1.22%` | 약 `+1.74%` |

이는 동일 local slope를 uniform에서 `relu(s)/L2` 방향으로 회전했을 때의
analytic score forecast다. Actual finite-step gain, expected ODE-Edit
improvement, CI 또는 population estimate가 아니다. 특히 Qwen의 큰 raw
slope-derived finite-step 값은 v1 p90 derivative failure가 보여 준
nonadditivity/calibration risk 때문에 보고하거나 gain으로 환산하지 않는다.
이 pilot 수치는 v2 selection, estimand, CI 어디에도 넣지 않는다.

### 10.6 핵심신호 우선 scientific gate amendment

사용자 지시에 따라 technical validity와 scientific signal의 문턱을 분리한다.
Rollback/source/hash/firewall/equal-`C`/outcome-leakage 위반은 계속
fail-closed `TECHNICAL BLOCK`이다. 반면 하나의 secondary metric, 한 event,
또는 CI endpoint 하나가 문턱을 못 넘었다는 이유만으로 핵심 routing signal을
kill하지 않는다.

- D1은 calibration과 static-policy freeze 단계이므로 technical validity가
  통과하면 독립 descriptive analysis까지 진행한다. D1 effect 자체로 GO 또는
  kill하지 않는다.
- Confirmatory primary는 사전 고정한
  `adaptive score_mix - frozen static mix`의 model별 paired realized
  progress다. Mean, 20% trimmed mean, median, positive-sign fraction과 paired
  CI를 함께 내되 primary estimand를 사후 교체하지 않는다.
- **명확한 continue**: 두 model의 mean이 replay envelope보다 양수이고 각
  model에서 `trimmed mean > envelope`, `median > envelope`,
  `sign fraction >= 0.55` 중 하나 이상이 같은 방향이면 핵심 signal이
  생존한다. 모든 event나 모든 secondary metric의 동시 통과는 요구하지 않는다.
- **architecture-conditional continue**: 한 model이 위 조건을 강하게
  만족하고 다른 model이 mean 기준으로 material하게 음수가 아니면, 작은
  bounded confirmatory/untouched 확인만 허용한다. 이를 cross-model
  generality로 보고하지 않는다.
- **회색지대**: mean 방향이나 robust summary가 엇갈리거나 CI가 envelope를
  가로지르면 즉시 kill/GO하지 않고, 이미 고정된 다음 confirmatory look 한
  번만 열어 분산과 outlier 민감도를 확인한다. Threshold/controller retuning은
  금지한다.
- **scientific kill**: 두 model 모두에서 primary mean과 20% trimmed mean이
  replay envelope 이하이고 sign fraction도 `<=0.50`이며, 같은 candidate
  panel의 oracle opportunity까지 null일 때만 current routing direction을
  kill한다. Oracle은 있으나 controller만 null이면 controller pivot이다.
- MV-2/ODE claim으로의 승격은 confirmatory와 untouched에서 primary 방향이
  재현되어야 하지만, 사전 지정 secondary metric 전부 또는 모든 case의
  양수를 요구하지 않는다.

이 amendment는 세부 threshold 실패로 실질 signal을 놓치는 것을 막되,
outlier 한두 개, 단일 model의 positive, calibration 재사용, outcome 기반
retuning으로 연구를 구제하지 못하게 한다.

### 10.7 Deprecated submission과 최소 감사

- 위 9절에 따른 기존 20-case `full C0` v1 wrapper/job은
  **deprecated / 제출 금지**다. 이미 존재하더라도 `D0` 결과 없이 실행하지
  않으며 v2 결과로 부르지 않는다.
- 새 실행은 `D0 [3:8]` pair를 먼저 제출하고, early-kill gate가 살아남을
  때만 별도 `D1 [8:20]` pair를 제출한다.
- 필수 감사·분석은 각 wave당 다음 최소 범위로 제한한다.
  1. v2 execution preflight red 1건
  2. model별 독립 post-run analysis 각 1건
  3. pair post-run red 1건
- 추가 감사를 자동 증식하지 않는다. Technical BLOCK, emergency triage,
  artifact mismatch가 생긴 경우에만 영향 범위에 맞춘 별도 기록을 연다.
