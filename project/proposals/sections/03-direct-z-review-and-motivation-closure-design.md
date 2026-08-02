# Direct-z 다각도 검토와 Motivation 최종 종료 실험 설계

- 날짜: 2026-08-02
- 방법명: **ODE-Edit**
- 문서 상태: **GH design draft — review/design only; 실행·Slurm 제출 없음**
- 현재 결론: direct-z atomic 가능성은 제한적으로 양성이지만, direction refresh와
  layer-share refresh의 개별 기여는 아직 식별되지 않았다. Motivation의 마지막
  실험은 이 두 축만 분리하며, capacity-QP·Alpha history·long-horizon 평가는
  Method Session으로 넘긴다.

## 범위와 agent 검토 상태

사용자 지시에 따라 direct-z 관련 canonical report, small analysis artifact와
ODE-Edit-side 구현만 검토했다. 과학 해석, 구현 의미론, closure/red 관점의
Terra Ultra 격리 agent 세 개를 시도했으나, 세 agent 모두 runtime metadata에서
`gpt-5.6-terra / ultra`를 명시적으로 확인할 수 없어 protocol에 따라 repo를
전혀 읽지 않고 종료했다. 따라서 아래 내용은 **독립 agent review가 아니라 GH의
근거 대조 결과**다. Sol agent를 독립 리뷰어로 대체하거나 Terra 확인을 추정하지
않는다.

이번 문서는 실험을 구상하는 데까지만 권한을 사용한다. 코드 구현, EasyEdit
수정, artifact 생성, Slurm 제출, GPU 예약, 기존 raw 결과 변경은 하지 않는다.

## 네 범주

### Proposal에서 온 내용

- Direct-z는 semantic goal이고, 같은 goal을 실현하는 parameter endpoint는 여러
  개일 수 있다.
- ODE-Edit의 핵심 후보는 current state에서 layer proposal direction과 layer
  responsibility를 다시 계산하는 것이다. 고정 update를 여러 조각으로 나누는
  것만으로는 다른 endpoint가 생기지 않는다.
- Controller는 rewrite-only information만 사용해야 하며 paraphrase, locality,
  downstream prompt는 평가 전용이다.
- Utility ranking이 stationary하거나 static layer scaling이 동등하면 ODE
  relinearization을 kill하고 더 단순한 routing으로 pivot해야 한다.

### Repo/protocol에서 확인한 사실

- Direct-z fixed panel은 모델별 8 cases를 fresh `W0`에서 독립 평가한 atomic
  diagnostic이다. `K=4`는 네 sequential fact가 아니라 한 fact 내부 waypoint다.
- Direct-z는 case마다 한 번 고정되고, current BF path는 매 hop proposal direction과
  layer coefficient를 함께 다시 계산한다.
- Layer coefficient는 rewrite-only central-probe slope `a_l`에 대해
  `relu(a) / ||relu(a)||_2`를 사용하고, 모두 non-positive이면 best one-hot으로
  fallback한다. 이는 probability simplex가 아니고 cumulative capacity를 보지
  않는 greedy unit-L2 share다.
- 각 operational hop은 native C-distance `D`의 `D/4`, probe radius는 `D/64`다.
  모든 hop을 사용하며 first-hit, trust-ratio accept/reject, capacity QP, cumulative
  `Psi_l`, persistent sequential history는 구현되지 않았다.
- Genuine Alpha direct-z panel은 precomputed projector를 solve 안에서 사용했지만
  `cache_c=0`인 isolated atomic geometry다. Canonical sequential Alpha history를
  시험한 것이 아니다.
- Native는 ordered layer execution이고 BF는 synchronous re-proposal path이므로,
  기존 BF-minus-native에는 direction refresh, share refresh, ordered-vs-synchronous
  scheduler와 endpoint normalization이 함께 묶여 있다.
- 별도 4-edit MEMIT microseq에서 unconditional full-distance always-refresh는
  Llama/Qwen 모두 current utility와 retention을 악화시켰고, capacity와
  neighborhood KL proxy는 두 모델에서 개선했다. 이 skeleton은 이미 kill됐다.

근거:

- `experiment-reports/global/2026-08-01-direct-z-possibility-memit-alpha-p0-final-synthesis.md`
- `experiment-reports/global/2026-08-01-direct-z-possibility-memit-alpha-cross-track-p0-v2.analysis.md`
- `experiment-reports/global/2026-08-02-session01-microseq-pair-m0-v1-gh.md`
- `experiment-reports/global/2026-08-02-session01-motivation-closure-gh.md`
- `project/run_scripts/ode_edit_motivation/quarter_step_refresh.py`
- `project/run_scripts/ode_edit_motivation/mv1_score_mix.py`
- `project/run_scripts/ode_edit_motivation/alphaedit_proposal_adapter.py`

### GH 추정

- Alpha에서 관측된 BF lift는 단순 radial scaling보다는 endpoint direction의
  재배치와 일관되지만, 현재 bundled contrast만으로 그 원인을 direction refresh로
  단독 귀속할 수 없다.
- Preservation 악화는 direct-z 자체의 실패라기보다 rewrite-only greedy objective,
  fixed full distance, 또는 `C` geometry와 held-out KL geometry의 불일치 중 하나로
  설명될 수 있다. 현재 자료는 이 원인들을 식별하지 않는다.
- Motivation에서 full capacity controller를 구현해 구제하려 들면 mechanism
  diagnostic과 Method 성능 검정을 섞게 된다. 마지막 Motivation 실험은 bundled
  effect를 분해하는 데 그쳐야 한다.

### 사용자 확인 필요

- 이 turn에서는 사용자가 “실험 구상만”을 지시했으므로 실행 권한은 없다.
- 아래 설계를 실제로 구현·제출하려면 별도 실행 지시가 필요하다.
- Repo의 `project/proposals/sections/01-motivation-validation.md`는 2026-07-31의
  `closed-negative / NO MV3` 상태를 유지하지만, 2026-08-02 terminal report는
  `atomic mechanism survives / sequential skeleton killed`로 갱신돼 있다. 실행 전
  canonical proposal narrative를 최신 terminal evidence에 맞춰 정합화해야 한다.

## 다각도 결과 검토

### 1. 실제로 layer별 가중치를 다르게 주었는가

그렇다. 다만 이는 **sequential edit history에 따른 누적 layer capacity 가중치**가
아니다. 한 atomic edit의 각 waypoint에서 현재 rewrite slope를 다시 probe하고,
그 양수 부분을 L2-normalize해 layer 4--8의 비중을 정한다. 동시에 layer proposal
direction도 다시 만든다. 따라서 기존 결과는 “가중치 재계산”과 “방향 재계산”을
분리하지 못한다.

### 2. Eff/Gen/Loc가 모두 나빠졌는가

아니다. 기존 report의 metric은 full benchmark 표준 Eff/Gen/Loc가 아니라 다음과
같은 local proxy다.

| 비교: BF minus same-family native | Llama | Qwen | 해석 |
|---|---:|---:|---|
| Alpha z-residual 감소 | `+0.02056762` (7/8) | `+0.08233046` (6/8) | 두 모델 양성 |
| Alpha output-NLL reduction | `+3.78883688` (8/8) | `+0.05692602` (7/8) | Eff에 가까운 proxy 양성 |
| Alpha paraphrase-NLL reduction | `+3.26029124` (8/8) | `+0.88091911` (7/8) | Gen에 가까운 proxy 양성 |
| Alpha preservation score (`-heldout KL`) | `-0.14487684` (0/8) | `-0.20097198` (1/8) | Loc에 가까운 proxy 악화 |
| Alpha off-token spill 감소 | `-0.03609012` (0/8) | `-0.01056363` (2/8) | Loc proxy 악화 |
| MEMIT z-residual 감소 | `-0.13500684` (0/8) | `-0.05456161` (2/8) | 두 모델 악화 |
| MEMIT output-NLL reduction | `+0.16067749` (8/8) | `+0.00002117` (2/8) | Llama만 명확 |
| MEMIT paraphrase-NLL reduction | `+0.80229617` (6/8) | `+0.20916645` (7/8) | 두 모델 양성 |
| MEMIT preservation score (`-heldout KL`) | `-0.09631979` (0/8) | `-0.08699755` (2/8) | 두 모델 악화 |

즉 Alpha BF에서는 Eff/Gen-like proxy가 좋아지고 Loc-like proxy가 나빠졌다.
MEMIT은 더 혼합되어 있다. “모든 성능이 나빠졌다”도, “전체 성능이 좋아졌다”도
정확하지 않다.

### 3. AlphaEdit의 소수 edit 열세를 고려했는가

부분적으로만 고려됐다.

- 올바르게 고려된 부분: primary 해석은 Alpha BF와 Alpha native의 **within-family
  paired lift**이고, Alpha-BF가 MEMIT-BF보다 절대적으로 우수하다고 주장하지
  않는다. 실제 absolute z/edit endpoint는 두 모델 모두 MEMIT-BF가 더 좋았다.
- 아직 고려되지 않은 부분: direct-z Alpha는 `cache_c=0`인 isolated first-edit
  geometry라 canonical sequential Alpha history를 평가하지 않았다. 따라서
  “AlphaEdit은 적은 edit에서 약할 수 있으나 history가 쌓이면 달라진다”는 질문은
  이 panel로 답할 수 없다.
- 결론: Motivation gate에서는 Alpha의 절대 MEMIT 우위를 요구하지 않는다.
  향후 sequential Method 검정에서는 canonical history-on Alpha를 baseline으로
  써야 하며, no-history direct-z 수치를 sequential baseline처럼 재사용하면 안 된다.

### 4. 현재 허용되는 claim

허용되는 최대 문장은 다음과 같다.

> Fixed atomic panel에서 direction과 layer share를 함께 refresh한 genuine-Alpha
> path는 same-C one-pass native보다 두 모델에서 더 나은 direct-z actual-write와
> local edit proxy를 선택했다. 이 결과는 preservation, sequential retention,
> capacity-aware routing, BF-over-DF 또는 deployable method superiority를 보이지
> 않는다.

## Motivation 최종 종료 실험: `M-CLOSE-DZ-2x2`

### 질문

기존 direct-z 양성 신호가 current-state **direction relinearization** 때문인지,
current-state **layer-share 재계산** 때문인지, 또는 둘의 interaction인지 분리한다.
이 질문 외에 capacity-QP, history, long-horizon 성능을 Motivation으로 다시 열지
않는다.

### 기존 A/B/C 실험과 중복되지 않는 이유

기존 quarter-step 및 adaptive A/B/C는 teacher-forced rewrite utility에서
`refreshed direction + refreshed share`, `fixed direction + refreshed share`,
`fixed direction + fixed share`를 비교했다. 반면 direct-z fixed panel은 actual-weight
z fidelity, generated-delta, edit, preservation을 측정했지만 BF와 native의
direction, share, scheduler 차이가 묶여 있었다. 새 실험은 **direct-z panel의 동일
case·target·metric 위에서** 네 번째 cell인 `refreshed direction + fixed share`까지
추가하고, 모든 cell을 같은 shadow state에서 평가해 이 묶음을 해제한다. 이전
teacher-forced 결과를 새 direct-z 결과처럼 재사용하지 않는다.

### 고정 범위

- 모델: `llama3-8b-inst`, `qwen2.5-7b-inst`.
- actuator family: MEMIT, genuine AlphaEdit.
- case: 기존 direct-z replay-lock의 모델별 8 cases를 그대로 사용한다. 이는 기존
  effect의 causal de-bundling이지 fresh-case generalization 검정이 아니다.
- target: 기존 frozen direct-z tensor를 read-only replay하며 recompute하지 않는다.
- layer: 4--8.
- geometry: 기존 precomputed covariance와 Alpha projector를 read-only 재사용한다.
- EasyEdit: source 수정 금지. ODE-Edit-side hook만 사용한다.
- budget: 기존 panel을 재현하도록 case별 locked reference C-distance `D_case`를
  그대로 쓴다. MEMIT은 ordered-native distance, Alpha는 paired MEMIT-BF에서
  lock된 reference distance다. Temporary hop은 `D_case/4`, sensor radius는
  `D_case/64`다. Family 사이의 `D_case` 정의를 새로 통일하거나 retune하지 않는다.
- Alpha: atomic question이므로 `cache_c=0`을 유지하고 결과를 isolated Alpha
  mechanism으로만 표기한다.
- 모델별 threshold, sign, branch, case selection, rescue tuning은 금지한다.
- 새 schema와 gate에는 `NFE`를 넣지 않는다.

### Common-shadow-state 2x2

먼저 기존 locked runner의 두 endpoint를 그대로 재현하는 `R-native`와 `R-BF`
sentinel을 둔다. Sentinel은 기존 report와 code path가 바뀌지 않았는지만 확인하며,
아래 factorial effect에 합치거나 sample 수로 세지 않는다. 기존 casewise scalar와
arm/proposal/target hash가 preregistered tolerance 안에서 맞지 않으면 2x2를
해석하지 않고 technical fail로 종료한다.

Static `B0, v0` path가 만든 동일 shadow state `W_s`에서 아래 네 temporary
counterfactual hop을 평가하고 매번 exact rollback한다. 각 cell이 자기 rollout로
다른 state distribution을 만들게 하지 않는다. 따라서 primary estimand는
**static-control trajectory에 조건부인 local causal effect**이며, 네 개의 독립
full rollout 사이 policy-level 우열이 아니다.

`B_s`는 `W_s`에서 한 번 만든 current-state basis이고, `v_s`는 그 `B_s`의
rewrite-only probe panel에서 한 번 결정한 share다. 같은 `v_s`를 B와 A cell에,
같은 `v0`를 C와 D cell에 재사용한다. Cell마다 share를 다시 최적화하면 두 factor가
다시 묶이므로 금지한다. 네 cell의 proposal/state/target/action hash를 먼저 고정한
후 outcome을 읽으며, outcome을 보고 다음 shadow state나 action을 고르지 않는다.

| Cell | Direction basis | Layer share | 식별 대상 |
|---|---|---|---|
| C | `B0` fixed/transported | `v0` fixed | static split control |
| B | `B0` fixed/transported | `v_s` refreshed | share-only effect |
| D | `B_s` refreshed | `v0` fixed | direction-only effect |
| A | `B_s` refreshed | `v_s` refreshed | bundled BF effect |

`s=0`에서는 `B_s=B0`, `v_s=v0`라 treatment contrast가 퇴화하므로 primary
estimand는 `s=1,2,3`에서 계산한다. 세 waypoint 값을 먼저 case 안에서 평균하여
case당 하나의 paired effect를 만들고, 그 뒤 8 cases를 집계한다. Waypoint를
독립 sample처럼 세어 pseudo-replication하지 않는다.

각 cell의 score는 동일 `W_s` 대비 one-hop improvement로 정의한다. 예를 들어
direct-z score는 W0 denominator를 고정한
`(||h(W_s)-z*||-||h(W_s+hBv)-z*||)/||h(W0)-z*||`다. Sign-normalized score를
`S(B,v)`라 하면 다음을 model/family/metric별로 계산한다.

\[
\begin{aligned}
E_{dir} &= \tfrac12\{S(B_s,v_0)-S(B_0,v_0)
                     +S(B_s,v_s)-S(B_0,v_s)\},\\
E_{share} &= \tfrac12\{S(B_0,v_s)-S(B_0,v_0)
                       +S(B_s,v_s)-S(B_s,v_0)\},\\
I_{dir\times share} &= S(B_s,v_s)-S(B_s,v_0)-S(B_0,v_s)+S(B_0,v_0).
\end{aligned}
\]

### Metric

- representation: direct-z residual reduction, generated-delta error.
- Eff-like: rewrite output-NLL reduction, output progress, exact margin.
- Gen-like: paraphrase-NLL reduction.
- Loc-like: heldout KL preservation score, off-token spill.
- contract: per-hop C-energy, exact rollback, proposal/state/target hash, outcome
  firewall, precomputed-only attestations.

Eff/Gen/Loc-like 표기는 full CounterFact benchmark 표준 metric과 혼동하지 않는다.
Controller action은 rewrite prompt와 기존 MEMIT prefixes만 보고 고정하며,
paraphrase/heldout outcome은 모든 cell specification과 hash가 commit된 뒤에만
읽는다.

### Lenient하지만 terminal한 gate

각 model을 따로 판정하고 pooling하지 않는다. CI exclusion은 요구하지 않는다.

1. **Technical pass:** 두 모델 동일 policy/config hash, direct-z recompute 0,
   precomputed covariance/projector only, exact rollback, finite metric, outcome
   firewall가 모두 통과해야 한다.
2. **Locked reproduction sentinel:** `R-native`와 `R-BF`가 기존 locked
   casewise result/hash를 재현해야 한다. 이는 technical gate이며 새 과학 signal이
   아니다.
3. **Local bundled contrast:** common-shadow `A-C`의 direct-z residual과 rewrite
   output-NLL을 보고한다. 기존 Alpha 양성 결과와 같은 방향이면 mechanism의
   local 재현으로 보되, 원래 `BF-native` contrast의 exact reproduction이라고 부르지
   않는다.
4. **Direction survives in one family:** `E_dir`의 direct-z residual과 rewrite
   output-NLL이 Llama와 Qwen 각각 mean `>0`, positive cases `>=5/8`이어야 한다.
5. **Actuator-generic claim:** 위 조건을 MEMIT과 genuine Alpha가 모두 통과할 때만
   허용한다. Alpha만 통과하면 `Alpha-geometry-conditioned`, MEMIT만 통과하면
   `MEMIT-conditioned`로 제한한다.
6. **Static-routing pivot:** direction gate는 실패하지만 `E_share`가 두 모델에서
   같은 기준을 통과하면 ODE relinearization을 kill하고 layer-share routing만
   남긴다.
7. **Preservation boundary:** Loc-like axis가 음수이면 atomic mechanism만 남기고
   preservation/sequential claim은 계속 kill한다. Loc 양성도 lifelong 근거로
   확대하지 않는다.
8. **Full kill:** direction과 share가 모두 model-common gate를 실패하거나 모델
   sign이 갈리면 cross-model ODE Motivation을 종료한다. 모델별 rescue는 없다.

### 이 한 번 이후의 terminal decision

| 결과 | Motivation 종료 판정 | 다음 단계 |
|---|---|---|
| `R` sentinel 실패 또는 local `A-C` sign 미재현 | Motivation negative | 종료; factor 해석·rescue 금지 |
| Direction이 두 family·두 model에서 생존 | partial-positive, actuator-generic atomic mechanism | 별도 Method Session에서만 constrained controller 검정 |
| Alpha에서만 두 model 생존 | partial-positive, Alpha-conditioned atomic mechanism | MEMIT-general claim 제거 |
| MEMIT에서만 두 model 생존 | partial-positive, MEMIT-conditioned atomic mechanism | Alpha/null-space general claim 제거 |
| Share만 두 model 생존 | ODE relinearization negative, routing-only pivot | static/dynamic coefficient 연구로 단순화 |
| 둘 다 실패 또는 model sign 불일치 | Motivation negative | 종료; 추가 retune/새 case 금지 |

어느 결과가 나오더라도 Motivation에서 case 수 확대, `K` sweep, threshold 변경,
모델별 branch, 16--32 edit rescue를 하지 않는다.

표의 첫 행에서 말하는 재현은 locked `R` sentinel의 technical reproduction과
common-shadow `A-C`의 local sign을 모두 만족한다는 뜻이다. 둘을 같은 estimand로
합치지 않는다. 이 한 번 뒤에는 결과가 양성이든 음성이든 **atomic Motivation을
terminal하게 닫는다**. Canonical Alpha history, capacity-QP와 4-edit retention은
살아남은 mechanism을 method로 만들지 판단하는 별도 Session이며 Motivation을
재개하는 구제 실험으로 쓰지 않는다.

## 선택하지 않은 실험과 이유

- **Capacity-QP + first-hit + trust-ratio:** 기존 always-refresh harm를 고치기 위한
  실제 Method 구현이다. Motivation의 마지막 causal question보다 범위가 크다.
- **Canonical Alpha history sequential panel:** Alpha의 소수-edit 및 history 효과를
  평가하는 데 필요하지만 atomic direct-z de-bundling과 다른 질문이다. Method
  Session에서 history-on native Alpha와 history-on ODE-Alpha를 paired 비교한다.
- **Lifelong/100+ edit:** 현재 atomic `n=8/model` 및 4-edit harm evidence에서 바로
  확장할 근거가 없다.
- **Absolute Alpha-vs-MEMIT superiority gate:** Alpha의 small-edit 출발점 차이와
  null-space trade-off를 method effect로 오인하므로 사용하지 않는다.

## 기대 방향과 kill risk

기존 evidence에 기반한 GH의 방향성 예측은 다음과 같다.

- Genuine Alpha의 `E_dir`은 Llama/Qwen에서 양수일 가능성이 가장 높다.
- MEMIT의 `E_dir`은 mixed일 가능성이 높아 actuator-generic claim은 위험하다.
- Eff/Gen-like gain과 Loc-like loss가 다시 분리될 가능성이 높다.
- 숫자 개선폭은 bundled contrast에서 direction-only effect를 식별할 수 없으므로
  예측하지 않는다.

예측과 달라도 gate를 바꾸지 않는다. 특히 Alpha만 양성이면 이를 MEMIT/Llama/Qwen
전체 method superiority로 확대하지 않는다.

## 향후 실행 envelope 초안 — 현재는 비활성

- 목적과 배경: 위 `M-CLOSE-DZ-2x2` causal de-bundling만 수행.
- 허용 write path: reusable code는 `project/run_scripts/ode_edit_motivation/`, raw와
  full log는 `local/`, compact metric/report만 Git 경로.
- Slurm 제출: **현재 not allowed**. 별도 사용자 실행 지시 뒤 preflight 필요.
- 자원 초안: server1에서 Llama/Qwen을 동시에 1 GPU씩, 총 2 GPU. Project cap 4를
  넘지 않으며 host memory request는 active local cap check를 다시 통과해야 한다.
- red gate: evaluation firewall, same-model-policy, cache read-only, exact rollback,
  resource cap 중 하나라도 block이면 제출 금지.
- artifact broadcast: 실제 run이 생길 때만 protocol의 `local/` broadcast 규칙 적용.
- 완료 보고: 별도 run metadata, 모델별 analysis, pair synthesis와 post-run audit.
- 금지: EasyEdit 수정, covariance/projector/direct-z 재계산, raw Git commit,
  모델별 rescue, 기존 artifact overwrite.
- 예상 산출물: 네 cell의 model/family별 paired scalar table, main/interaction effect,
  terminal Motivation verdict.
- 중단 조건: technical contract 실패, model policy hash 불일치, resource cap 실패,
  non-finite metric, rollback 또는 outcome firewall 실패.
