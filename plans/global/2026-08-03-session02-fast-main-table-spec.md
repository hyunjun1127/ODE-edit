# Session 02 — Fast Main Table Structural Lock

- 작성: **2026-08-03 KST**
- GH: `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`
- 상태: **`SUPERSEDED_BY_COMPUTE_AWARE_V2; DO_NOT_EXECUTE`**
- primary method: **ODE-Edit**
- primary actuator: **MEMIT only**
- models: `Llama3-8B-Instruct`, `Qwen2.5-7B-Instruct`
- source review SHA-256:
  `5ab23b222cc64c523f62a5769d58aca4fb4f049f5a6b66b7df7c288e0df4fa7c`

이 문서는 초기 fast-track 설계 기록이다. 계산량을 co-primary objective로 올린 뒤
[`2026-08-03-session02-compute-aware-main-table-spec.md`](2026-08-03-session02-compute-aware-main-table-spec.md)가
primary execution spec으로 대체했다. 이 문서의 평균 refresh 목표, 별도 screen 생략,
5-arm 100-edit matrix 또는 execution gate를 실행에 사용하지 않는다.

## 1. 판정

첨부 리뷰의 큰 결론은 수용한다.

- MEMIT으로 ODE identity를 먼저 식별한다.
- 1--2 edit technical smoke 뒤 별도 4-edit screen 없이 10-edit table로 간다.
- 10-edit gate가 살면 같은 다섯 arm으로 즉시 100-edit main table을 만든다.
- AlphaEdit, signed-capacity hard barrier, solver family 비교, full downstream,
  1K+는 첫 main table 뒤로 미룬다.
- Scalar first-hit과 Ordered adaptive를 강한 단순 baseline으로 추가한다.

다음 두 항목은 그대로 쓰지 않고 보정했다.

1. monotone load는 micro-step write 제곱합이 아니라 **completed outer edit의 terminal net
   write**를 edit당 한 번만 누적한다. 그래야 step subdivision이 비용을 인위적으로
   줄이지 않는다.
2. target-vs-old event는 서로 다른 object 길이 bias를 막기 위해 sequence
   log-likelihood를 token 수로 normalize한다.

## 2. Primary question

> Dynamic state-refreshed joint execution이 Native, 단순 scalar early-stop, static
> synchronous routing, 같은 controller의 ordered adaptive execution보다 100-edit
> prior-retention frontier를 두 모델에서 공통으로 개선하는가?

Capacity/load proxy는 mechanism metric이다. Main success criterion은 current acquisition,
prior-retention AUC/final, paraphrase, neighborhood, compute다.

## 3. Fixed target와 event

- direct-z는 outer edit entry state에서 정확히 한 번 계산하고 arm 안에서 고정한다.
- Sequential arm의 model state가 갈라진 뒤 direct-z tensor를 arm 사이에 강제로 공유하지
  않는다. 대신 request, target, direct-z hparams와 compute-once contract를 동일하게 둔다.
- controller context는 canonical direct-z construction에 쓰는 rewrite prompt/prefix로
  제한한다.
- evaluation paraphrase, neighborhood, held-out retention, downstream outcome은 action
  freeze 전에 접근할 수 없다.

허용 context \(c\), new object \(o^\star\), old object \(o^{\rm old}\)에 대해

\[
\ell_c(o;W)=\frac1{|o|}\sum_j\log P_W(o_j\mid c,o_{<j}),
\]

\[
\Phi_{\rm event}(W)
=\max_c\left[\ell_c(o^{\rm old};W)-\ell_c(o^\star;W)\right].
\]

`Phi_event <= 0`의 first hit에서 종료한다. Tokenization, leading-space 처리, empty/one-token
edge case와 context list/hash는 numerical lock에 명시한다.

## 4. Primary controller

모든 edit layer proposal은 같은 current snapshot에서 unit-\(C\) actuator로 만든다.

\[
\widehat B_l=\frac{B_l}{\|B_l\|_{C_l}+\epsilon}.
\]

Completed outer edit \(i\)의 terminal net write \(U_{i,l}\)로

\[
\Omega_{t,l}
=\sum_{i<t}\frac{\|U_{i,l}\|_{C_l}^2}{D_l},
\qquad
D_l=\|W_{0,l}\|_{C_l}^2+\epsilon_D
\]

를 계산한다. Current edit 안에서는 \(\Omega_{t,l}\)를 freeze하고 first-hit terminal에서
net write를 한 번만 append한다.

Current rewrite slope \(a_l\ge0\), trust radius \(h\)에서

\[
\bar p=\max_{y\ge0,\ \|y\|_2\le h}a^\top y,
\qquad
p=\min\{\kappa[\Phi_{\rm event}]_+,\ \beta\bar p\}
\]

를 정하고

\[
\min_y\sum_l(1+\Omega_{t,l})\frac{y_l^2}{D_l}
\]

subject to

\[
y\ge0,\qquad
a^\top y=p,\qquad
\|y\|_2\le h,\qquad
y_l=0\ \text{if }a_l\le\epsilon_a
\]

를 푼다. Solver output \(y\)가 실제 applied coefficient이며 post-QP rescale은 없다.

Signed base-relative \(\Psi\)와 cross term은 diagnostic으로만 기록한다. Primary controller에
hard layer capacity envelope를 두지 않는다.

## 5. Five-arm contract

| Arm | Frozen/refreshed state | Applied rule | 식별 반론 |
|---|---|---|---|
| Native MEMIT | canonical ordered state | canonical full write | EasyEdit baseline |
| Scalar first-hit | native terminal layer writes frozen | common scalar \(\alpha\in[0,1]\) first-hit | 덜 쓰기만 하면 되는가 |
| Static synchronous | entry same-snapshot proposal/allocation frozen | same trust/event, frozen direction/share | static routing이면 충분한가 |
| Ordered adaptive | current state refreshed | canonical ascending cyclic coordinate, one layer per micro-step | controller 효과인가 joint synchronous 효과인가 |
| Full ODE-Edit | all layers current-state refreshed | joint QP coefficient and joint trial | full method |

### Scalar first-hit

Entry state에서 canonical ordered MEMIT layer writes를 계산한 뒤 exact rollback하고
\(\Delta W_{\rm MEMIT}\)을 freeze한다. Prelocked increasing \(\alpha\) grid에서 첫
event-hit bracket을 찾고 그 안에서만 bisection해 \(\alpha\in[0,1]\)의 first hit를
근사한다. \([0,1]\) 안에서 hit가 없으면 `scalar_event_fail`로 기록하고
\(\alpha>1\) rescue를 하지 않는다. Event non-monotonicity와 grid resolution을 기록한다.

### Static synchronous

Entry same-snapshot의 \(B_l,a_l,\Omega_l\)와 layer share를 freeze한다. Current
\(\Phi_{\rm event}\)와 common trust/first-hit만 global magnitude를 바꿀 수 있다.

### Ordered adaptive

Canonical ascending layer order를 cyclic하게 돈다. 각 coordinate 직전에 current
residual/key/proposal/slope를 다시 만들지만 현재 layer 하나만 nonzero coefficient를
가질 수 있다. Cycle이 끝나면 이미 방문한 layer도 새 joint state에서 다시 방문할 수
있다. Full과 event, load, trust, rollback threshold가 같다.

### Full ODE-Edit

모든 layer proposal을 같은 current snapshot에서 만들고 joint QP \(y\)를 trial한다.
Accept 뒤 모든 layer를 다시 선형화하고 first-hit까지 반복한다.

## 6. Fast execution matrix

### MT — technical identity

| Model | Family | Fresh edits | GPU arms |
|---|---|---:|---|
| Llama | MEMIT | 1--2 | Native, Full |
| Qwen | MEMIT | 1--2 | Native, Full |

나머지 세 baseline은 deterministic/unit contract와 artifact schema를 함께 통과한다.
Scientific 비교는 하지 않는다.

### MI — ODE identity

| Model | Family | Fresh edits | Orders | Arms |
|---|---|---:|---:|---|
| Llama | MEMIT | 10 | 2 | five-arm contract |
| Qwen | MEMIT | 10 | 2 | five-arm contract |

두 모델 job은 같은 submission batch로 올리고 같은 case/order manifest를 사용한다.

### MS — first main table

| Model | Family | Fresh edits | Orders | Arms |
|---|---|---:|---:|---|
| Llama | MEMIT | 100 | 2 | five-arm contract |
| Qwen | MEMIT | 100 | 2 | five-arm contract |

MI gate 통과 즉시 같은 source/config로 stream length만 preregistered 100으로 확장한다.

## 7. MI mechanism panel

10-edit에서 다음만 별도 panel로 보고한다.

- proposal drift: \(\cos_C(B_l^{s+1},B_l^s)\)
- layer ranking turnover: Spearman/Kendall of \(a^{s+1},a^s\)
- allocation drift: cosine of \(y^{s+1},y^s\)
- selected pair ordered non-commutativity
- first-hit round, accept/reject, trust ratio
- \(\Omega\), signed \(\Psi\), max-load share, Gini
- NFE/edit, proposal build count, wall time/edit

Fixed K-split, dynamic-share-only, dynamic-direction-only는 main table을 늦추지 않도록 MI
primary arm에서 제외한다.

## 8. MI gate

Exact numerical tolerance는 outcome 0건 상태의 numerical lock에 기록한다. Structural
gate는 다음과 같다.

1. Full current acquisition non-collapse against Native on both models
2. Full-vs-Scalar와 Full-vs-Static prior-retention AUC가 어느 모델에서도 material
   negative가 아니고 pooled paired direction은 positive
3. Full-vs-Ordered가 두 모델 공통으로 material negative가 아님
4. proposal/ranking/allocation drift가 numerical noise보다 큼
5. average refresh `<=2` 또는 추가 NFE를 정당화하는 retention frontier signal
6. model-specific threshold/event/step/sign/fallback 없음

Kill/pivot은 다음과 같다.

- `Full ~= Scalar`: ODE를 중단하고 scalar first-hit 연구로 pivot
- `Full ~= Static`: dynamic refresh/ODE necessity claim kill
- `Full ~= Ordered`: joint synchronous execution claim 제거
- field/ranking stationary: static routing으로 pivot
- \(\Omega\) 감소와 retention 악화: load geometry claim kill/redefine
- 한 모델만 살리는 rescue 필요: common method gate fail

## 9. First main table

| Method | Current rewrite ↑ | Prior-retention AUC ↑ | Final prior retention ↑ | Paraphrase ↑ | Neighborhood ↑ | Max-layer \(\Omega\) load ↓ | NFE/edit ↓ | Time/edit ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

보조 표에는 checkpoint perplexity/lightweight downstream proxy, signed \(\Psi\), Gini,
direct-z fidelity와 endpoint displacement를 둔다. Evaluation paraphrase/neighborhood는
controller action 뒤 별도 evaluator에서만 연다.

Main success priority는:

1. current acquisition non-collapse
2. Scalar/Static 대비 prior-retention AUC
3. final prior retention
4. paraphrase/neighborhood non-worsening
5. Full-vs-Ordered joint execution value
6. compute overhead

\(\Omega\), \(\Psi\), Gini는 main functional success를 대신하지 않는다.

## 10. Deferred work

첫 main table에서 제외한다.

- AlphaEdit/projector/history extension
- signed \(\Psi\) hard barrier
- direct-z refresh
- Euler/Heun/RK 비교
- formal Lie bracket sweep
- all-vocabulary top-1 terminal
- full downstream suite
- 1K/10K/full stream
- model-specific rescue

MEMIT 100-edit에서 Full이 Scalar/Static보다 살아남은 뒤 AlphaEdit 100-edit를 같은 controller
rule로 연다. 1K+는 MEMIT/필요 Alpha extension 뒤에만 연다.

## 11. Numerical lock와 implementation handoff

SH1은 outcome을 생성하지 않고 다음만 제안한다.

- exact case IDs와 두 order hash
- tokenization/context manifest
- smooth surrogate \(\tau\)
- \(h_0,\kappa,\beta,\eta_{\rm reject},\gamma_\downarrow,\gamma_\uparrow,S_{\max}\)
- QP/equality/event tolerance
- scalar search iterations/tolerance
- artifact schema와 five-arm command
- expected NFE/time/memory

GH가 이 값을 numerical lock으로 승인한 뒤에만 MT GPU를 연다. MT는 technical identity만
판정하며 numerical lock을 scientific outcome에 맞춰 조정하지 않는다.

## 12. Execution boundary

- EasyEdit source write 금지; ODE-edit-side reusable hook만 허용
- precomputed covariance read-only; download/recompute 금지
- raw output/log/generation은 ignored `local/`
- Git에는 source, test, small manifest, compact Korean report만
- Llama 완료를 기다리지 않고 Qwen을 같은 batch에 제출
- SH1: primary implementation/MT/MI/MS
- SH2: locked-source independent reproduction와 second order
- 각 model 결과 뒤 별도 Terra Ultra analysis agent report; runtime mismatch는 block
- current state: Slurm HOLD until SH boundary and numerical lock pass
