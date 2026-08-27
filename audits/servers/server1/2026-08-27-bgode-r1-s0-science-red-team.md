# BGODE-R1 S0 수학·과학 red-team audit

- instruction: `ODEEDIT-S05-BGODE-R1-SEQUENCE-EVENT-BARRIER-GUIDED-ODE-SCIENCE-R1`
- audit scope: CPU/source S0 only
- 판정: **PASS WITH MODEL-EXECUTION HOLD**
- model/GPU/Slurm action: 0/0/0

## 공격적 질문과 판정

| 항목 | 공격 관점 | 증거 | 판정 |
|---|---|---|---|
| B>1 | 하나의 5-vector가 request별 equality를 어떻게 모두 만족하는가 | trie/moment/receipt에서 `B!=1` typed reject | PASS |
| event normalization | prefix 또는 길이가 다를 때 event가 겹치거나 빠지는가 | child/off-child exhaustive test, logsumexp sum1 | PASS |
| equal tokenization | source=target tokenization에서 두 leaf를 위장하는가 | constructor reject | PASS |
| numerical stability | tiny sequence probability가 underflow하면 normalization을 잃는가 | log-softmax accumulation, extreme logits fixture | PASS(S0) |
| termination tuning | endpoint가 좋은 boundary를 고르는가 | implementation은 typed frozen token만 받고 outcome selector 없음 | PASS; semantic lock OPEN |
| barrier identity | anchored excess를 off-manifold에서도 moving KL이라 부르는가 | on-manifold equality와 explicit negative test 분리 | PASS |
| Fisher claim | `G`를 exact anchored-KL Hessian이라 부르는가 | categorical Fisher–GN 명칭과 PSD test | PASS |
| velocity rationale | heuristic beta QP를 ODE라고 부르는가 | velocity Rayleighian 후 `beta=h*u`만 허용 | PASS |
| rank rescue | singularity를 ridge/damping으로 숨기는가 | deterministic eigentolerance; range/rank/denominator fail-close | PASS |
| gradient anchor | `g_t/g_0` equivalence를 penalty에도 확장하는가 | equality에서만 equality test, penalty negative fixture | PASS |
| pair mass | source-down을 main이 보장한다고 과장하는가 | pair-mass는 formula diagnostic/future branch | PASS |
| h subdivision | infeasible equality를 작은 h로 통과시키는가 | rank-deficient A가 모든 h에서 reject | PASS |
| native reduction | ambiguous controller root로 native를 재현하는가 | explicit `N=1,barrier-off,u=1,rho=1` bypass | PASS |
| ordered factor | controlled native writer와 동일하다고 부르는가 | native-ordered-rollout-derived dictionary + mismatch telemetry | PASS |
| history | Euler node가 current key를 history에 섞는가 | node append0/terminal 0 or1 receipt | PASS |
| fixed z | node마다 target을 다시 계산하는가 | compute1/recompute0 receipt | PASS |
| leakage | evaluator/locality가 action에 들어가는가 | forbidden counters0 | PASS(S0 interface) |
| barrier attribution | synthetic Full win을 model claim으로 승격하는가 | 문서에서 testability fixture로만 명시 | PASS |
| monotonicity/CBF | barrier 감소와 global safety를 보장하는가 | 명시적 nonclaim | PASS |
| ODE necessity | subdivision만으로 endpoint가 좋아진다고 주장하는가 | affine equality + nonlinear relinearization 분리 | PASS |

## 집중 test 결과

- command: `python -m pytest -q -W error project/run_scripts/barrier_guided_ode/tests`
- result: `29 passed`
- py_compile: PASS
- fixture type: CPU FP32 synthetic only
- broad unrelated regression: 0

## 남은 blocker

다음은 코드 결함이 아니라 model 실행 전 과학 정의다.

1. termination convention/tokenizer mapping
2. exact B=1 sealed sample unit
3. native `T_AE<=0` boundary
4. model JVP backend와 physical forward accounting
5. ordered dictionary rebuild의 measured compute/memory

이 다섯 항목 없이 S1을 실행하면 sample/termination/backend를 결과에 맞춰 선택할 수 있다.
따라서 현재 audit은 S0 source integration은 허용하지만 model load/GPU/Slurm은 block한다.

## Claim boundary

현재 열 수 있는 문장은 다음뿐이다.

> BGODE-R1의 B=1 event partition, forward-KL reference, Fisher pullback 및 equality
> Rayleighian은 CPU synthetic setting에서 내부 수학 identities와 typed failure boundary를
> 만족한다.

Full barrier의 model-level 우월성, standard locality, retention, generalization, classical
CBF 또는 monotone barrier는 열리지 않았다.
