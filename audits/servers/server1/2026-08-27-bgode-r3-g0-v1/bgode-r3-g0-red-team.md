# BGODE-R3 G0 adversarial audit

This audit is CPU/source-only. It does not reinterpret or mutate R1/R2 and it
does not authorize G1/G2 execution.

| Risk | Gate | Result |
|---|---|---|
| Event conventions drift across topology | one layout identity consumed by probability, score, q0, KL, equality | PASS |
| Source-prefix macro destroys conditional covariance | source-exclusive tokens remain atomic; aggregate only for `sS` | PASS |
| Target-prefix target set is accidentally atomic | one target-exclusive macro row | PASS |
| Short prefix mislabeled as exact completion | specification uses prefix-resolved exclusive terminology | PASS |
| Tokenizer/output-head vocab mismatch | separate dimensions; distinguished IDs checked in both | PASS |
| Per-token Python event object blow-up | numeric tensor rows only | PASS |
| W0 conditional silently refreshed | tensor identity and layout identity rechecked on every reference use | PASS |
| Source aggregate substituted for q0 | q0 uses every non-target fine row | PASS |
| Probability stabilization changes science | no probability floor; conversion failure is typed | PASS |
| Target tilt changes non-target relative mass | q0 invariance test across four times | PASS |
| KL decomposition uses a different partition | exact moving and anchored identities on same layout | PASS |
| Old pair-mass equality leaks into main | equality is target logit plus normalized source conditional | PASS |
| Rank-one redundant equality rejected | complete consistency accepts the redundant zero-rate equation | PASS |
| Inconsistent equations silently switch mode | typed `EqualityInfeasible` | PASS |
| Factor scale controls Plain | all coordinates normalized by FP64 factor-Gram norm | PASS |
| Normalized coordinate is not the physical write | raw coefficient `beta/alpha` dense equivalence and block norm test | PASS |
| Ordered mismatch uses velocity instead of applied coefficient | API accepts applied `beta` only | PASS |
| R2 `range(G)` authority survives | no Gram inverse/eigendecomposition/range gate in R3 solver | PASS |
| Condition number is squared | direct SVD of `XZ` | PASS |
| Tiny FP32-unresolved modes promoted by FP64 | fixed `max(m,n)*eps_FP32` cutoff; retained direction FD interface | PASS |
| Full and Fisher differ at the reference origin | sealed-reference t=0 equality test | PASS |
| Barrier theorem claimed globally | only local derivative identity is specified/tested | PASS |
| Euler endpoint is forced | source has no endpoint correction/localizer interface | PASS |
| Node factor memory becomes trajectory state | explicit node release ledger, retained count zero | PASS |
| Existing adapter behavior changes | R2 focused legacy regression 21/21 PASS | PASS |
| GPU/model/Slurm action leaks into G0 | action counts all zero | PASS |

Scientific nonclaims remain binding: G0 says nothing about natural actuator
feasibility, endpoint efficacy, locality, finite-step monotonicity, or Full
beating Fisher. `scientific_promotion=false`.
