# BGODE-R1 CPU science core

이 package는 `BGODE-R1` S0의 pure CPU math만 제공한다. Model runtime, EasyEdit mutation,
weight transaction, evaluator, Slurm launcher는 포함하지 않는다.

## Module boundary

- `event_trie.py`: B=1 joint first-departure schema와 tiny-vocab brute evaluator
- `reference_path.py`: log-space single-coordinate reference, KL/decomposition, ceiling diagnostic
- `event_moments.py`: `S,a,b,g` aggregation
- `fisher_pullback.py`: symmetric categorical Fisher–GN pullback
- `rayleighian_controller.py`: deterministic equality KKT/pinv solver와 `beta=h*u`
- `alphaedit_actuator_interface.py`: 기존 ordered genuine AlphaEdit adapter의 interface guard
- `policies.py`: explicit native bypass
- `telemetry.py`: raw-free failure/counter receipts

## AlphaEdit reuse contract

Production provider는 기존
`project.run_scripts.ode_edit_motivation.alphaedit_proposal_adapter.AlphaEditProposalAdapter`
여야 한다. BGODE는 AlphaEdit solve를 복제하지 않는다. Accepted construction은
`genuine-p-inside-solve[/history]` + `ORDERED_GAUSS_SEIDEL`뿐이다. Post-hoc `B@P`는 reject한다.

## Future B=1 compute/memory estimate

기호:

- `L=5`: editable layers/actuators
- `Q`: joint trie internal prefixes
- `V`: vocabulary size
- `N`: Euler nodes
- `F_dict`: 한 ordered dictionary rebuild의 physical model forwards/captures

현재 correctness-first JVP plan은 Euler node마다 다음을 요구한다.

| component | node당 예상 logical work |
|---|---:|
| ordered factor dictionary | `F_dict`; 현재 adapter source상 layer별 key capture1 + terminal capture1, 즉 5+5 capture calls. 실제 physical graph 수는 profiler 전 OPEN |
| event primal probabilities | `Q` next-token prefix forwards |
| event directional score | `Q*L = 5Q` JVPs (serial reference backend) |
| moment reduction | `O(Q*V*L^2)` arithmetic |
| equality controller | eigendecomposition/KKT `O(L^3)=O(125)` |

따라서 correctness reference의 trajectory logical upper form은

`N * [F_dict + Q primal forwards + 5Q JVPs]`.

Dictionary/geometry를 freeze하는 one-step/frozen arms는 해당 rebuild와 JVP를 entry에서
1회만 수행한다. Native bypass의 controller/JVP/root count는 0이다. Actual physical count는
JVP batching/hook backend가 정해진 뒤 receipt로 재측정하며 FLOPs로 역추정하지 않는다.

Vocab-side streaming memory는 node별 `V x L` score buffer + `V` logit/probability + `L x L`
moment다. 예를 들어 `V=128,256`, `L=5`, FP32이면 최소 buffer는
`128256*6*4 = 3,078,144 bytes` (약 2.94 MiB)와 작은 5x5 matrix다. 이는 model activation,
KV/cache, ordered low-rank factors를 제외한 event reducer만의 값이다. Full event Python object
materialization은 production에서 금지한다.

## Runtime open definitions

- termination semantic lock
- sealed B=1 sample identity
- `T_AE<=0` rule
- serial/batched/hook JVP backend
- measured `F_dict`, activation peak, factor storage

이 항목이 닫히기 전 S1 model execution은 HOLD다.
