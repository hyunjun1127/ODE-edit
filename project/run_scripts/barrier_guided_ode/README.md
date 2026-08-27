# BGODE-R1 science core and S1 binding

이 package는 `BGODE-R1` S0 pure CPU math와, 그 수학을 기존 AlphaEdit adapter에 결속하는
S1 단일-request pilot binding을 제공한다. S1은 B=1만 허용하며 production AlphaEdit solve를
복제하지 않는다.

## Module boundary

- `event_trie.py`: B=1 joint first-departure schema와 tiny-vocab brute evaluator
- `reference_path.py`: log-space single-coordinate reference, KL/decomposition, ceiling diagnostic
- `event_moments.py`: `S,a,b,g` aggregation
- `fisher_pullback.py`: symmetric categorical Fisher–GN pullback
- `rayleighian_controller.py`: deterministic equality KKT/pinv solver와 `beta=h*u`
- `alphaedit_actuator_interface.py`: 기존 ordered genuine AlphaEdit adapter의 interface guard
- `policies.py`: explicit native bypass
- `telemetry.py`: raw-free failure/counter receipts
- `s1_contract.py`: 봉인된 B1 request000, model별 고정 termination 및 여섯 arm 순서
- `s1_streaming_events.py`: vocab event를 Python object로 펼치지 않는 `V x 5` reducer
- `s1_alphaedit_runtime.py`: fixed-z authority, serial forward JVP/FD gate, exact W0 transaction
- `s1_experiment.py`: 한 model load에서 여섯 arm을 순차 실행하는 observation-only evaluator binding

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

## S1 closed runtime definitions

- Llama termination=`<|eot_id|>` id128009 단일 boundary
- Qwen2.5 termination=`<|im_end|>` id151645 단일 boundary
- sample=canonical Phase123 B1 ordinal0(case19795)
- horizon=`T_AE=r_AE-r0`; nonfinite/nonpositive이면 typed terminal boundary
- correctness backend=prefix별 5방향 serial forward-mode JVP, 첫 node central-FD gate
- Native/Plain/Fisher/Full/One-step/Frozen 여섯 arm 고정 순서
- endpoint evaluator는 arm action freeze 뒤만 실행되고 controller return path에는 연결되지 않음

Qwen matched extension은 동일한 runtime loop, sample bytes/order/context, Claim-A horizon,
여섯 arm 순서와 JVP/FD gate를 재사용한다. 모델별로 달라지는 값은 봉인된 HF snapshot,
Official AlphaEdit artifacts와 outcome-free termination token뿐이며 별도 source/result/log/state
namespace를 사용한다.

Dictionary 내부의 physical forward 수는 source-level logical capture와 분리하여
`NOT_RESOLVED_SOURCE_LEVEL_HELPER_INTERNALS`로 기록한다. 이를 wall time에서 역추정하지 않는다.
