# BGODE-R3 G2 pre-GPU 사실 보고서

## 판정

`R3_G2_PRE_GPU_PASS`이다. G1 natural unequal-non-prefix는 Llama/Qwen 2/2 terminal PASS이며, G2는 동일 ordinal 26/case 17454만 사용한다. 실행 전 model/GPU/Slurm action은 0/0/0이다.

## 고정 matrix

- array: 0=Llama, 1=Qwen, `0-1%2`, 1 GPU/task, project cap3
- 각 모델: Plain/Fisher/Full × N={4,8,16,32} = 12 panel
- W0, fixed-z, T_AE, normalized actuator, equality `[1,0]`이 arm/N 사이 동일하다.
- 모든 node에서 ordered dictionary 재구축, 전체 internal-prefix serial JVP+central FD, factor-space residual과 actual dense W delta를 기록한다.
- history append0, rho/root/localizer/ridge/damping/floor/fallback0, evaluator/controller influence0이다.

## 결과 전 봉인한 Cauchy gate

- pair: D4,8 / D8,16 / D16,32
- actual dense block W endpoint 거리, target-logit gap, q0-conditional KL gap 모두 successive contraction 필요
- physical first-order ratio floor: 1.25
- final relative physical distance ceiling: 0.05
- final normalized target-logit/q-KL gap ceiling: 0.05/0.05
- rounding bound: `256*eps_FP32*(1+observed scalar scale)`
- 기준 실패 시 tolerance를 완화하지 않고 `R3_G2_NUMERICAL_CONVERGENCE_HOLD`로 보존한다.

## gate와 identity

- focused tests 41/41; py_compile/bash/session PASS
- Llama/Qwen EasyEdit/HF/tokenization preflight PASS
- source HEAD/tree: `16616d872def3b2dcceafa2fd181187f957e3ad9` / `3533c36befbf48dbef876167a3fbbc771d06586c`
- source members root: `cfafbd0ca3b9b23bfd4a804570872e120c0ebc64eab402681366c544394791d1`
- numerical lock identity: `db90b5c74eb45d198a01c9afb8487311a744969624b124048fdc03b9d4a037cf`
- G1 terminal receipt identity: `e8449e73922d83a1d2233976a97c750e4a89889cd29d895b3dd70d3356a1fbfe`

Future fixed-T curve T={0.5,1,2,3,5}, N=32는 predeclare만 했고 이번 G2에서 실행하지 않는다. scientific promotion=false이다.
