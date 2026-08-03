# Session 02 original-checkpoint dtype addendum

- 작성: **2026-08-03 KST**
- GH session: `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`
- 사용자 결정: **현재 및 향후 method experiment는 checkpoint 원본 dtype을 보존한다.**
- 판정: **`ORIGINAL_DTYPE_REQUIRED; LEGACY_FP32_P0_SUPERSEDED; GPU_RETRY_HOLD`**

## 결론

두 pinned checkpoint의 원본 floating tensor dtype은 모두 BF16이다. 따라서 Session 02의
`original model`은 BF16이며, 임의 FP32 cast도 임의 BF16 cast도 허용하지 않는다. Loader는
pinned checkpoint가 선언·저장한 dtype을 그대로 관측하고 검증해야 한다.

## 네 범주 구분

### 사용자에게서 온 내용

- 실험은 항상 원본 모델로 수행한다.
- 낮은 precision으로 바꾸어 memory를 줄이는 방식이나, 기존 실행과 맞추기 위해 FP32로
  올리는 방식을 사용하지 않는다.

### Repo와 local pinned snapshot에서 확인한 사실

- Llama revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`의 `config.json`은
  `torch_dtype=bfloat16`이며, 4개 safetensors shard의 tensor `291/291`이 BF16이다.
- Qwen revision `a09a35458c702b33eeacc393d103063234e8bc28`의 `config.json`은
  `torch_dtype=bfloat16`이며, 4개 safetensors shard의 tensor `339/339`이 BF16이다.
- `project/run_scripts/ode_edit_motivation/gpu_runtime.py`의 기존 Motivation loader는
  `torch_dtype=torch.float32`를 명시한다. 파일 docstring도 MV-0 EasyEdit fidelity를 위해
  float32를 사용한다고 기록한다.
- failed P0의 retained 두 manifest는 모두 runtime dtype을 `torch.float32`로 기록했다.
- 따라서 failed P0는 원본-checkpoint dtype profiler가 아니다. scientific outcome은 원래부터
  0건이며, 기술 failure provenance로만 보존한다.

### GH 추정

- Qwen OOM에는 forced FP32 model residency와 dense-system temporary allocation이 함께 기여했을
  가능성이 높다. 새 BF16 P0 전에는 각 원인의 인과 비중을 단정할 수 없다.
- Llama의 legacy-FP32 hook/FD 실패는 scalar finite-difference validator 문제를 발견했지만,
  원본 BF16에서 같은 수치가 재현된다는 증거는 아니다.
- Original BF16에서는 continuous tangent와 quantized committed write를 구분하는 A/B/T/C
  contract가 필요하다. 이는 precision rescue가 아니라 원본 모델 수치 의미를 보존하기 위한
  implementation contract다.

### 사용자 확인 필요

- 현재 없음. 사용자가 original-model policy를 명시했으므로 loader와 P0 lock을 그 기준으로
  수정한다. GPU 실행은 revised lock의 별도 GH 승인 전까지 닫혀 있다.

## Loader boundary

- 과거 Motivation의 exact reproduction을 위해 legacy float32 entry point는 명시적 이름과
  metadata로 남길 수 있다.
- Session 02 method runner는 checkpoint-original entry point만 사용한다.
- Runtime metadata는 dtype을 hardcode하지 않고 loaded floating parameters에서 관측한다.
- Pinned config, safetensors header와 loaded parameter dtype이 일치하지 않으면 fail-close한다.
- Llama/Qwen에 동일한 loader policy를 적용하며 alias별 dtype rescue를 금지한다.

## Trial boundary

- `A`: captured activation/gradient의 FP32 continuous tangent derivative
- `B`: alpha-zero scalar-gate continuous tangent reference
- `T`: finite coefficient의 quantized row-block, read-only commit emulator
- `C`: 실제 BF16 accepted write

Derivative correctness는 `A/B`, finite trial fidelity는 `T/C`로 각각 닫는다. Continuous `B`와
quantized finite `C`를 동일하다고 요구하거나 하나의 tolerance로 섞지 않는다.

다음 technical retry에서는 semantic closure를 우선하여 simple `T`를 모든 candidate trial에
사용한다. Cheap `B -> T` two-tier는 false-reject와 이중 trust verdict가 아직 정의되지 않아
허용하지 않는다. `T` 비용은 숨기지 않고 `trial` component와 `N_trial`에 전량 포함한다.

## 기존 결과의 claim boundary

- Session 01 Motivation 결과는 당시 locked legacy-float32 runtime의 결과다.
- 해당 결과를 original-BF16 evidence로 재명명하거나 main-table 수치와 직접 합치지 않는다.
- Motivation의 방향성 handoff는 provenance caveat와 함께 가설 형성 근거로만 유지한다.
- Session 02의 method correctness, compute와 main-table claim은 새 original-BF16 lineage에서만
  판정한다.

## 다음 실행 gate

1. original loader의 config/header/loaded-parameter dtype identity
2. 두 model 공통 source/schema
3. `A/B` fixed stress PASS
4. `T/C` fixed BF16 output/event identity PASS
5. Qwen transient dense solve의 동일 수식과 bounded temporary lifetime
6. original-BF16 P0의 실제 peak memory 및 Full/Native compute
7. 기존 failed roots 보존과 새 proposal ID/output root

이 gate가 CPU/dry 상태에서 닫히고 GH가 revised numerical lock을 승인하기 전에는 Slurm을
제출하지 않는다.
