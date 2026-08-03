# Session 02 checkpoint-original BF16 / simple-T P0 승인

- 작성: **2026-08-03 KST**
- GH session: `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`
- 대상 SH1 session: `019fc63e-5217-7250-9c22-c5b2ec4248f0`
- 판정: **`APPROVED_FOR_PAIRED_TECHNICAL_P0_ONLY`**
- 통합 source head: `d784c7e`
- numerical-lock proposal ID: `c4176176fe54132466d0d75397d647564be8f152396e4c6c1fde63ed09326023`
- lock SHA-256: `6fe38820652cecd2bde8bbe2fbf47ac65943d776e915af56a65dc07840381be8`

## 결론

Llama3-8B-Instruct와 Qwen2.5-7B-Instruct를 각각 pinned checkpoint의 원본 BF16
dtype으로 로드하는 paired technical P0를 승인한다. 이는 scientific 우열 실험이 아니라
loader, derivative, finite-trial/commit identity, memory와 compute feasibility를 확인하는
기술 실행이다. Evaluation과 generation은 수행하지 않는다.

## 네 범주 구분

### 사용자에게서 온 내용

- 모든 method experiment는 변환된 precision이 아니라 원본 모델로 수행한다.
- Llama와 Qwen은 같은 method, loader policy, controller와 gate를 사용한다.
- 두 모델 job은 순차 제출하지 않고 같은 batch로 제출한다.

### Repo와 pinned local snapshot에서 확인한 사실

- Llama revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`는 config가
  `bfloat16`이고 safetensors tensor `291/291`이 BF16이다.
- Qwen revision `a09a35458c702b33eeacc393d103063234e8bc28`는 config가
  `bfloat16`이고 safetensors tensor `339/339`이 BF16이다.
- 이전 failed P0는 legacy loader가 FP32를 강제한 실행이므로 원본-model evidence가 아니다.
- 새 method runner는 `load_fixed_model_checkpoint_original`만 호출하고, floating parameter
  dtype set과 config dtype이 모두 BF16인지 실제 load 뒤 fail-close로 검증한다.
- Adaptive finite candidate는 두 모델 모두 simple `T`, 즉
  `QuantizedRowBlockFunctionalTrial(row_block=64)`를 사용한다.
- CPU/mock 상태에서 A/B `24/24`, T/C output/event `36/36`, Method `62/62`, loader
  `6/6`이 통과했다. Scientific outcome은 0건이다.

### GH 추정

- Original BF16 residency와 transient dense solve가 이전 Qwen FP32 OOM 위험을 줄일 가능성은
  있으나, 실제 model-scale memory가 측정되기 전에는 해결됐다고 단정하지 않는다.
- Simple-T는 BF16 accepted write와 같은 유한 update를 판정하는 가장 직접적인 구현이지만,
  모든 rejected candidate에도 비용이 들어가므로 Full/Native compute gate가 중요하다.

### 사용자 확인 필요

- 현재 없음. 이 문서는 사용자가 확정한 original-model policy를 실행 lock으로 구체화한다.

## 기존 canonical spec 보정

이 승인과
`plans/global/2026-08-03-session02-original-checkpoint-dtype-addendum.md`는
`plans/global/2026-08-03-session02-compute-aware-main-table-spec.md`의 continuous finite-trial
설명과 P0 backend 선택 부분을 다음 범위에서 보정한다.

- Continuous FP32 overlay `B`는 derivative/reference 전용이다.
- 원본 BF16 model의 finite adaptive trial은 quantized row-block emulator `T`가 담당한다.
- Accepted write `C`와의 identity는 T/C로 판정한다.
- `B -> T` two-tier와 cached trial graph는 이번 P0에서 금지한다.

이 보정은 direct-z, event, QP, trust constants, case/order, step cap 또는 accepted-write
coefficient를 바꾸지 않는다.

## 승인된 P0 범위

| 항목 | 승인값 |
|---|---|
| Models | `llama3-8b-inst`, `qwen2.5-7b-inst` |
| Case | canonical prefix `2022` |
| Arms | Native, Static, One-refresh, Full |
| Dtype | checkpoint-original `torch.bfloat16` |
| Trial | simple-T, row block 64 |
| Repetitions | warm-up 1 + recorded 3 |
| Evaluation/generation | 금지 |
| GPU | 1/model, paired aggregate 2 |
| CPU | 8/job |
| Host memory | 65,000 MiB/job |
| Wall time | 04:00:00/job |
| Project cap | server1 4 GPU |

새 output root는 다음 두 경로만 허용한다.

- `local/results/session02-p0-original-dtype-simple-t-v1-llama3-8b-inst-c4176176`
- `local/results/session02-p0-original-dtype-simple-t-v1-qwen2.5-7b-inst-c4176176`

Log는 tracked sbatch template의 `local/logs/session02-p0-%x-%j.{out,err}` 아래에서 두
승인 job name에 대응하는 파일만 허용한다. 기존 failed P0 root/log는 읽기 보존하며 덮어쓰기,
삭제 또는 retry root 재사용을 금지한다.

## 실행 전 red gate

1. SH1 dedicated worktree가 이 승인 commit을 exact HEAD로 사용하고 tracked clean이어야 한다.
2. session/role/repository boundary가 PASS여야 한다.
3. lock/proposal ID와 source hashes가 위 승인값과 일치해야 한다.
4. 두 output root가 모두 없어야 한다.
5. submit 직전 ODE-Edit running+pending GPU에 새 2 GPU를 더해 cap 4 이하여야 한다.
6. 두 job을 한 submission batch로 제출해야 하며 단일-model 선행 실행은 금지한다.
7. `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, EasyEdit read-only와 기존 precomputed
   artifact read-only 재사용을 유지한다.

## 즉시 중단 조건

- 실제 loaded parameter/config dtype이 BF16이 아님
- model revision, request/context, hparams/covariance hash 불일치
- A/B scalar-reference gate 또는 T/C post-commit identity gate 실패
- coefficient-zero identity 실패, target mutation, rollback 또는 terminal integrity 실패
- CUDA OOM, cap 초과, unexpected network/download/recompute/EasyEdit write
- 한 모델에만 적용되는 dtype, trial, tolerance 또는 controller rescue 필요

한 job의 기술 실패가 sibling을 자동 취소하는 조건은 아니다. 이미 시작된 sibling은 data-loss나
cap 위험이 없는 한 자체 terminal evidence까지 진행하되 retry는 별도 GH envelope 없이는 금지한다.

## P0 이후 판정

- 이 P0만으로 method superiority나 Motivation claim을 승격하지 않는다.
- 두 모델 모두 terminal technical PASS이고 `Full/Native <= 3x`면 compute green이다.
- `(3x, 4x]`는 yellow, `>4x`는 P1/main expansion HOLD다.
- Terra Ultra analysis runtime을 검증하지 못하면 결과 수치 보존은 하되 scientific 해석과
  claim promotion은 HOLD한다.
