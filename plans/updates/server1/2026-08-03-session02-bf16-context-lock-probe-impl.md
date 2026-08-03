# Session 02 BF16 context-lock probe 구현

상태: `CPU_IMPLEMENTATION_PASS / OBSERVED_CONTEXT_ID_NONE / GPU_SLURM_HOLD`.

## 목적과 경계

`ODEEDIT-S02-BF16-CONTEXT-LOCK-PROBE-IMPL-V1`은 original-checkpoint BF16
모델에서 EasyEdit fresh context sampling을 같은 process에서 정확히 두 번
재현하는 outcome-free probe다. 기존 legacy-FP32 lock과 BF16 sampling의 차이는
아직 추정이며, 이 구현 단계에서는 실제 모델이나 context ID를 관측하지 않았다.

Probe는 다음만 수행한다.

1. fixed offline revision/tokenizer와 checkpoint-original BF16 dtype을 검증한다.
2. pinned EasyEdit source 13개를 preflight한다.
3. 각 repeat 직전에 `seed_runtime(17)`을 호출한다.
4. 같은 source와 `fresh=True`로 context를 정확히 두 번 생성한다.
5. source, nested template 문자열, canonical template bytes, manifest ID,
   group sizes를 exact 비교한다.

Dataset, covariance, edit request, direct-z, controller, evaluation에는 접근하지
않는다. Source에는 Slurm submission 기능이 없고 sbatch template은 별도 future
token 없이는 fail-close한다.

## 산출물 계약

Local output root는 absent/create-once이며 다음 세 파일만 생성한다.

- `context_manifest.json`: 두 repeat의 raw template를 포함하며 local-only다.
- `summary.json`: model/revision/dtype, seed, repeat IDs, group sizes,
  template-byte hashes, exact-match와 no-edit scope만 포함한다.
- `terminal_manifest.json`: 앞 두 파일의 SHA-256/size와 terminal verdict다.

Mismatch도 두 repeat evidence를 보존한 뒤
`NONDETERMINISTIC_CONTEXT_HOLD`, exit 4로 종료한다. Raw template는 stdout,
summary, terminal manifest, report 또는 Git에 복제하지 않는다.

## CPU gate

- focused probe tests: `8/8 PASS`, warnings as errors.
- full method regression: `70/70 PASS`, warnings as errors.
- deterministic repeat, mismatch fail-close, raw-free summary/terminal,
  collision/offline fail-close, BF16 dtype/config failure, both-alias schema,
  legacy-loader AST absence와 no-submit resource lock을 포함한다.
- `py_compile`, `bash -n`, CLI help, `git diff --check`: PASS.

파일 SHA-256:

- probe: `00f839555fd76289696abcbaee4d108d392f56d58c4903f945e9c71f0e04583c`
- sbatch: `57f18be146c40b26b9b9d98598dea31190609ca313b66fb778512c0cebfc27ec`
- tests: `ef1dc06a489cc198888d2d935ee5a68e85bda83f316a80b14ff09fe35416d9af`

## Future paired calibration shape

- one model/job, 1 GPU, 8 CPU, 65000 MiB, `00:30:00`
- pair aggregate: 2 GPUs, server1 cap 4 이하
- proposed roots:
  - `local/results/session02-bf16-context-lock-probe-v1-llama3-8b-inst-26aa173`
  - `local/results/session02-bf16-context-lock-probe-v1-qwen2.5-7b-inst-26aa173`

Future envelope가 승인할 때만 사용할 command shape:

```text
sbatch --job-name=odeedit_s02_ctx_llama --export=ALL,ODEEDIT_CONTEXT_PROBE_SUBMISSION_AUTHORIZED=bf16-context-lock-probe-v1,MODEL_ALIAS=llama3-8b-inst,OUTPUT_ROOT=local/results/session02-bf16-context-lock-probe-v1-llama3-8b-inst-26aa173 project/run_scripts/session02_bf16_context_lock_probe.sbatch
sbatch --job-name=odeedit_s02_ctx_qwen --export=ALL,ODEEDIT_CONTEXT_PROBE_SUBMISSION_AUTHORIZED=bf16-context-lock-probe-v1,MODEL_ALIAS=qwen2.5-7b-inst,OUTPUT_ROOT=local/results/session02-bf16-context-lock-probe-v1-qwen2.5-7b-inst-26aa173 project/run_scripts/session02_bf16_context_lock_probe.sbatch
```

이번 envelope에서는 위 명령을 실행하지 않았다. 실제 observed BF16 context ID,
GPU memory/time, legacy-lock 비교는 모두 `UNOBSERVED`이며 별도 paired calibration
envelope 전까지 HOLD다.
