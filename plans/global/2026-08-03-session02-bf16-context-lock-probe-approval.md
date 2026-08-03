# Session 02 original-BF16 context-lock probe 승인

- 작성: **2026-08-03 KST**
- GH session: `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`
- SH1 session: `019fc63e-5217-7250-9c22-c5b2ec4248f0`
- source head: `e26d5e9`
- 판정: **`APPROVED_FOR_PAIRED_CONTEXT_CALIBRATION_ONLY`**

## 목적

원본 BF16 P0는 두 모델 모두 edit/direct-z 전 `fresh context manifest differs before
action`으로 종료됐다. EasyEdit context는 loaded model의 top-k sampling으로 생성되므로,
legacy-FP32 lineage의 기존 lock ID가 original-BF16 generation과 다를 가능성을 outcome-free
probe로 확인한다.

## 승인 근거

- probe source SHA-256:
  `00f839555fd76289696abcbaee4d108d392f56d58c4903f945e9c71f0e04583c`
- sbatch SHA-256:
  `57f18be146c40b26b9b9d98598dea31190609ca313b66fb778512c0cebfc27ec`
- GH 재검증: focused `8/8`, full Method `70/70`, compile/shell/help/diff gate PASS.
- original loader만 사용하며 dataset, covariance, edit, direct-z, controller, evaluation에
  접근하지 않는다.
- 같은 process에서 각 repeat 직전에 seed 17을 재설정하고 `fresh=True` generation을 정확히
  두 번 수행한다.
- 두 context의 source, nested strings, canonical bytes, manifest ID와 group sizes가 exact
  동일할 때만 PASS한다.
- raw context strings는 `local/`의 `context_manifest.json`에만 저장하고 log, Git, report,
  summary에는 기록하지 않는다.

## 승인된 실행

- Models: `llama3-8b-inst`, `qwen2.5-7b-inst`
- 같은 submission batch의 2 jobs
- each: 1 GPU, 8 CPU, 65,000 MiB host memory, 00:30:00
- aggregate: 2 GPU, server1 project cap 4 이하
- evaluation/edit/direct-z/scientific outcome: 0
- retry: 금지

허용 output roots:

- `local/results/session02-bf16-context-lock-probe-v1-llama3-8b-inst-e26d5e9`
- `local/results/session02-bf16-context-lock-probe-v1-qwen2.5-7b-inst-e26d5e9`

허용 log prefix:

- `local/logs/session02-bf16-context-probe-odeedit_s02_ctx_llama-*`
- `local/logs/session02-bf16-context-probe-odeedit_s02_ctx_qwen-*`

## Red gate와 중단 조건

1. 이 승인 commit의 exact HEAD, clean tracked worktree, canonical SH1 boundary.
2. source/sbatch hash와 두 output root absent.
3. submit 직전 project running+pending+new2가 cap 4 이하.
4. checkpoint-original parameter/config dtype BF16과 pinned model/tokenizer revision.
5. EasyEdit pinned source before/after identity와 offline/no-download contract.
6. repeat mismatch면 `NONDETERMINISTIC_CONTEXT_HOLD`, exit 4; sampling/seed를 바꾸지 않는다.
7. 한 모델만 다른 generation policy가 필요하면 common gate fail.

## 결과 사용 경계

- 두 repeat가 exact PASS한 모델의 manifest ID와 group sizes만 새 numerical-lock 후보로 쓸 수
  있다. Raw templates는 Git에 올리지 않는다.
- legacy lock과 BF16 ID가 다르면 그것은 dtype별 context-generation provenance 차이의 관측
  사실이다. Method 성능이나 ODE claim은 아니다.
- 두 모델 모두 PASS하기 전에는 P0 retry와 P1을 열지 않는다.
- Terra Ultra runtime mismatch는 이 순수 technical calibration의 hash 판정을 막지 않지만,
  scientific interpretation에는 계속 적용한다.
