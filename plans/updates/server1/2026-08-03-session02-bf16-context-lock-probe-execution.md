# Session 02 BF16 context-lock paired calibration 실행

상태: `PAIR_PASS / OUTCOME_FREE_NEW_LOCK_CANDIDATES / P0_RETRY_NOT_AUTHORIZED`.

## 실행 경계

- instruction: `ODEEDIT-S02-BF16-CONTEXT-LOCK-PROBE-PAIR-V1`
- branch: `codex/odeeditsh1-bf16-context-probe-run-v1`
- execution head: `37a713233b95614d2743a12abc4d1036daed95f0`
- probe SHA-256: `00f839555fd76289696abcbaee4d108d392f56d58c4903f945e9c71f0e04583c`
- sbatch SHA-256: `57f18be146c40b26b9b9d98598dea31190609ca313b66fb778512c0cebfc27ec`
- pre-submit project GPU: active/pending 0 + requested 2 = 2, cap 4 이하

두 job은 2026-08-03 19:23:36 KST에 동시에 시작했다.

| Model | Job | State | Elapsed | Exact repeat |
| --- | --- | --- | --- | --- |
| Llama3-8B-Instruct | `16063` / `odeedit_s02_ctx_llama` | `COMPLETED 0:0` | 14 s | PASS |
| Qwen2.5-7B-Instruct | `16064` / `odeedit_s02_ctx_qwen` | `COMPLETED 0:0` | 13 s | PASS |

## Outcome-free calibration

| Model | Legacy-FP32 lock ID | Observed original-BF16 ID | Template-byte SHA-256 | Groups |
| --- | --- | --- | --- | --- |
| Llama | `3020b3f5cea62e6cfbd173f0c99a4348cecf7e84425bb087720ced7f395482e5` | `22c26dc11fb13acd51d5bdc483e4b9dd46fa40029fe10b167bff1a1642f7e686` | `0a2069beafc60e170251103028fde716a160a60a8048bb000a649cf26c233bb0` | `[1,5]` |
| Qwen | `e0c5f61d874334a2cab26f82fb3594d9ab88c9e5816be390274a0bb5e98c93bd` | `5b7144416638fb3deec1f12f204a401e06edd1293aa2fe4a22f9080f3e8bd41b` | `ff84e360b7a4a413275da818862c7c4431ac21c7fbeaf26e4d2296278f9a4bf9` | `[1,5]` |

각 모델의 repeat 1/2는 source, template bytes, manifest ID와 group sizes가
exact 동일했다. 두 observed ID는 legacy-FP32 lock ID와 다르다. 이는 현재
pinned source/revision/seed 아래 original-BF16 fresh template identity가 기존
lock과 다르다는 provenance 사실이며 method outcome이나 우월성 증거가 아니다.

모든 runtime dtype field는 `torch.bfloat16`, policy는
`checkpoint-original`이었다. Scope counters는 context generation 2회 외
edit/direct-z/covariance/dataset/evaluation 0이며 scientific outcome도 0이다.

## 보존과 다음 단계

Raw template는 각 local `context_manifest.json`에만 보존했다. Programmatic
검사에서 stdout, stderr, summary, terminal manifest에 raw generated string이
없음을 확인했다. Server2는 onboarding HOLD이므로 rsync하지 않았다:
`BROADCAST_EXCEPTION_SERVER2_NOT_READY`.

Observed IDs는 outcome-free new-lock 후보로만 GH에 전달한다. Numerical lock
수정과 P0 retry는 이 envelope에서 수행하지 않았으며 별도 GH 승인이 필요하다.
