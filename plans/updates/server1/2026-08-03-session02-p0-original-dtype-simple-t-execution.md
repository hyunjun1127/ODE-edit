# Session 02 original-dtype simple-T P0 실행 업데이트

상태: `P0_TECHNICAL_BLOCK / ZERO_SCIENTIFIC_OUTCOME / RETRY_AND_P1_HOLD`.

## 실행 경계

- instruction: `ODEEDIT-S02-P0-ORIGINAL-DTYPE-SIMPLE-T-PAIR-V1`
- execution branch: `codex/odeeditsh1-p0-orig-v1`
- execution head: `dfafe486cb5307c3afe5cd8c1ce8b5fe6ed33def`
- proposal ID: `c4176176fe54132466d0d75397d647564be8f152396e4c6c1fde63ed09326023`
- lock SHA-256: `6fe38820652cecd2bde8bbe2fbf47ac65943d776e915af56a65dc07840381be8`
- submission gate: tracked clean, fixed artifact/hash/request/cache PASS,
  output roots absent, project GPU `0 + 2 <= 4`, total host-memory request
  `130000 <= 396234 MiB`.

두 job은 2026-08-03 19:07:55 KST에 같은 batch에서 연속 제출되어 동시에
시작했다.

| Model | Job | Terminal | Elapsed | Failure boundary |
| --- | --- | --- | --- | --- |
| Llama3-8B-Instruct | `16061` / `odeedit_s02_p0_orig_llama` | `FAILED 1:0` | 41 s | action 전 fresh-context manifest gate |
| Qwen2.5-7B-Instruct | `16062` / `odeedit_s02_p0_orig_qwen` | `FAILED 1:0` | 63 s | action 전 fresh-context manifest gate |

두 실행 모두 `prepare_concrete_environment`에서
`fresh context manifest differs before action`으로 fail-close했다. 이 지점에
도달하기 전에 common original-dtype loader의 policy, checkpoint dtype,
observed floating-parameter dtype 및 config dtype 검사가 통과했지만, manifest
write는 context preflight 뒤에 있으므로 실제 field를 담은 `manifest.json`은
생성되지 않았다.

## 산출 상태

- 각 root에는 0-byte `controller_steps.jsonl`, `compute.jsonl`,
  `evaluation.jsonl`만 있다.
- first arm, direct-z, A/B, T/C, write, evaluation은 시작되지 않았다.
- `summary.json`, `manifest.json`, `terminal_manifest.json`, `direct_z`는 없다.
- Full/Native ratio, GPU peak memory, component timing은 unavailable이며 0으로
  해석하지 않는다.
- retry/resubmit/source/lock/tolerance 변경은 수행하지 않았다.

## 다음 상태

Exact context mismatch RCA와 새 GH envelope 전까지 retry와 P1을 HOLD한다.
두 모델에서 같은 gate가 실패했다는 사실은 common provenance incompatibility
가능성을 높이지만, 원인은 현재 artifact만으로 확정하지 않는다.

Raw broadcast는 server2가 `assigned-onboarding-hold`이고 session boundary,
method runtime, rsync dry verification이 닫히지 않아 수행하지 않았다:
`BROADCAST_EXCEPTION_SERVER2_NOT_READY`. 분석 상태는
`ANALYSIS_PENDING/TERRA_RUNTIME_MISMATCH`이다.
