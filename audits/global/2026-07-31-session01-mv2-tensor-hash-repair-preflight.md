# Session 01 Motivation — MV-2 tensor-hash repair preflight

## 범위

- proposal에서 온 내용:
  동일 locked MV-2 stale/refreshed diagnostic을 기술적으로 완주한다.
- repo/protocol에서 확인한 사실:
  instrumented retry job `15600`은 first Llama case의 scientific commitment
  전에 다시 fail-closed했고 safe stack이 exact plumbing line을 특정했다.
- GH 추정:
  없음. safe stack으로 failure boundary가 확인됐다.
- 사용자 확인 필요:
  없음. 사용자의 빠른 Llama/Qwen 동시 실행 지시 안에서 technical plumbing만
  복구한다.

## 확정된 failure boundary

Job `15600`의 single safe stack:

```text
mv2_refresh.py:1345 run_mv2_event_loop
mv2_refresh.py:741 _run_mv2_event
frozen_target_lineage.py:113 proposal_direction_hash
frozen_target_lineage.py:99 proposal_direction_payload
hooks.py:34 tensor_sha256
```

- exception type: `RuntimeError`
- Llama feature/action/outcome/receipt: `0`
- Qwen: sibling fail-fast, scientific summary 없음
- Slurm OOM state/marker: 없음
- 두 child: exact simultaneous start, resource envelope 준수

따라서 이는 ODE-Edit mechanism의 null이 아니라 CUDA tensor byte-hashing
plumbing failure다.

## 최소 수정

1. `tensor_sha256()`의 canonical logical-byte 순서를 유지하면서
   `detach -> CPU -> contiguous -> uint8 view` 순서로 바꾼다.
2. source tensor dtype/value, autograd, model state는 변경하지 않는다.
3. CUDA device에서 byte reinterpret/contiguous 임시 allocation을 만들지
   않는다.
4. fixed float16/bfloat16/float32/float64와 contiguous/transposed layout의
   `hash(cuda) == hash(cpu canonical)` preflight를 model load 전에 실행한다.
   probe는 작은 synthetic tensor만 사용하며 request/model data를 포함하지
   않는다.
5. `hooks.py`와 regression test를 child/helper tracked-HEAD allowlist에
   명시한다.
6. job `15600` raw는 삭제하지 않고 ignored
   `local/results/raw/session01_motivation/failed/mv2refresh_pair_v1_job15600/`
   아래에 보존한다.

## scientific lock 불변

- models:
  `llama3-8b-inst`, `qwen2.5-7b-inst`
- one simultaneous parent, child별 one GPU
- exact salted rank `[100:112]`, 12 case/model
- layers `4–8`, `q=1/256`, `h=1/2`, seed `17`
- direct-z one-compute, frozen lineage, equal second-`C`, six-arm order
- direction `A-B` primary, coefficient `B-C` conditional,
  total `A-C` secondary forecast
- bootstrap/gate/practical floor/first-match pair rule 불변

## 필수 검사

- [x] tensor-hash/lineage/trajectory/MV-2 focused tests `27/27`
- [x] 전체 Motivation tests `167/167`
- [x] `py_compile`, `bash -n`, parent `sbatch --test-only` 통과
- [x] safe trace non-leakage 유지
- [x] job `15600` raw recoverable archive 완료
- [x] original/retry1 marker 보존, repair marker/output 부재
- [x] session/cap/repository 및 commit 직전 clean-boundary 검사 통과
- [x] independent red review `PASS`, residual P1/P2 없음

## 중단 조건

- CUDA hash parity preflight 실패
- hash 결과가 CPU canonical bytes와 다름
- model/case/arm/q/h/seed/gate 변경
- failed raw 삭제/덮어쓰기
- dirty/unpushed Git, marker/output 중복, active duplicate, cap/session mismatch
- child 비동시 시작 또는 어느 child든 nonzero

이 gate는 exact locked pair의 tensor-hash repair execution 1회만 허용한다.
추가 자동 retry, retuning, partial-result rescue, MV-3 또는 scientific claim은
허용하지 않는다.

Commit·push 뒤 helper가 tracked HEAD, clean `main == origin/main`, 네 exact
gate, marker/output, active duplicate, session 및 resource cap을 다시
fail-closed로 검사한다. 실제 CUDA parity는 각 child에서 model load 전에
실행되며 실패 시 scientific stream 없이 중단된다.

- 최종 판정: `PASS` — locked MV-2 pair의 tensor-hash repair execution 1회에만 유효
