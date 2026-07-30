# Session 01 Motivation — MV-2 CUDA hash RCA diagnostic preflight

## 필요성

- repo/protocol에서 확인한 사실:
  jobs `15597`, `15600`, `15603`은 모두 exact simultaneous start 뒤 first
  Llama case의 scientific commitment 전에 fail-closed했다.
- job `15603`은 small synthetic CUDA/CPU hash parity를 통과했지만 actual
  proposal hash의 `tensor_sha256()` sync/copy line에서 `RuntimeError`가
  재발했다.
- GH 추정:
  CUDA async failure가 첫 host synchronization에서 뒤늦게 표면화됐거나,
  actual factor의 size/layout에서만 sync/copy failure가 발생한다.
- 사용자 확인 필요:
  없음. scientific pair를 반복 낭비하지 않는 최소 1-GPU technical RCA다.

## diagnostic contract

- model: Llama first fixed model only
- GPU: exactly `1`, CPU `8`, memory `65000M`, walltime `00:12:00`
- case: locked stream의 first case에서 failure boundary만 관찰
- `CUDA_LAUNCH_BLOCKING=1`로 asynchronous CUDA failure를 originating
  operation에 귀속
- tensor-hash failure는 raw exception message 대신 fixed
  `phase/category/dtype/shape/device/numel`만 local stderr에 기록
- output은 scientific analysis 금지, 종료 후 ignored failed archive로 이동
- EasyEdit/model/cache/dataset/stats는 read-only
- exact diagnostic env 조합에서 event loop, placeholder, planned count,
  case-results를 first request 하나로 제한하고 `all_pass=false`,
  `run_status=technical_diagnostic`을 강제한다.

이 diagnostic은 사용자의 Llama/Qwen 동시 scientific 실행 지시를 대체하지
않는다. final MV-2는 두 model을 다시 same parent에서 동시에 실행해야 한다.

## safe categories

- `cuda_out_of_memory`
- `cuda_illegal_memory_access`
- `cuda_misaligned_address`
- `cuda_device_assert`
- `cuda_cublas`
- `cuda_cusolver`
- `cuda_runtime`
- `host_out_of_memory`
- `tensor_layout`
- `unknown_runtime`

Exception message/source line/locals/request/target text는 저장하지 않는다.

## 필수 검사

- [x] category/hash/MV-2 focused tests `18/18`
- [x] 전체 Motivation tests `170/170`
- [x] `py_compile`, `bash -n`, diagnostic `sbatch --test-only` 통과
- [x] prior raw archive와 retry2 marker 보존
- [x] canonical output/diagnostic marker 부재
- [x] session/server1 `1 GPU / 65000M` cap 및 commit 직전 boundary 통과
- [x] independent red 재검토 `PASS`, 최초 P1 closure, residual P1/P2 없음

## 중단 조건

- diagnostic이 Qwen을 실행하거나 1 GPU/12분 cap을 넘음
- scientific parameter를 바꾸거나 partial output을 분석함
- exception message/request/target/locals가 persist됨
- dirty/unpushed Git, marker/output 중복, active duplicate, session mismatch
- source/cache/EasyEdit write 또는 online access

## 실행

exact helper:
`project/run_scripts/submit_session01_mv2refresh_hashdiag_server1.sh`

이 audit은 위 technical RCA 1회만 허용한다. 결과에 따라 exact root cause를
고친 뒤 새 red gate 없이 final pair를 자동 재제출할 수 없다.

Commit·push 뒤 helper가 clean `main == origin/main`, tracked HEAD, exact
verdict, retry2/diagnostic marker, canonical output, active duplicate,
session/resource cap을 다시 fail-closed로 검사한다.

- 최종 판정: `PASS` — Llama first-case CUDA hash RCA diagnostic 1회에만 유효
