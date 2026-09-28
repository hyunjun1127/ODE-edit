# GH → SH3: BLUE의 기존 MEMIT_seq 구현 직접 사용

동일 task: `ODEEDIT-GH-SH3-MEMIT-HISTORY-FIXED10K-20260928-R1`.
추가 nonce: `ODEEDIT-GH-SH3-MEMIT-HISTORY-USE-BLUE-20260928-R1`.
사용자: “blue 에 구현된거 이용하라고 전달해”.
Target session `01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3`, server3,
CWD `/data/janghj/ODE-edit`, repo `hyunjun1127/ODE-edit`.

앞선 실행 envelope의 권한/쓰기 경계는 그대로이며 구현 선택을 다음으로 명확히 한다.

- `xpq-tech/BLUE`의 고정 commit
  `311b076a92e4ed0f14f5c8b4909732da781bc5f7`,
  **`memit/memit_seq_main.py::apply_memit_seq_to_model`**을 실제 import/call한다.
- 단순 참고 후 별도 history writer를 재구현하지 않는다. 기존 BLUE MEMIT_seq의
  history 전달·solve·post-key 누적 코드를 그대로 사용한다. 구현 중인 별도
  writer가 있으면 새 제출에 연결하지 말고 보존한 뒤 기존 구현으로 연결한다.
- **BLUE 저장소 사용과 `hparams.blue=true`는 다르다.** 이번 요청은 history
  baseline이므로 `MEMIT_seq`, `blue=false`, layers4–8을 유지한다.
- SH3는 독립 runner/경로 binding/상태 연속성/observer/noCP/Slurm adapter만
  구현한다. 원 native 함수의 수식/예산/누적 시점은 수정하지 않는다.
- 실제 import 경로·고정 Git commit·파일 SHA·entrypoint와 returned cache_c
  연결을 실행 lock 및 첫 ACK/M0에 명시한다. 불가피한 호환성 오류는 원문과
  최소 adapter diff를 보고하되 조용히 다른 MEMIT 구현으로 대체하지 않는다.
- 앞선 fixed10k 동일 순서 B100×100/단일 cold W0,H0 chain/cap1/noCP/
  release 후 monitoring0/추가 arm0/원자료KEEP/own-scope 게시 권한 유지.

이는 같은 active task의 구체화다. 새 task·추가 chain·별도 성능 gate를 만들지
않으며 GH 재승인을 기다리지 않는다. 같은 stream에서 nonce 수신 ACK를 회신한다.
