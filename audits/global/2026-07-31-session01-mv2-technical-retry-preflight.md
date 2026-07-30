# Session 01 Motivation — MV-2 technical retry preflight

## 범위와 판정 단위

- proposal에서 온 내용:
  partial update 뒤 stale direction/coefficient와 refreshed
  direction/coefficient를 exact equal-`C`로 분리한다.
- repo/protocol에서 확인한 사실:
  최초 paired job `15597`은 두 child가 같은 시각에 시작했으나 Llama
  first case가 `RuntimeError`로 끝났고 fail-fast가 Qwen을 종료했다.
  이 audit은 scientific result가 아니라 그 technical failure의 단 한 번
  재시도만 다룬다.
- GH 추정:
  first case에서 GPU reserved memory가 높았고 synchronous proposal path에
  inference guard가 없었던 점이 1순위 원인이다. 기존 artifact에는 stack
  location이 없어 확정 원인이라고 주장하지 않는다.
- 사용자 확인 필요:
  없음. 사용자는 필수 감사만 수행하고 Llama/Qwen을 순차가 아니라 동시에
  빠르게 제출하라고 명시했다.

## 최초 시도 증거

- parent: `15597`, `FAILED`, exit `1:0`,
  `2026-07-31 06:50:55–06:56:10 KST`
- child `.0`: Llama, `FAILED`, exit `1:0`, elapsed `00:05:12`
- child `.1`: Qwen, `CANCELLED`, exit `0:9`, fail-fast sibling termination
- 두 child start: exact `2026-07-31 06:50:57 KST`
- peak RSS:
  Llama `9,388,964K`, Qwen `15,350,368K`; Slurm host-memory OOM 아님
- Llama local summary:
  first case attempted, `RuntimeError`, feature/action/outcome/receipt `0`,
  remaining 11 case not run
- Qwen:
  scientific stream을 만들기 전 sibling failure로 종료

따라서 partial stream을 scientific null 또는 unfavorable result로 분석하지
않는다.

## 최소 수정

1. `EasyEditBridge.propose_synchronous_memit_factors()`의 upstream
   representation/solver 구간을 `torch.no_grad()`로 감싼다.
   proposal tensor 값, fixed target, covariance, solver, case, arm, gate는
   바꾸지 않는다.
2. `run_mv2_event_loop()`의 ignored local stderr에
   exception type과 `file/line/function` stack location만 남긴다.
   exception message, source line, locals, request/target text는 기록하지 않는다.
3. 기존 failed raw는 삭제하지 않고 ignored
   `local/results/raw/session01_motivation/failed/mv2refresh_pair_v1_job15597/`
   아래로 이동해 보존한다.
4. logical run ID와 exact locked panel은 그대로 유지하고, 별도 retry marker로
   재제출을 1회만 허용한다.

## 불변 scientific contract

- model:
  `llama3-8b-inst`, `qwen2.5-7b-inst`
- same parent allocation, child별 exactly one GPU, 동시 시작
- salted rank `[100:112]`, exact 12 case/model, first100과 disjoint
- layers `4–8`, `q=1/256`, `h=1/2`, run seed `17`
- six-arm order, direct-z one-compute lineage, matched second-`C`
- primary:
  `A-B` direction refresh
- conditional:
  `B-C` coefficient refresh
- secondary expected-effect proxy:
  `A-C` total refresh
- first-match pair gate 및 practical floor `1e-4`

Threshold, model, case, arm, bootstrap, outcome field를 technical failure에
맞춰 바꾸지 않는다.

## 필수 실행 전 검사

- [x] focused synchronous/MV-2/trajectory/lineage tests `24/24`
- [x] 전체 Motivation unit tests `165/165`
- [x] sensitive-string non-persistence test 통과
- [x] `py_compile`, `bash -n`, parent `sbatch --test-only` 통과
- [x] original job marker 보존, failed raw recoverable archive 완료
- [x] canonical output path 및 retry marker 부재
- [x] session/repository/main 및 commit 직전 clean-boundary 검사 통과
- [x] server1 cap `2 GPU / 130000M` 허용
- [x] 변경 파일 independent red review `PASS`, residual P1/P2 없음

## 중단 조건

- scientific contract 또는 selected cases가 최초 lock과 달라짐
- trace가 request/target/credential/source line/locals를 persist함
- failed raw를 덮어쓰거나 삭제함
- retry marker/output이 이미 존재함
- dirty/unpushed Git, session mismatch, cap 초과, duplicate active job
- child 동시 시작 실패
- 어느 child든 nonzero이면 sibling fail-fast 후 추가 retry 자동 금지

## 허용 범위

이 audit은 exact helper
`project/run_scripts/submit_session01_mv2refresh_pair_server1.sh`의 technical
retry 1회만 허용한다. MV-3, threshold 변경, 추가 fold, outcome 기반 rescue,
scientific claim은 허용하지 않는다.

Commit·push 뒤 helper가 다시 `clean main == origin/main`, exact verdict,
canonical output 부재, original/retry marker, active duplicate, session 및
resource cap을 fail-closed로 재검사한다.

- 최종 판정: `PASS` — locked MV-2 pair의 technical retry 1회에만 유효
