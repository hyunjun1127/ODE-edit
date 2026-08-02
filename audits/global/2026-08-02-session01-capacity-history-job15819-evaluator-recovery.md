# Session 01 capacity/history job 15819 evaluator-only recovery

- 날짜: 2026-08-02
- 방법명: ODE-Edit
- 판정: **technical recovery approved; scientific action rerun 금지**

## 실패 원인

Job `15819`의 controller 8개는 모두 4/4 action receipt와 terminal `all_pass=true`로
완료됐다. 그러나 controller 시작 commit `9900a51` 이후 GH가 실행 중 direct-z
proposal 문서와 audit만 commit해 main HEAD가 `e3eb763`으로 바뀌었다. Evaluator는
모든 controller hash를 검증한 뒤, 첫 evaluation row를 decode하기 전 line 689의
whole-repository Git identity equality에서 fail-closed했다. 이 동시 Git commit은
GH 운영 오류다.

## 복구 가능성 근거

- `9900a51..e3eb763`의 변경은 direct-z proposal 문서와 audit 두 경로뿐이다.
- Capacity controller, evaluator, analyzer, QP, geometry, Alpha history/factor/adapter는
  byte-identical하다.
- Controller 8개는 모두 `completed`, `all_pass=true`, exact stream count 4,
  receipt 4, artifact hash exact, `outcome_fields_loaded=false`다.
- Failed evaluator는 output directory를 만들기 전 종료해 평가 artifact가 0개다.
- Terra Ultra RCA/red/repro agent 세 개는 runtime metadata를 검증하지 못해 파일을
  읽지 않고 종료했다. 독립 agent pass로 세지 않는다.

## Recovery lock

1. Controller와 같은 detached commit `9900a51`의 clean worktree에서 unmodified
   evaluator/analyzer를 실행한다.
2. 잠긴 evaluator의 local-path boundary를 지키기 위해 controller 8개를 detached
   worktree의 ignored `local/`에 exact-copy한다. source/staged 양쪽의 symlink 부재와
   recursive byte equality가 확인되지 않으면 시작하지 않는다.
3. Current HEAD와 `9900a51` 사이의 위 scientific paths가 identical인지 job 시작
   시 다시 검증한다.
4. Controller proposal/action은 재계산·재실행·변경하지 않는다.
5. Llama/Qwen × MEMIT/Alpha-history evaluator 네 worker를 같은 4-GPU allocation에서
   동시에 시작한다. 한 worker 실패 시 전체 fail-fast한다.
6. 기존 `c0_v3` controller artifact를 read-only로 사용하고, 아직 존재하지 않는
   `caphist_eval_*_c0_v3`, combined, pair output만 exclusive-create한다.
7. 성공한 evaluator directory는 표준 main-repo `local/`로 exact-copy하고 recursive
   byte equality를 다시 검증한 뒤 그 복사본만 merge/analyze한다.
8. NFE field는 evaluator, analyzer, report와 gate에 넣지 않는다.

이 recovery는 새 scientific arm이나 partial model rescue가 아니라, 이미 고정된
8개 action을 처음 평가하는 evaluation-only completion이다.

## 실행·자원

- parent recovery job: `odeedit_capacity_history_pair_v3`.
- server1: 4 GPU, 32 CPU, 260000M, 4시간.
- 완료된 global controller barrier를 재검증한 뒤 evaluator 네 worker만 동시에
  시작한다. worker당 native와 QP evaluator를 순서대로 실행한다.
- Llama 이후 Qwen을 제출하는 순차 pipeline은 금지한다.
- source raw root: `local/results/raw/session01_motivation/`.
- locked staging root:
  `local/scratch/caphist-eval-9900a51/local/results/raw/session01_motivation/`.
- log root: `local/logs/slurm/session01_motivation/`.

## 제출 전 staging 검증

- controller 8개, 208 files, 32,086,056 bytes를 exact-copy했다.
- 상대 경로와 file SHA-256을 정렬해 다시 hash한 source/staged aggregate는 모두
  `3cf4d41dc46b05bec42a43f5a2d87d0047d44b69cace1c8caa745ba17db0a536`다.
- 양쪽 tree의 symlink는 0개이고 `diff -qr --no-dereference`는 8/8 pass했다.
- 잠긴 commit의 `verify_all_controllers()`는 Llama/Qwen 각각 네 branch 모두
  `all_pass=true`로 검증했다. 이 preflight는 model/evaluation row를 load하지 않았다.

## Job 15823 immediate failure와 두 번째 technical repair

- Job `15823`은 네 evaluator worker를 동시에 시작했으나 53초에 fail-fast 종료했다.
- Llama MEMIT/Alpha worker는 evaluation fields와 W0 baseline forward를 수행한 뒤,
  edit replay 전에 evaluator manifest sanitizer에서
  `MV0Error: forbidden artifact field: evaluation`로 실패했다.
- 원인은 `evaluator_policy_parameters()`의 안전한 protocol metadata key
  `evaluation`이 raw outcome 차단용 exact forbidden key와 충돌한 것이다. Metric 값,
  manifest, checkpoint, summary는 기록되지 않았고 Llama의 빈 output directory 두
  개만 생겼다. Qwen worker는 sibling fail-fast로 취소됐다.
- 결과 수치를 관찰하거나 policy/case/gate/model을 바꿀 정보는 없었다. 세 Terra Ultra
  RCA/red/repro agent는 runtime metadata를 검증하지 못해 파일을 읽지 않고 종료했다.
- 두 번째 recovery는 sanitizer를 완화하지 않는다. ODE-Edit-side tracked entrypoint가
  locked evaluator의 policy metadata key만 `evaluation`에서 `metric_protocol`로
  바꾸고, 원래 protocol value는 byte-for-byte 보존한다.
- Entry point SHA-256, locked evaluator SHA-256, locked commit, `metadata_only=true`,
  `sanitizer_relaxation=false`, `controller_action_rerun=false`를 policy parameters에
  포함하므로 evaluator `policy_parameters_sha256`에도 귀속된다.
- 빈 directory는 `local/failed/session01_motivation/job15823/`로 보존 이동한 뒤 같은
  locked `c0_v3` evaluator identity만 다시 사용한다. Controller는 재실행하지 않는다.
- 제출 전 entrypoint SHA-256은
  `5ae85c4e1cca282a96dad6cda8f4a68e0c718adba55ac0bf6eb2c532527b6ea8`, locked
  evaluator SHA-256은
  `fa6f6240774621178c80c6b394c16480d520a4d94a9933871020b3c39edf1b72`다.
- Recovery/evaluator/analysis/controller unit test 16/16과 locked-import entrypoint
  `--help` preflight가 통과했고,
  빈 Llama directory 두 개는 위 failure path로 보존 이동했다.

## Terminal recovery 결과

- 두 번째 evaluator-only recovery job `15824`는 `COMPLETED 0:0`, elapsed
  `00:06:42`다.
- evaluator 8개가 각각 4 checkpoints와 `all_pass=true`로 끝났고 model/pair
  analyzer까지 생성됐다.
- Pair scientific verdict는 `CAPACITY_HISTORY_HARM_SIGNAL`이다. 세부 metric,
  integrity와 claim boundary는
  `experiment-reports/global/2026-08-02-session01-caphist-pair-c0-v3-synthesis.md`와
  `audits/global/2026-08-02-session01-capacity-history-c0-v3-postrun.md`를 따른다.
