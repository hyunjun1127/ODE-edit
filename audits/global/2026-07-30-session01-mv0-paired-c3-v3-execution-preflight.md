# Session 01 MV-0 paired c3 v3 동시 실행 gate

- 작성일: 2026-07-30
- 대상: server1에서 Llama/Qwen 3-case fidelity를 GPU 1개씩 실제 동시 실행
- 최종 판정: `PASS` — exact paired MV-0 c3 v3 한 건에만 유효

## repo/protocol에서 확인한 사실

- v1 `job 15508`은 model I/O 전 상대 `git` lookup 실패였다. 독립
  failure analysis 뒤 `/usr/bin/git` 절대 경로로 수정했다.
- v2 `job 15512`는 allocation GPU 2개를 받았지만 첫 child step이
  allocation 전체 memory `130000M`을 점유했다. 두 번째 child가
  시작하지 못해 사용자의 동시 제출 요건을 충족하지 않았다.
- GH는 `scontrol show job/step`으로 이 상태를 확인한 뒤 31초에
  `scancel 15512`를 실행했다. 사유는 running-job triage 및 유휴 GPU
  reservation 보호이고, 영향 범위는 해당 pair job 한 건뿐이다.
- v2 Llama는 model load 중 취소되어 partial `manifest.json`만 남았고
  Qwen output은 생성되지 않았다. 이는 fidelity/Motivation 결과로
  집계하지 않는다.
- 두 실패의 marker, log, partial artifact는 `local/`에 그대로
  보존하고 독립 분석 보고서를 Git에 남긴다.
- restricted-PATH 회귀 test를 포함한 CPU suite `40/40`, Python compile,
  세 shell script의 `bash -n`, agent access boundary가 통과했다.
- live cap check는 현재 project GPU `0`에서 GPU `2`, host memory
  `130000M`을 허용했다. `sbatch --test-only`는 parent allocation을
  수용했고 실제 job을 만들지 않았다.

## 최소 수정과 exact v3 envelope

수정은 두 `srun --exclusive --exact` child에 각각 `--mem=65000M`을
명시한 것이다. model snapshot, case selection, EasyEdit adapter,
metric, threshold, covariance/projector read-only 정책은 변경하지
않는다.

| 항목 | 값 |
| --- | --- |
| job | `odeedit_mv0_pair_c3v3` |
| node | `devbox` |
| parent allocation | A6000 2, task 2, task당 CPU 8, host memory `130000M` |
| child 1 | GPU 1, CPU 8, memory `65000M`; `llama3-8b-inst mv0_llama_c3_v3 3` |
| child 2 | GPU 1, CPU 8, memory `65000M`; `qwen2.5-7b-inst mv0_qwen_c3_v3 3` |
| time | `06:00:00` |
| marker | `local/state/slurm-submissions/session01_motivation/mv0_pair_c3_v3.submitted/` |

허용 명령은 인자 없는 다음 한 건뿐이다.

```bash
project/run_scripts/submit_session01_mv0_pair_server1.sh
```

Helper는 두 독립 failure analysis, 이 gate, wrapper와 runner가 tracked
상태인지 확인한다. 이어서 clean `main == origin/main`, session
boundary, v1/v2 marker·log·partial-output 보존, v3 output/marker 부재,
same-name active job 부재, server1 aggregate GPU/memory cap을 모두
fail-closed로 검사한 뒤 `sbatch --export=NONE`을 한 번 호출한다.

## red-team 최소 kill 조건

- v1/v2 failure record, marker, log, partial artifact 손상
- v3 identity/run/marker 불일치
- child별 GPU 1, CPU 8, memory `65000M` 분할 누락
- 실행 직후 `.0`과 `.1` 두 step이 동시에 `RUNNING`이 아님
- `/usr/bin/git` 또는 restricted-PATH 회귀 test 실패
- dirty/untracked worktree 또는 `main != origin/main`
- download, moments 재계산, projector 재계산·수정
- 두 child 중 하나라도 nonzero
- model별 planned/attempted/pass가 `3/3/3`이 아니거나 rollback/fidelity
  bound 위반

실행 직후 `scontrol show step`으로 두 step의 동시 `RUNNING`, 각각
`gres/gpu=1`, `cpu=8`, `mem=65000M`을 확인한다. 불일치 시 GPU 낭비를
막기 위해 즉시 취소하고 다시 제출하지 않는다.

성공하더라도 이 실행은 implementation fidelity gate다. Motivation
signal이나 ODE-Edit 기대 개선폭은 MV-1 이후 held-out diagnostic으로만
추정한다. 각 model은 별도 analysis agent, pair 전체는 별도 post-run
red audit을 거친다.

GH 직접 제출/취소는 사용자의 명시적 신속 실행 지시와
`PROTOCOL.md:506-509` emergency/time-critical/running-job triage 예외에
근거한다.
