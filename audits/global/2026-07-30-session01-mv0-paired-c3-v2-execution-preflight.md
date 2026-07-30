# Session 01 MV-0 paired c3 v2 재실행 gate

- 작성일: 2026-07-30
- 대상: server1에서 Llama/Qwen 3-case fidelity를 GPU 1개씩 동시 재실행
- 최종 판정: `PASS` — exact paired MV-0 c3 v2 한 건에만 유효

## repo/protocol에서 확인한 사실

- 최초 pair `job 15508`은 `FAILED/1:0`, elapsed `00:00:05`였다.
  두 child step은 모델·artifact I/O 전에 `FileNotFoundError`로 종료됐고
  raw output directory는 생성되지 않았다.
- 최초 제출 marker
  `local/state/slurm-submissions/session01_motivation/mv0_pair_c3_v1.submitted/`
  와 원본 log는 실패 이력으로 보존한다.
- 독립 failure analysis는 이를 Motivation 결과가 아닌
  implementation/runtime failure로 판정했다.
- 원인은 `--export=NONE` nested `srun`에서 runner가 bare `git`을 찾지
  못한 것이다. 수정 범위는 repository state 조회용 executable을 검증된
  `/usr/bin/git` 절대 경로로 고정한 것뿐이다.
- fixed model snapshot, case selection, EasyEdit adapter, metric, threshold,
  covariance/projector read-only 정책은 변경하지 않았다.
- venv-only `PATH` 회귀 test를 포함한 CPU suite `40/40`, Python compile,
  세 shell script의 `bash -n`, agent access boundary가 통과했다.
- live cap check는 server1의 현재 project GPU `0`에서 GPU `2`,
  host memory `130000M`을 허용했다. `sbatch --test-only`는 같은
  allocation을 수용했고 실제 job을 생성하지 않았다.

## exact v2 envelope

| 항목 | 값 |
| --- | --- |
| job | `odeedit_mv0_pair_c3v2` |
| node | `devbox` |
| allocation | A6000 2, task 2, task당 CPU 8, host memory `130000M` |
| time | `06:00:00` |
| child 1 | `llama3-8b-inst mv0_llama_c3_v2 3` |
| child 2 | `qwen2.5-7b-inst mv0_qwen_c3_v2 3` |
| marker | `local/state/slurm-submissions/session01_motivation/mv0_pair_c3_v2.submitted/` |

허용 명령은 인자 없는 다음 한 건뿐이다.

```bash
project/run_scripts/submit_session01_mv0_pair_server1.sh
```

Helper는 failure analysis와 이 gate가 tracked인지, clean
`main == origin/main`, session boundary, exact output 부재, 새 durable
marker 부재, same-name active job 부재, server1 aggregate resource cap을
모두 fail-closed로 확인한 뒤 `sbatch --export=NONE`을 한 번만 호출한다.

## red-team 최소 kill 조건

- `/usr/bin/git` 부재, CPU test 또는 shell syntax 실패
- v1 marker/log 삭제나 v1 run ID 재사용
- untracked/dirty 또는 `main != origin/main`
- GPU 2 또는 host memory `130000M` cap 위반
- download, moments 재계산, projector 재계산·수정
- 두 child 중 하나라도 nonzero
- model별 planned/attempted/pass가 `3/3/3`이 아니거나 rollback/fidelity
  bound 위반

성공하더라도 이 실행은 implementation fidelity gate일 뿐이며 Motivation
signal이나 ODE-Edit 개선폭을 증명하지 않는다. 각 model 결과는 별도
analysis agent가 독립 해석하고 pair 전체는 별도 post-run red audit을
받아야 한다.

GH 직접 재제출은 사용자의 명시적 신속 실행 지시와
`PROTOCOL.md:506-509` 예외에 근거한다.
