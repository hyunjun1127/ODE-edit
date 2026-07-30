# Session 01 MV-0 paired c3 최소 실행 gate

- 작성일: 2026-07-30
- 대상: server1에서 Llama/Qwen 3-case fidelity를 GPU 1개씩 동시 실행
- 최종 판정: `PASS` — exact paired MV-0 c3 한 건에만 유효

## 선행 사실

- Llama smoke `mv0_llama_smoke_v1`: 독립 analysis/red PASS
- Qwen smoke `mv0_qwen_smoke_v1`: 독립 analysis/red PASS
- 두 smoke 모두 planned/attempted/pass `1/1/1`, exact rollback,
  native-adapter weight/logits/NLL error `0.0`, scheduler `COMPLETED/0:0`
- current CPU suite: 39/39 PASS
- 두 model fixed provenance와 동일 case-ID-only selection manifest preflight
  PASS

이 판정은 implementation fidelity calibration을 3 deterministic case로
확장하는 것뿐이며 Motivation signal이나 method gain을 측정하지 않는다.

## Exact paired envelope

| 항목 | 값 |
| --- | --- |
| job | `odeedit_mv0_pair_c3` |
| node | `devbox` |
| allocation | A6000 2, task 2, task당 CPU 8, host memory `130000M` |
| time | `06:00:00` |
| child 1 | `llama3-8b-inst mv0_llama_c3_v1 3` |
| child 2 | `qwen2.5-7b-inst mv0_qwen_c3_v1 3` |
| marker | `local/state/slurm-submissions/session01_motivation/mv0_pair_c3_v1.submitted/` |

허용 명령은 인자 없는 다음 한 건뿐이다.

```bash
project/run_scripts/submit_session01_mv0_pair_server1.sh
```

Pair helper는 두 smoke의 analysis/red 문서와 이 audit, paired/runtime
wrapper, runner가 tracked인지 확인하고, untracked를 포함한 clean
`main == origin/main`, session boundary, 두 output의 부재, durable pair
marker, same-name job 부재를 검사한다. 이어서 aggregate
`scripts/check-slurm-resource-cap.sh server1 2 130000`이 통과할 때만
`sbatch --export=NONE`을 한 번 호출한다.

Live cap check와 `sbatch --test-only`는 2 GPU, CPU 16, `130000M`,
`devbox` envelope를 수용했으며 실제 job은 생성하지 않았다.

한 allocation 안에서 두 `srun --exclusive --exact` step이 각각 GPU 1개와
CPU 8개만 받아 동시에 공통 read-only EasyEdit runtime을 실행한다. 각 manifest/summary는
같은 Slurm job ID와 exact pair job name/node를 기록한다. 기존 moments만
read-only load하고 projector는 verify-only이며 download/recompute는 fatal이다.

## 중단과 후속

- 두 child 중 하나라도 nonzero면 pair job은 실패다. 다른 child는 종료까지
  기다려 denominator를 숨기지 않는다.
- 각 model은 planned/attempted/pass `3/3/3`, exact rollback, fidelity bound,
  compact firewall을 독립적으로 통과해야 한다.
- 한 model이라도 실패하면 MV-1을 열지 않는다.
- 성공 뒤 model별 결과 report는 실행과 분리된 별도 analysis agent가 각각
  작성하고, 별도 post-run red agent가 pair resource/concurrency를 판정한다.
- raw artifact/log/direct-z는 `local/`에만 두며 peer SH/clone이 없으면
  broadcast exception을 기록한다.

GH 직접 제출은 사용자의 명시적 실행 지시와 `PROTOCOL.md:506-509` 예외에
근거한다. 사용자는 이후 양 model job을 순차 대기하지 말고 가능한 한 함께
제출하라고 명시했다.
