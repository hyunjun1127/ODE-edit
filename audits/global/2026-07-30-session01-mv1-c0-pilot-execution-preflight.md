# Session 01 Motivation — MV-1 C0 3-case 동시 pilot execution gate

- 작성일: 2026-07-30
- 범위: server1에서 고정 Llama/Qwen의 MV-1 C0 3-case pilot을 GPU 1개씩 동시에 실행

## 네 범주

### Proposal에서 온 내용

- layer별 edit progress 이질성이 실제 outcome을 예측하는지 먼저 진단한다.
- 이 pilot은 ODE-Edit의 개선폭을 확정하지 않고, C0 측정·실행 계약이 성립하는지만 확인한다.

### repo/protocol에서 확인한 사실

- 고정 대상은 `llama3-8b-inst`, `qwen2.5-7b-inst` 두 model이며 하나의 paired allocation에서 동시에 시작한다.
- EasyEdit와 pinned moments/projector는 read-only다. 재계산, 수정, download는 금지된다.
- GH session은 `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`, CWD는 `/mnt/raid5/janghj/ODE-edit`, repository ID는 `hyunjun1127/ODE-edit`다.
- server1 cap은 project GPU 3개다. 이번 요청은 GPU 2개와 host memory `130000M`이다.
- raw artifact/log/state는 `local/`에만 쓰며 Git에는 넣지 않는다.

### GH 추정

- 3 case는 runtime, action precommit, rollback, artifact count를 검증하기 위한 최소 pilot이다.
- 수치적 기대효과 추정은 이 pilot의 안정성 확인 뒤 잠긴 C0/held-out 분석에서만 가능하다.
- canonical 구현 명세의 일반 C0 wave limit `12:00:00`과 달리 이 gate는
  요청된 3-case pilot 한 건만 `06:00:00`으로 제한한다. full C0의 limit을
  바꾸는 결정이 아니다.

### 사용자 확인 필요

- 없음. 사용자가 Llama 뒤 Qwen을 기다리지 말고 함께 제출하라고 명시했다.

## exact 실행 envelope

| 항목 | 고정값 |
| --- | --- |
| parent | `odeedit_mv1_c0p_pair_v1`; `devbox`; A6000 2; task 2; CPU 16; `130000M`; `06:00:00` |
| Llama child | GPU 1; CPU 8; `65000M`; `llama3-8b-inst mv1_llama_c0p_v1 3` |
| Qwen child | GPU 1; CPU 8; `65000M`; `qwen2.5-7b-inst mv1_qwen_c0p_v1 3` |
| runner | `python -m project.run_scripts.ode_edit_motivation.mv1_calibration` |
| output | `local/results/raw/session01_motivation/{mv1_llama_c0p_v1,mv1_qwen_c0p_v1}` |
| marker | `local/state/slurm-submissions/session01_motivation/mv1_c0p_pair_v1.submitted/` |

허용 명령은 인자 없는 다음 한 건뿐이다.

```bash
project/run_scripts/submit_session01_mv1_c0_pair_server1.sh
```

Helper는 지정 wrapper·runner·test·spec·audit의 Git 추적 상태, clean
`main == origin/main`, agent/session/repository boundary, 두 output과 marker
부재, 동일 job 부재, server1 GPU/host-memory cap을 fail-closed로 검사한 뒤
`sbatch --export=NONE`을 정확히 한 번 호출한다.

정적 execution 검증에서 세 shell script의 `bash -n`, global-head
write-boundary check, Motivation CPU suite `55/55`, `mv0_fidelity.py`와
`mv1_calibration.py` Python compile이 통과했다. `sbatch --test-only`는 GPU
2, CPU 16, `130000M` allocation을 수용했고 실제 job은 생성하지 않았다.
이 gate는 runner의 Motivation claim을 승인하지 않는다.

독립 red review는 `epsilon(q=1/16) == d(q=1/256)`인 endpoint 중첩을
발견했다. runner는 각 fraction의 feature/action을 계산 직후
write→`fsync`→fraction-specific exclusive receipt 순서로 확정한 다음 다음
fraction probe를 열도록 수정했고, 재검토에서 P0/P1 blocker 없음 판정을
받았다. 따라서 작은 budget action은 같은 endpoint의 큰-budget probe보다
먼저 고정된다.

## 최소 red-team 중단 조건

- 두 child가 동시에 시작하지 않거나 각각 GPU 1, CPU 8, memory `65000M`이 아님
- repo/session/job/run ID/case count가 exact envelope와 다름
- dirty worktree, `main != origin/main`, 기존 output/marker, 동일 active job
- online download, moments/projector 재계산·수정, EasyEdit write
- action receipt가 outcome보다 늦거나 rollback/hash/firewall 위반
- 어느 child든 nonzero, 또는 model별 `planned/attempted/pass != 3/3/3`

제출 직후 두 Slurm step의 동시 `RUNNING`과 child별 자원을 확인한다.
불일치 시 해당 pair 한 건만 취소하고 재제출하지 않는다. 완료 후 model별
독립 analysis agent와 pair post-run red audit을 수행한다. 현재 다른 active
server-head가 없으므로 즉시 artifact broadcast는 예외로 기록하고, 향후
peer가 활성화되면 protocol helper로 검증·전송한다.

GH 직접 제출 근거는 사용자의 time-critical 동시 실행 지시와
`PROTOCOL.md:506-509`다. 영향 범위는 이 paired pilot 한 건과 위 `local/`
경로뿐이다.

- 최종 판정: `PASS` — MV-1 C0 3-case 동시 pilot 한 건에만 유효
