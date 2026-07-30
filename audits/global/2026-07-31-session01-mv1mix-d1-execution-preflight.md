# Session 01 Motivation — MV-1 score-mix D1 동시 execution gate

- 작성일: 2026-07-31
- 범위: server1에서 고정 Llama/Qwen의 `calibration[8:20]` score-mix D1을
  GPU 1개씩 동시에 실행
- 현재 판정: `PASS` — D1 analyzer 독립 red 및 최종 정적 검증 완료

## 네 범주

### Proposal에서 온 내용

- same-snapshot rewrite-side signal이 event별 allocation opportunity를
  식별하는지 calibration에서 검증한 뒤에만 ODE refresh나 큰 실험을 연다.
- static allocation으로 설명되거나 signal이 actual progress로 이어지지 않으면
  현재 controller 방향을 kill 또는 pivot한다.

### repo/protocol에서 확인한 사실

- D0 pair job `15546`은 두 child가 2026-07-31 00:06:55 KST에 동시에
  시작했고 Llama/Qwen 모두 `planned/attempted/pass=5/5/5`, outcomes `25`,
  exact rollback과 receipt count를 통과했다.
- 독립 D0 analyzer에서 두 model 모두 5/5 `G_mix`가 각 model의
  `e_m=1e-12`를 넘었다. 따라서 D0 early-kill 조건은 성립하지 않으며
  canonical rule상 paired D1만 허용된다.
- D1은 기존 pilot `[0:3]`과 D0 `[3:8]`을 제외한 calibration `[8:20]`
  12 case/model이다.
- 고정 대상은 `llama3-8b-inst`, `qwen2.5-7b-inst`이며 하나의 paired
  allocation에서 동시에 시작한다.
- EasyEdit와 pinned moments/projector는 read-only다. moments/projector
  재계산, 수정, download와 projector deserialization은 금지된다.
- GH session은 `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`, CWD는
  `/mnt/raid5/janghj/ODE-edit`, repository ID는 `hyunjun1127/ODE-edit`다.
- server1 project cap은 GPU 3개, GPU당 Slurm host-memory cap은
  `198117 MiB`다. 이번 요청은 GPU 2개와 host memory `130000M`이다.

### GH 추정

- D0의 positive realized contrast는 continuous mixture가 uniform과
  구별되는 최소 signal이지만, 5-case calibration이며 adaptive probe
  overhead도 있어 method gain이나 population estimate가 아니다.
- D0 wall-clock은 Llama 약 25분, Qwen 약 34분이었다. 같은 per-case
  contract의 12-case D1은 model별 약 1–1.5시간으로 추정하며 `06:00:00`
  limit 안이다.
- D1은 calibration 17 case를 완성하고 slope-only frozen static comparator를
  고정하기 위한 단계다. D1 positive도 GO, confirmatory success, MV-2를
  뜻하지 않는다.

### 사용자 확인 필요

- 없음. 사용자는 필수 감사만 수행하고 Llama/Qwen을 순차가 아니라 함께
  빠르게 제출하라고 명시했다. D0의 사전 고정 continue rule이 충족됐다.

## exact 실행 envelope

| 항목 | 고정값 |
| --- | --- |
| parent | `odeedit_mv1mix_d1_pair_v1`; `devbox`; A6000 2; task 2; CPU 16; `130000M`; `06:00:00` |
| Llama child | GPU 1; CPU 8; `65000M`; `llama3-8b-inst mv1mix_llama_d1_v1 8 12` |
| Qwen child | GPU 1; CPU 8; `65000M`; `qwen2.5-7b-inst mv1mix_qwen_d1_v1 8 12` |
| runner | `python -m project.run_scripts.ode_edit_motivation.mv1_score_mix` |
| output | `local/results/raw/session01_motivation/{mv1mix_llama_d1_v1,mv1mix_qwen_d1_v1}` |
| marker | `local/state/slurm-submissions/session01_motivation/mv1mix_d1_pair_v1.submitted/` |

허용 명령은 인자 없는 다음 한 건뿐이다.

```bash
project/run_scripts/submit_session01_mv1mix_d1_pair_server1.sh
```

Helper는 D0 pair post-run `D1 CONTINUE`, 이 preflight의 exact `PASS`,
runner·D1 analyzer·tests·wrappers의 Git 추적 상태, clean
`main == origin/main`, agent/session/repository boundary, 두 output과 marker
부재, 동일 job 부재, server1 GPU/host-memory cap을 fail-closed로 검사한 뒤
`sbatch --export=NONE`을 정확히 한 번 호출한다.

## 결과 전 analysis/static policy lock

- D1 analyzer는 exact `12×5` outcome panel과 artifact/hash/receipt/rollback/
  equal-`C`/selection firewall을 검증한다.
- Model별 D0 `[3:8]` 5 feature와 D1 `[8:20]` 12 feature의 five
  single-layer slope만 평균해 `w_static=relu(sbar)/L2`를 고정한다.
- All-nonpositive fallback은 max-slope one-hot, exact tie는 낮은 layer다.
- Static fit API는 outcome을 받지 않으며 output에 `outcome_fields_used=[]`,
  feature-panel hash와 policy hash를 남긴다.
- D1 outcome은 descriptive calibration으로만 보고하며 effect claim, GO,
  confirmatory success 또는 MV-2 승격에 쓰지 않는다.

## 최소 red-team 중단 조건

- 두 child가 동시에 시작하지 않거나 각각 GPU 1, CPU 8, memory `65000M`이 아님
- repo/session/job/run ID/slice/case count가 exact envelope와 다름
- dirty worktree, `main != origin/main`, 기존 output/marker, 동일 active job
- D0 pair post-run `D1 CONTINUE` 또는 D1 analyzer precommit 부재
- online download, moments/projector 재계산·수정·deserialize, EasyEdit write
- outcome leakage, receipt ordering, equal-`C`, rollback/replay/hash/firewall 위반
- 어느 child든 nonzero, 또는 model별 `planned/attempted/pass != 12/12/12`

제출 직후 두 Slurm step의 동시 `RUNNING`과 child별 자원을 확인한다.
불일치 시 해당 pair 한 건만 취소하고 재제출하지 않는다. 완료 후 model별
독립 analysis agent와 pair post-run red audit만 수행한다. 현재 다른 active
server-head가 없으므로 즉시 artifact broadcast는 예외로 기록한다.

GH 직접 제출 근거는 사용자의 time-critical 동시 실행 지시와
`PROTOCOL.md:506-509`다. 영향 범위는 이 paired D1 한 건과 위 `local/`
경로뿐이다.

## 사전검증 증거

- 독립 red agent는 raw outcome과 기존 model report를 열지 않고 analyzer와
  test만 검토했다. D1 `[8:20]` exact `12×5` contract, D0+D1 17개
  feature-only fit, outcome input 차단, ReLU/L2와 fallback/tie,
  panel/policy hash, CLI output firewall에서 P0/P1 blocker가 없었다.
- EasyEdit 고정 runtime에서 전용 D1 analyzer test `6/6`, Motivation 전체
  test `78/78`, `py_compile`, wrapper/helper `bash -n`,
  `git diff --check`, `scripts/check-agent-access.sh --all-changed`가 통과했다.
- `sbatch --test-only`이 parent GPU 2·CPU 16·`130000M` 요청을 수락했다.
  이는 실제 제출이 아니며 job/output/marker는 생성되지 않았다.
- session boundary는 GH session/repository/path를 exact match했고,
  cap check는 active GPU 0 + requested GPU 2 ≤ project cap 3,
  parent memory `130000M` ≤ host cap `396234 MiB`를 확인했다.

- 최종 판정: `PASS` — MV-1 score-mix D1 동시 pair 한 건에만 유효
