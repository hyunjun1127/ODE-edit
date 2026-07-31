# Motivation quarter-step pair 실행 전 필수 audit

대상: `session01-qstep4-pair-v1`
범위: 제출에 필요한 최소 technical/scientific/resource gate만 점검

## 범주별 근거

- proposal에서 온 내용:
  edit 경로에서 방향을 다시 계산하는 Motivation을 검증한다.
- repo/protocol에서 확인한 사실:
  이전 MV-2는 technical-valid였지만 `D/32` 첫 state treatment에서
  Llama/Qwen 방향 효과가 갈렸다. protocol은 explicit user/time-critical
  요청에서 GH direct submit을 허용하되 동일 cap과 envelope를 요구한다.
- GH 추정:
  `D/4` hop은 under-manipulation을 해소할 수 있으나, 효과 sign이나
  ODE-Edit 우위는 아직 미확인이다.
- 사용자 확인 필요:
  없음. 사용자가 `D/4` 분할 검토, 설계·제출·분석, Llama/Qwen 동시
  제출을 명시했다.

이 실행은 과거 post-run의 `NO MV3`를 몰래 우회하는 MV-3가 아니다.
사용자의 새 명시 지시에 따른 fresh rank `[112:124]` 독립 Motivation
diagnostic이며 기존 raw/report를 수정하지 않는다.

## Scientific lock

- methods: `A=direction+coefficient refresh`, `B=fixed direction+coefficient
  refresh`, `C=fixed direction+fixed coefficient`
- target: W0 direct-z 한 번 계산 후 exact lineage로 고정
- budget: ordered MEMIT 기준 `D`, 4개 hop 각각 `D/4`, energy
  `E_native/16`, probe `D/64`
- first hop: A/B/C exact shared action/state
- primary: step 4의 `A4-B4`, `B4-C4`, `A4-C4`
- controls: W0 no-op, native ordered one-shot, 동일 update split4
- 금지: best-hop/policy rescue, partial-case exclusion, AlphaEdit 혼합,
  threshold retuning

판정: **PASS**. outcome 전에 feature/action/path/lineage/branch order를
fsync하고 exclusive receipt를 생성하도록 구현됐다.

## Fresh selection 및 fixed artifact

- case count: `12`
- rank: `[112:124]`; first100 및 MV-2 `[100:112]`와 disjoint 검증
- selection manifest:
  `4a8da8d6267fe77b11d0875a5b38e2d34c02d0dfc7bf73113164ede81ce09f2c`
- selected order hash:
  `e34cad88b831d9c9b4bd0cba339fdfcb4186f9e4fce734d3d74db67f2081f04a`
- prior MV-2 order hash:
  `94422fa5faf602d14818036245750241e29343011a8aa780483e3f18c226617e`
- Llama fixed artifact manifest:
  `273d06af367904aac61931331e6aa2ea09204b0db1f6c7557bca16a9e2c84d2b`
- Qwen fixed artifact manifest:
  `d247b27edf638b7a9625954e8bf2aeac745b61247230f8bcf733e991a3ed9d16`

판정: **PASS**. cache/dataset/covariance/projector download·recompute는
wrapper와 runtime에서 차단된다.

## Contract 및 regression

- 전체 unit test: `181 tests`, `OK`, `10.910s`
- quarter/lineage/trajectory targeted test: `22 tests`, `OK`
- `bash -n`: child, pair, submit helper 통과
- `git diff --check`: 통과
- 민감정보/구명칭 scan:
  새 runner/analyzer/wrapper/spec에서 `Stage 0`, `BF-ODE`, private key,
  raw connection/credential pattern 없음
- JSON analyzer:
  mapping key order와 무관하게 exact key set·stream sequence·수치 관계를
  검증하는 regression 포함
- shellcheck:
  host에 설치되지 않아 실행하지 못했다. `bash -n`과 fixed-command/
  allowlist 검증으로 대체했으며 이는 비차단 기록이다.
- delegated review:
  Terra Ultra runtime metadata를 확인할 수 없는 agent는 파일/tool을
  사용하지 않고 종료했다. primary Sol Ultra가 full regression과
  fail-closed review를 수행했으며, 미확인 model agent 결과를 audit
  근거로 사용하지 않았다.

판정: **PASS**. residual P1/P2 blocker 없음.

## Resource, session, collision

- session check:
  `PASS repository=hyunjun1127/ODE-edit server=server1
  session=019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2 model=Sol Ultra`
- active ODE-Edit GPU: `0`
- cap check:
  `requested_gpus=2`, cap `3`, decision `submit_now`
- host memory:
  requested `130000M`, allowed pair total `396234M`, decision `submit_now`
- job collision:
  `odeedit_qstep_pair_v1` active job 없음
- exclusive output/marker:
  두 run directory 및 `qstep4_pair_v1.submitted` 모두 없음
- allocation:
  parent `2 GPU / 16 CPU / 130000M / 12:00:00`, child별
  `1 GPU / 8 CPU / 65000M`
- memory forecast:
  이전 Qwen peak reserved는 약 `47.78 GB`였고 A6000 물리 용량 안에
  남는다. nested path backup으로 여유가 줄 수 있으므로 first fatal
  OOM이면 pair 전체를 fail-fast 종료하며 자동 retry하지 않는다.

판정: **PASS**. 제출 직전 helper가 clean pushed main, output/marker,
active job, session, GPU/memory cap을 다시 검사한다.

## 실행 및 사후 경계

- exact command:
  `project/run_scripts/submit_session01_qstep4_pair_server1.sh`
- GH direct 사유/영향/후속 경로는 `messages/inbox/server1.md`의
  `session01-qstep4-pair-v1` envelope에 기록됨
- raw/log/state는 Git-ignored `local/`만 허용
- peer SH/clone 부재이므로 artifact broadcast는 no-peer 예외를 사후
  보고하고 임의 SSH/rsync를 실행하지 않음
- child nonzero, technical gate 위반, cap 초과, unsafe leakage는 즉시
  중단이며 partial scientific claim을 만들지 않음

- 최종 판정: `PASS` — locked quarter-step Llama/Qwen 동시 pair 한 건에만 유효
