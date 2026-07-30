# server1 inbox — Session 01 Motivation MV-1 score-mix D0

## 명령 메타데이터

- 명령 ID: `session01-mv1mix-d0-pair-v1`
- 작성 시각: 2026-07-30
- 작성 agent: `head-server1-gh`
- 대상 서버: `server1`
- 대상 Codex session ID: 미지정 — server-head 등록 전 SH 실행 금지
- 대상 repository CWD: `/mnt/raid5/janghj/ODE-edit`
- 대상 Git repository identity: `hyunjun1127/ODE-edit`
- 우선순위: high
- ack 필요 여부: server-head session 등록 후 yes
- 완료 보고 경로:
  `messages/server-heads/server1/2026-07-30.md`,
  `runs/mv1mix_*_d0_v1/`,
  `experiment-reports/servers/server1/`

## 목적과 배경

v1 pilot은 technical fidelity를 통과했지만 finite six-arm set이 continuous
routing을 식별하지 못했다. D0는 pilot case를 제외한 calibration `[3:8]`
다섯 case에서 outcome-free `score_mix`와 same-`C` `uniform`의 realized
progress를 비교해 Motivation을 최소 비용으로 early-kill하는 diagnostic이다.

현재 server-head가 없고 사용자가 time-critical 동시 제출을 명시했으므로
GH가 `PROTOCOL.md:506-509` 예외로 exact pair 한 건만 직접 제출한다. 향후
server-head는 자신의 session ID가 `servers/active/server1.md`에 등록된 뒤에만
ack, monitoring, post-run 보고와 가능한 artifact broadcast를 인수한다.

## 실행 권한 envelope

- 허용 write path:
  - raw: `local/results/raw/session01_motivation/mv1mix_{llama,qwen}_d0_v1/`
  - log: `local/logs/slurm/session01_motivation/`
  - state: `local/state/slurm-submissions/session01_motivation/`
  - small report: `runs/mv1mix_*_d0_v1/`,
    `experiment-reports/servers/server1/`,
    `audits/servers/server1/`,
    `messages/server-heads/server1/`
- Slurm 제출 허용 여부: GH one-shot helper에만 allowed; 미등록 SH의 추가
  제출·재제출은 not allowed
- GPU cap: server1 project 동시 최대 3; 이번 pair 총 2, child별 1
- host-memory cap / 요청 memory: GPU당 `198117 MiB`; parent `130000M`,
  child별 `65000M`
- red-team gate:
  `audits/global/2026-07-30-session01-mv1mix-d0-execution-preflight.md`의
  exact `PASS`
- artifact broadcast 의무: active peer가 없으므로 즉시 실행 예외. peer
  활성화 후 protocol helper로 검증·전송하거나 no-peer 예외를 완료 보고에 기록
- 완료 보고 경로: 위 명령 metadata와 동일
- session boundary 확인 command:
  `scripts/check-session-boundary.sh <등록된-server1-SH-session-id>`

## 예상 산출물

- model별 exact 5 event, event별 feature 1, commitment 1, outcome 5, receipt 1
- sanitized manifest/features/actions/outcomes/events/summary
- model별 한국어 독립 분석, pair post-run red audit, small run metadata

## 중단 조건

- session/CWD/repository/job/run/slice/resource envelope 불일치
- Llama/Qwen이 같은 allocation에서 동시에 시작하지 않음
- moments/projector recompute·download·write·deserialize 또는 EasyEdit write
- outcome leakage, receipt ordering, equal-`C`, rollback/replay/hash/firewall 위반
- output/marker 중복, dirty/unpushed Git, 어느 child든 nonzero

## 금지 사항

- 미등록 SH의 submit/retry/cancel, 다른 repo 또는 다른 Codex session 조작
- raw prompt/target/logit/generation, weights, checkpoint, dataset, full log의 Git 유입
- credential, token, private SSH 값의 Git 기록
- `rsync --delete`, destructive mirror, EasyEdit 수정, online dependency/model download
- D0 positive를 method gain, GO, confirmatory 또는 MV-2 승인으로 해석

---

# D1 추가 instruction — `session01-mv1mix-d1-pair-v1`

## 목적과 배경

D0의 두 model 모두 5/5 `G_mix`가 replay envelope를 넘어 early-kill
조건이 깨졌다. Canonical continue rule에 따라 pilot/D0와 겹치지 않는
calibration `[8:20]` 12 case/model을 동시에 실행해 calibration 17 case를
완성하고 slope-only frozen static comparator를 고정한다.

## 실행 권한 envelope

- 대상 Codex session ID: 미지정 — server-head 등록 전 SH 실행 금지
- 허용 write path:
  - raw: `local/results/raw/session01_motivation/mv1mix_{llama,qwen}_d1_v1/`
  - log/state: `local/logs/slurm/session01_motivation/`,
    `local/state/slurm-submissions/session01_motivation/`
  - small report: `runs/mv1mix_*_d1_v1/`,
    `experiment-reports/servers/server1/`, `audits/servers/server1/`
- Slurm 제출 허용 여부: GH one-shot helper에만 allowed; 미등록 SH의 추가
  제출·재제출은 not allowed
- GPU cap: server1 project 최대 3; 이번 pair 총 2, child별 1
- host-memory cap / 요청: GPU당 `198117 MiB`; parent `130000M`, child별
  `65000M`
- red-team gate:
  `audits/global/2026-07-31-session01-mv1mix-d1-execution-preflight.md`의
  exact `PASS`
- artifact broadcast 의무: active peer 없음 예외를 완료 보고에 기록하고,
  peer 활성화 후 protocol helper로 검증·전송
- 완료 보고 경로: `messages/server-heads/server1/2026-07-31.md`,
  `runs/mv1mix_*_d1_v1/`, `experiment-reports/servers/server1/`
- session boundary: 등록된 server1 SH session ID로
  `scripts/check-session-boundary.sh` 통과

## 예상 산출물

- model별 12 event, event별 feature/action/receipt 1, outcome 5
- sanitized manifest/streams/summary
- model별 독립 D1 분석과 D0+D1 slope-only frozen static policy
- pair post-run red audit

## 중단 조건과 금지 사항

- D0의 instruction과 같은 session/repository/resource/read-only/firewall
  중단 조건을 적용한다.
- D1 slice가 `[8:20]`이 아니거나 pilot/D0 case와 겹치면 중단한다.
- Static policy에 outcome을 사용하거나 D1 positive를 gain/GO/MV-2로
  해석하는 것을 금지한다.
