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

---

# C1 추가 instruction — `session01-mv1mix-c1-pair-v1`

## 명령 메타데이터

- 대상 서버: `server1`
- 대상 Codex SH session ID: 미지정 — 다른 repo/session 및 미등록 SH 실행 금지
- GH session ID: `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`
- repository CWD / identity:
  `/mnt/raid5/janghj/ODE-edit` / `hyunjun1127/ODE-edit`
- 완료 보고 경로:
  `messages/server-heads/server1/2026-07-31.md`,
  `runs/mv1mix_*_c1_v1/`,
  `experiment-reports/servers/server1/`,
  `audits/servers/server1/`

## 목적과 배경

- proposal에서 온 내용: same-snapshot rewrite-side signal이 실제
  allocation opportunity를 식별하는지 먼저 검증하고, 실패하면 큰 ODE
  실험 전에 controller 방향을 kill 또는 pivot한다.
- repo/protocol에서 확인한 사실: D0+D1은 calibration 전용이며, C1은
  outcome-blind hash split의 고정 fold `0`, model별 12 case에서 adaptive
  `score_mix`와 frozen `static_mix`를 같은 `C`로 비교하는 첫 held-out
  diagnostic이다.
- GH 추정: calibration forecast는 기대 gap의 사전 추정일 뿐 method gain,
  benchmark 개선 또는 MV-2 승인값이 아니다.
- 사용자 확인 필요: 없음. 사용자는 필수 감사만 거쳐 Llama/Qwen을 순차가
  아니라 함께 빠르게 제출하라고 명시했다.

## 실행 권한 envelope

- 허용 write path:
  - raw: `local/results/raw/session01_motivation/mv1mix_{llama,qwen}_c1_v1/`
  - log/state: `local/logs/slurm/session01_motivation/`,
    `local/state/slurm-submissions/session01_motivation/`
  - small report: 위 완료 보고 경로
- Slurm 제출 허용 여부: preflight와 D1 post-run gate를 통과한 GH
  `submit_session01_mv1mix_c1_pair_server1.sh` one-shot에만 `allowed`;
  미등록 SH의 submit/retry/cancel은 `not allowed`
- GPU cap: server1 project 동시 최대 3; 이번 parent 총 2, child별 1.
  overflow면 `pending_resource_cap`으로 중단
- host-memory cap / 요청: GPU당 `198117 MiB`; parent `130000M`,
  child별 `65000M`
- red-team gate:
  `audits/global/2026-07-31-session01-mv1mix-c1-execution-preflight.md`의
  exact `PASS`와
  `audits/global/2026-07-31-mv1mix-d1-pair-v1.postrun.md`의 exact
  `C1 PREPARE`; `warn`은 GH가 명시적으로 core signal과 무관함을 기록한
  경우만 허용하고 `block`은 제출 금지
- artifact broadcast 의무: active peer clone이 없어 즉시 broadcast
  예외를 완료 보고에 기록한다. peer 활성화 후 검증된 protocol helper만
  사용하며 `rsync --delete`는 금지한다.
- Codex session boundary: SH가 생기면 이 section과
  `servers/active/server1.md`에 그 SH 전용 ID를 먼저 등록하고
  `scripts/check-session-boundary.sh <server1-SH-session-id>`를 통과해야
  한다. 현재는 GH one-shot helper의 GH session exact match만 허용한다.

## 예상 산출물

- model별 exact 12 event, event별 feature/action/receipt/direct-z 1,
  exact 6-arm outcome으로 총 72 outcome
- sanitized manifest/streams/summary와 local-only raw direct-z
- model별 별도 agent의 한국어 분석 및 small JSON
- pair-level red post-run 판정과 calibration forecast 대비 held-out gap
- 예상효과 표기는 `adaptive - frozen_static` raw progress unit로만 하며,
  retention·benchmark accuracy·완성된 ODE-Edit gain으로 환산하지 않는다.

## 중단 조건

- session/CWD/repository/job/run/fold/model/resource envelope 불일치
- 두 child가 같은 allocation에서 동시에 시작하지 않음
- policy 또는 forecast가 tracked·clean·pre-outcome frozen 상태가 아님
- exact 12-case fold나 six-arm order 불일치
- outcome leakage, equal-`C`, receipt ordering, rollback/replay/hash/firewall 위반
- moments/projector 재계산·수정·download·deserialize 또는 EasyEdit write
- output/marker 중복, dirty/unpushed Git, active duplicate job, child nonzero

## 금지 사항

- 다른 repository 또는 다른 Codex session의 inbox/job/artifact 조작
- 미등록 SH의 추가 제출·재제출·취소
- raw prompt/target/logit/generation, `.pt`, weights, checkpoint, dataset,
  full log 및 credential의 Git 유입
- online dependency/model download, EasyEdit 수정, cache 재계산
- C1 단일 model/calibration forecast를 cross-model gain, ODE dynamics,
  paper GO 또는 즉시 MV-2 승인으로 과대해석

---

# Untouched 추가 instruction — `session01-mv1mix-untouched-pair-v1`

## 명령 메타데이터

- 대상 서버 / repository:
  `server1` / `/mnt/raid5/janghj/ODE-edit` / `hyunjun1127/ODE-edit`
- 대상 Codex SH session ID: 미지정 — 다른 repo/session 및 미등록 SH 실행 금지
- GH session ID: `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`
- 완료 보고 경로:
  `messages/server-heads/server1/2026-07-31.md`,
  `runs/mv1mix_*_untouched_v1/`,
  `experiment-reports/servers/server1/`,
  `audits/servers/server1/`

## 목적과 배경

- proposal에서 온 내용: 작은 held-out diagnostic에서 same-snapshot routing
  signal이 살아남을 때만 다음 mechanism test를 연다.
- repo/protocol에서 확인한 사실: C1 pair는 technical PASS이며 model별
  analysis에서 양 model 모두 core clear input을 냈다. 실제 실행 권한은
  pair red audit의 exact `UNTOUCHED PREPARE`가 있을 때만 열린다.
- GH 추정: untouched 20은 C1보다 약 1.67배 많은 event다. C1 wall time을
  단순 환산하면 Llama 약 1시간 45분, Qwen 약 2시간 20분이며 `08:00:00`
  안이다.
- 사용자 확인 필요: 없음. 사용자는 필수 감사 후 두 model을 순차가 아니라
  함께 빠르게 제출하라고 명시했다.

## 실행 권한 envelope

- 허용 write path:
  - raw: `local/results/raw/session01_motivation/mv1mix_{llama,qwen}_untouched_v1/`
  - log/state: `local/logs/slurm/session01_motivation/`,
    `local/state/slurm-submissions/session01_motivation/`
  - small report: 위 완료 보고 경로
- Slurm 제출 허용 여부: pair post-run과 untouched preflight exact gate를
  통과한 GH `submit_session01_mv1mix_untouched_pair_server1.sh` one-shot에만
  `allowed`; 미등록 SH submit/retry/cancel은 `not allowed`
- GPU cap: server1 project 동시 최대 3; 이번 parent 총 2, child별 1.
  Overflow면 `pending_resource_cap`으로 중단
- host-memory cap / 요청: GPU당 `198117 MiB`; parent `130000M`,
  child별 `65000M`
- red-team gate:
  `audits/global/2026-07-31-mv1mix-c1-pair-v1.postrun.md`의 exact
  `UNTOUCHED PREPARE`와
  `audits/global/2026-07-31-session01-mv1mix-untouched-execution-preflight.md`의
  exact `PASS`; `block`은 제출 금지
- artifact broadcast 의무: active peer clone이 없어 즉시 broadcast
  예외를 완료 보고에 기록한다. Peer 활성화 뒤 protocol helper만 사용하며
  destructive mirror는 금지한다.
- Codex session boundary: SH가 생기면 SH 전용 ID를 이 inbox와
  `servers/active/server1.md`에 먼저 등록하고
  `scripts/check-session-boundary.sh <server1-SH-session-id>`를 통과해야
  한다. 현재는 GH one-shot helper의 GH ID exact match만 허용한다.

## 예상 산출물

- canonical untouched split exact 20 event/model
- event별 feature/action/receipt/direct-z 1, six-arm outcome 6으로
  model별 총 120 outcome
- sanitized manifest/streams/summary, model별 독립 한국어 분석,
  pair-level final MV-1 red audit
- C1과 같은 `adaptive - frozen_static` raw progress unit, replay/oracle/
  forecast summary; benchmark %, retention, ODE gain으로 환산 금지

## 중단 조건

- session/CWD/repository/job/run/model/resource envelope 불일치
- 두 child가 같은 allocation에서 동시에 시작하지 않음
- selected split이 canonical untouched exact 20이 아니거나 이미 개봉됨
- C1과 policy hash/q/six-arm/controller/threshold/seed가 다름
- fold1을 함께 실행하거나 C1 outcome으로 policy를 refit/retune함
- outcome leakage, equal-`C`, receipt ordering, rollback/hash/firewall 위반
- moments/projector 재계산·수정·download·deserialize 또는 EasyEdit write
- output/marker 중복, dirty/unpushed Git, active duplicate job, child nonzero

## 금지 사항

- 다른 repository/Codex session 조작, 미등록 SH의 제출·재제출·취소
- raw prompt/target/logit/generation, `.pt`, weights, checkpoint, dataset,
  full log 및 credential의 Git 유입
- online dependency/model download, EasyEdit/cache 수정·재계산
- untouched 결과 전 MV-2, ODE dynamics, retention 또는 paper GO 주장
