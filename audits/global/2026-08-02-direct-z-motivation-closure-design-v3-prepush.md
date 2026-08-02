# Direct-z Motivation 종료 설계 v3 — 최소 pre-push audit

- 날짜: 2026-08-02
- 대상: `project/proposals/sections/03-direct-z-review-and-motivation-closure-design.md`
- 판정: **waived-for-design-only**
- 실행 판정: **block — 별도 사용자 실행 지시와 execution preflight 전에는 구현·Slurm 제출 금지**

## Agent gate

Semantics, metric/fairness, closure-design 관점의 Terra Ultra subagent 세 개를
격리 호출했다. 세 agent 모두 runtime metadata에서 `gpt-5.6-terra / ultra`를
검증하지 못해 repo 파일을 읽지 않고 즉시 종료했다. 따라서 독립 agent review는
0건이며 pass로 세지 않는다.

## GH 확인

- Primary session boundary는 `Sol Ultra`, server1, 이 repository로 확인했다.
- Canonical final synthesis, MEMIT/Alpha pair report, compact analysis JSON,
  replay-lock, ODE-Edit-side direction/share 구현을 대조했다.
- 기존 replay-lock에 per-layer `B0` hash와 `v0`가 없음을 확인하고, 설계를
  replay-lock anchor와 legacy feature/action anchor의 two-source sentinel로
  정정했다.
- 기존 terminal endpoint를 새 sample로 재사용하지 않고, common-shadow 2x2의
  local effect만 estimand로 유지했다.
- EasyEdit, 실행 코드, raw artifact, Slurm state는 변경하지 않았다. 새 job도
  제출하지 않았다.

## Waiver 경계

독립 Terra review 부재는 원래 pre-push sensitive gate의 block 사유다. 그러나 이번
patch는 실행 권한을 열지 않는 proposal-side 설계 정정이고, 오히려 잘못된 legacy
identity claim을 축소한다. GH는 이 문서 patch의 push만 waive한다. 실제 구현·실행
전에는 Terra Ultra review 또는 사용자가 승인한 대체 gate, resource cap, firewall,
exact rollback, precomputed-only 검사를 새 preflight에서 통과해야 한다.
