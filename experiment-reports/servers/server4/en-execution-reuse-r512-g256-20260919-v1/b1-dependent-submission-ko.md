# B1 의존성 대기 등록 / 모니터링 중단

사용자 “B1 부분 pending으로 걸어놓을 수 있나?”에 따라 B1 **50449**를
`afterok:50410`으로 등록·held inspection·release했다.
2026-09-18T20:56:53Z(한국 2026-09-19 05:56:53)의 확인은
`PENDING / Dependency / afterok:50410(unfulfilled) / RunTime=00:00:00`이다.
이는 정상 의존성 대기이며 실제 B1 실행/검증/완료를 뜻하지 않는다.

준비가 성공하면 같은 task 한 lane에서 B1 한 job이 실행된다.
READY·640문서·source/input identity를 시작 전에 별도 effective lock으로 결속한다.
준비 실패 또는 유효 READY 부재 시 실행하지 않는다. 기존 준비 source/job/자료는 변경하지 않았다.
Agent 모니터링·daemon·callback·자동 재제출은 없다. 사용자 recall까지 추가 조회하지 않는다.

| 항목 | 값 |
| --- | --- |
| Job | `50449 / odeedit_en_reuse_g256_B1_s4` |
| 의존성 | `afterok:50410` |
| Source | `5574f2c63a355043ba28c8e557383be3075a5e48` |
| Tree | `6707783f6b4042fa02346a393052f0554651a212` |
| Archive SHA256 | `47b71a42f52e543bfc6135064ac67fd8047d90d58923f1730481afa06e6cc99e` |
| Admission lock SHA256 | `74b82b4294f96f957fcbbed47d6fd0b2731d211b253ab05045a9fb702ab5651a` |
| Submission receipt SHA256 | `1ad27eb5eddac633877f092737002e7bd63076593fcc766eea3f7a3ff5e66480` |
| 자원 | 1 GPU / 8 CPU / 60416 MiB / 24 h / export NONE / Requeue 0 |
| 동시 task / project GPU 상한 | 1 / 2 |
| CPU regression | 145 PASS / 40.744초; actual Llama PASS 아님 |
| 제출 시 여유 / 보수적 reserve | 131347988480 / 121057705984 bytes; 독점예약·waiver 아님 |
| Scope | cold B100 한 batch, legacy/reuse 두 schedule; 공유 N4 endpoint |
| Sequential | 미승인, `max_batches=1`, `sequential_authorized=false`, `auto_continue=false` |

Local root: `/data/janghj/ODE-edit/local/en-execution-reuse/20260919-v1/B1/attempt-v1/`.
원 admission lock은 불변이다. `execution.ready.lock.json` 및 `preparation-binding-at-start.json`은
유효 READY가 생성되고 B1이 실제 시작할 때만 생긴다. 후속 분석은 두 lock의 lineage를 연결한다.
수치·method·evaluation·checkpoint 의무는 유지하며 B2/held sequential 등록은 없다.

Local compact handoff:
`local/en-execution-reuse/20260919-v1/receipts/b1-dependent-pending-handoff-r1.json`.
상세 결과 보고/main 통합은 이번 대기 등록에서 수행하지 않았다.
