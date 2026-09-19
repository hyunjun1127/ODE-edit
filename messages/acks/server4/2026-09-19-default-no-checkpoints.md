# SH4 기본 checkpoint 미저장 정책 ACK

- Instruction: `ODEEDIT-ALL-SH-DEFAULT-NO-CHECKPOINT-20260919-V1`.
- Publication: `4d871dc6aeb328675d4912f4525ed23434bc0ad8`.
- FULL_READ: 게시 envelope, 정책 전문, PROTOCOL 변경분 확인.
- 실제 host/owner: `server4` / `janghj`; repository `hyunjun1127/ODE-edit`, root `/data/janghj/ODE-edit`.
- 등록 session: `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`, `servers/connection-inventory.md`와 결속.
- 현재 mechanism-first 미제출 실행기는 직전 사용자 직접 지시로 `save_checkpoints=false`, W/M/RNG/selected weight/복원용 전체 delta 영속 저장 없음. 활성 저장 예외 없음. 기존 원 사양의 CP 요구는 해당 task의 최신 직접 noCP 지시로 대체됨.
- 같은 process의 RAM state/rollback/history는 유지. `checkpoint_saved=false`, `exact_resume=NOT_AVAILABLE`.
- 기존 제출 50974의 immutable source/lock 및 기존 checkpoint는 수정·삭제하지 않음. 이 정책 전달로 scheduler/result 조회 또는 다른 task 재개 없음.
- 현재 별도 승인된 실행기 완성/dependency 등록 준비에만 적용. GPU cap과 모니터링 중지 정책 불변. `NO_BROADCAST_NOT_REQUIRED`.
