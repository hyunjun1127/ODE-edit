# SH4 → GH: 50983 교체 / cleanup 수리 재제출

최신 사용자 “repair 해서 올려”에 따라 **51056 / odeedit_slmf_S10r2_s4**를 held 검사13항목 PASS 후 release했다. 등록 직후 PENDING/Reason=None/dependency 없음이며 이후 scheduler/log/result 조회는 하지 않았다.

- 구 50983은 미실행 상태에서 정확 ID 취소 및 CANCELLED 확인. 자료 삭제0, 다른 job mutation0.
- 51055는 완화 정책상 hook 비교 완료 뒤 중복 oracle 제거 오류로 실패했다. `reset()` 뒤 추가 `remove()`만 제거하며, 함수 AST/관련 source SHA로 계산 경로 불변을 확인했다.
- 유효한 fixed4 hook 증거를 재사용하여 hook refit0. 새 process의 입력/runtime을 결속한 뒤 남은 full T0 및 기존 gate 조건부 B1→S3→S10을 수행한다. 과거 실패를 전체 T0 PASS로 승격하지 않는다.
- CPU213 PASS; 새 full T0/science actual은 NOT_OBSERVED.
- 실행 `bdaed735f28cb2d0a24e56cc00723373ef857d9d`, tree `a757ee7b9251a4e21a9b19897d17a56054bb5a4d`.
- archive `44c78f6f8929b4a5fbbef09b165f29ca092ed4aa69843ace1c8b0f9cbf07b16b`; lock `09c31436af189650125509abb1e7c8dfa5ad0b9535ba5cd7f8c7cdb12050d9ad`.
- cap2 내 job GPU1/CPU8/60416MiB, W/M checkpoint0, exact_resume NOT_AVAILABLE. 이전 실패 allocation 240 GPU-sec와 새 비용은 분리한다.

[수리 보고](../../../experiment-reports/servers/server4/single-layer-mechanism-first-20260919-v1/repair-r2/cleanup-repair-ko.md), [등록 receipt](../../../audits/servers/server4/single-layer-mechanism-first-20260919-v1/repair-r2/registration.json), [50983 취소 receipt](../../../audits/servers/server4/single-layer-mechanism-first-20260919-v1/repair-r2/cancel-50983.json).

`WAITING_USER_RESUME`, monitoring_active=false, automatic_resume=false. NO_BROADCAST_NOT_REQUIRED.
