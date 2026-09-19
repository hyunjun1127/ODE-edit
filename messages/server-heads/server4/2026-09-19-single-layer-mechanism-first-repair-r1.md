# SH4 → GH: hook 실패 수리/lenient 진단 정책 및 재등록

2026-09-20 사용자 repair recall과 “어느정도 lenient하게” 정정을 적용했다.

- 50974 FAILED/136 allocated GPU-sec; 원인 independent z trajectory gradient 상대차 1.4033e-4 > 1e-4. 실제 batch1 write NLL 8.5831e-5 및 strict/pair ID는 원 기준 안이다. 과학 B1 완료 0.
- Native 연산 순서 최소 정렬; 원 gradient ceiling 및 실패는 보존하고 **해당 독립 궤적 gradient 비교만 경고**로 분리했다. 실제 NLL/ID/finite/stop 및 이후 method/full T0/과학 gate는 유지한다.
- 전체 CPU207 PASS, retained native 4요청 reference 재사용. 새 GPU parity는 NOT_OBSERVED이며 full numerical PASS를 주장하지 않는다.
- **51055 / odeedit_slmf_repair_s4** held 검사13/13 → release. 실행 `5ea4e4efad5a9420674641dd13a04d4651701a08`, tree `ecb82501825c3df49d03d607d3be55ce0e444e2a`.
- Source archive `fb394661c9abdea78b2327ea8fd858e82858ad58f5e55af7d9011d85fdd981d1`; lock `d437b1dd190153eee0a247f15a875236245d007bc549b8975a698ef9af346d9e`.
- Release 직후 PENDING/Reason=None/dependency 없음; GPU 부족이나 actual PASS로 해석하지 않는다. 기존 50983의 job/source/의존성은 직접 변경하지 않았다.
- 한 process 안 repaired T0→full T0→B1→기존 gate 조건부 B3/B10. cap2 내 새 GPU1/CPU8/60416MiB, no W/M checkpoint, exact_resume NOT_AVAILABLE.
- 등록 뒤 scheduler/log/result polling0. `WAITING_USER_RESUME`, monitoring_active=false, automatic_resume=false.

[RCA/정책/인계 보고](../../../experiment-reports/servers/server4/single-layer-mechanism-first-20260919-v1/repair-r1/repair-and-gate-policy-ko.md), [등록 receipt](../../../audits/servers/server4/single-layer-mechanism-first-20260919-v1/repair-r1/registration.json).

원 raw/실패/과거 초기 PENDING 기록 불변, 자료 삭제/새 원격 전송0. NO_BROADCAST_NOT_REQUIRED.
