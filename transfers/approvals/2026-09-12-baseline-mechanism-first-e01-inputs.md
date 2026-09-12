# GH 승인: E0/E1 baseline 진단의 선택 입력 수신

instruction_id: ODEEDIT-S06-BASELINE-MECHANISM-FIRST-E01-SH1-V1
권한 근거: 사용자 2026-09-12 지정 E0 복원·native replay와 E1 existing per-case 분석 실행 요청.
수신/rsync 단일 owner: SH1, server1.
새 destination: /mnt/raid5/janghj/ODE-edit/local/baseline-mechanism-first-e01/20260912-v1/imports/<source-id>/ 또는 SH1 독립 worktree의 동일 local namespace.

허용 source:
- server2의 local/checkpoint-migration-server4/20260911-v1 catalog와 Git migration-map.csv로 결속한 AlphaEdit singleton L4–L8 entry/다음저장endpoint/B1 reference CP. 실제 경로는 보존 map의 initial6-v1/new177-v1 등 destination을 사용한다.
- server4의 완료 baseline report raw-inventory가 가리키는 singleton AlphaEdit/MEMIT current/seen-full와 E0에 필요한 context/target/source/entry/commit companions. 원 CP는 이미 삭제됐으므로 읽기/복구/재전송 대상이 아니다.
- 이미 GH/S1에 있는 동일 파일은 우선 재사용, 중복수신0.

실행 전 SH1은 실제 파일별 sourcehost/path/bytes/SHA, 필요범위, 예상총량, destination공간을 allowlist manifest로 작성한다. Source SH와 ownership/immutable-completed 확인 후 해당 manifest만 읽기전용 수신한다. 범위 불일치·해시불일치·공간부족은 해당입력 HOLD.
기존 helper의 전체동명경로 broadcast 대신 위 고유 선택 imports로 직접 rsync하는 예외만 허용한다. --delete, overwrite, sourceunlink, unresolved broad glob, fullrepo/fullmodel/전체183CP 수신은 금지한다. 임시partial→검산→새경로seal; 기존수신은 size/SHA가 맞으면 재사용한다.
SH1이 단일 destination writer이며 SH2/SH4는 source지원뿐이다. 다른SH task/session을 일반 실험 재개시키는 권한이 아니다.
수신결과는 transfers/verifications/2026-09-12-baseline-mechanism-first-e01/에 기록. Raw local-only, Git metadata-only. 새raw의 무차별 third-server broadcast는 NO_BROADCAST_NOT_REQUIRED.
