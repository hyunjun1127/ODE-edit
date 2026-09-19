# SH2 checkpoint mechanism — terminal evidence report

Instruction ODEEDIT-S06-S2-CHECKPOINT-MECHANISM-AUDIT-20260920-V1.
사용자 재개 후 job51071/51116 종료, allocated GPU-sec269. C00 실제 prefix 검증 PASS,
C01 NS NLL row237 및 M1 Gram4원소가 고정 허용치 초과. Projector singleton shape-header
검사만 수정했고 성공·실패 평가 bytes/기준을 그대로 재사용했다. NLL 재시도/완화0.

31cells: PASS5(보고Z00 포함),FAILED1,BLOCKED25. 후속 두-lane 확대는 의존 gate가
충족되지 않아 미제출. H1 SUPPORTED(이 stream의 관측 차이),H2–H4 UNRESOLVED.
Archive10k final RS9939/10000,PS19136/20000,NS65348/100000. 이는 새 모델 parity가 아니다.
Actual B1 성공 수100/190/867은 일치하지만 성공 수가 row parity를 대체하지 않는다.

보고서: `experiment-reports/servers/server2/checkpoint-mechanism-audit-2026-09-20-v1/report-ko.md`
SHA256 `464ebf33a2a315ce3aae9d3397d6a74cd83d330ef21668228d5b1d3ca415cff8`.
원 raw/12CP local 보존, 새 checkpoint/edit/z fitting0, scientific_promotion=false.
NO_BROADCAST_NOT_REQUIRED. 본인 source/report만 main 통합; 완료 후 STOP, 후속 새실험0.
