# SH4 v2 인계

## 최신 명시 recall — 본실험 6/6 제출 후 자원 대기 인계

기술49421 COMPLETED/0:0, actual 9단계 PASS; allocated3379GPU초. 본실험49466_[0–5]는 C45678/N4/F48/G48/C4/C48로 정상 held 검사·release, array%2/각1GPU/8CPU/60416MiB/16h다. 마지막 관측은 전부 Resources 대기, GPU8/8 할당·가용0이다. host memory 여유도 부족하여 GPU만의 단독 원인이라고 쓰지 않는다. Priority 양수/dependency 해제/수동hold0를 확인했다.

**MAIN_GPU_RESOURCE_PENDING_HANDOFF / MONITORING_PAUSED_AWAITING_USER**. 기술 PASS와 본실험 초기 gate를 구분하며 본실험 gate는 NOT_OBSERVED다. 원 실행21297ec/lock 불변, no disk W/M checkpoint. [최신 본실험 인계](../../../experiment-reports/servers/server4/sequential-local-z-allocation-seq1000-2026-09-17-v2/resume-main-r1/main-initial-handoff-ko.md).

Resume `/data/janghj/ODE-edit/local/sequential-local-z-allocation/20260917-v2/resume-r1/resume-manifest.json` SHA `dd5b95aa340dd12bcc371b1558592c632bc008494dcc98dcb600f32a0eafd52e`. monitoring_active=false / automatic_resume=false; jobs 변경·후속조회0.

## 이전 기술 PENDING 인계 — 역사적 관측 보존

ODEEDIT-S06-SEQUENTIAL-LOCAL-Z-ALLOCATION-V2-SH4-V1; FULL_READ/M0 completed. PENDING_HANDOFF; technical job 49421; science registered False.
Source `21297ec19e7f5aecec16d2fdb14cc79380a1df94` / tree `26be0758ee75503161c7cafffdec8397e6cf8165`; lock SHA `a41cb76a25ab98b02c043397cd9cf4db0768e9ae312931b349008c3e9b303d18`.
Actual validation NOT_OBSERVED; science NOT_OBSERVED. cap2, no previous-task mutation. [준비·제출 보고](../../../experiment-reports/servers/server4/sequential-local-z-allocation-seq1000-2026-09-17-v2/preparation-submission-ko.md).
Resume `/data/janghj/ODE-edit/local/sequential-local-z-allocation/20260917-v2/resume-manifest.json` SHA `80f8997e4e951eccb2e20cbc3f72e663cc13c241124b77c495c44f935b12bc6e`. monitoring_active=false / automatic_resume=false / explicit_user_call only.
