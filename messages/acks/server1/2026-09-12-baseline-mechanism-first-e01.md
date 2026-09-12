# SH1 E0/E1 신규 독립 task M0

- instruction_id: ODEEDIT-S06-BASELINE-MECHANISM-FIRST-E01-SH1-V1
- nonce: ODEEDIT-GH-SH1-BASELINE-MECHANISM-E01-20260912-R1
- base HEAD/tree: f8d78c88db24cd5580c844b6a3621e48f9bc3734 / ed4ed74b2a45cf653925346bcd43ea0aa7c041ce
- branch: codex/server1-baseline-mechanism-first-e01-v1
- worktree: /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-baseline-mechanism-first-e01-v1
- 설계: regular non-symlink, 원본 mode0664, 53160 bytes / 582 LF, SHA256 e4dc0b1fc0666775deb6e43cbdf18269c8e8e6d41d70c192e2bd4a2661a6ccbe. 1..582 FULL_READ_PASS. Git 사본과 원본 결속.
- 개인 실행 지시 162행, 역할계획, 선택수신 승인, PROTOCOL 1269행(말단 1210–1269 추가 확인), connection-inventory 109행 FULL_READ_PASS. 필수 실제 source/reference 읽기는 진행 중이며 단순 존재를 전체 검증으로 간주하지 않는다.
- 최초 자원 관측: 기존 AOS45633 1GPU RUNNING, 신규 제출0, cap2, 요청 상한182272MiB, 새 GPU-hour cap=null. 기존 job 상태 관측은 신규 admission용이며 해당 task 재개/변경0.
- shared root의 dirty/untracked 파일 보존. 별도 worktree의 Git identity는 SH1/server-head로 설정.
- evidence/case, fixture/native, instrumentation의 소유 파일을 분리해 CPU 병렬 준비. 부모는 contracts/CLI/수신/통합/제출, 이후 read-only red 검사를 분리.
- E0/E1만 실행. E2–E5/새10k/controller 실행0. 실제 초기 gate 뒤 MONITORING_PAUSED_AWAITING_USER. GPU 준비 PASS/완료 결과는 아직 주장하지 않는다.
- S2 CP/S4 companion은 선택 allowlist·소유 확인 후 SH1 단일 writer로 수신. 기존 S1 exact 사본 우선. 사실 package는 SH1, 최종 해석은 GH 소유.
