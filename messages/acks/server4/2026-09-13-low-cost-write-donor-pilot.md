# SH4 low-cost write/donor pilot FULL_READ / P0

Instruction `ODEEDIT-S06-LOW-COST-WRITE-DONOR-PILOT-SH4-V1`.
Source main `0e158bd8907348c4f2d065df4a4fc5670e2fabad`, tree `5856d61772daaca9e7a1553fd0015eea96acff3d`.
원문·설계·cells·schema v2·PROTOCOL·실행 envelope FULL_READ. 게시 문서는 첨부 원본 bytes와 구분한다. Server4에서 같은-host fresh native와 후보를 실행한다.

P0 CPU: S2 L4 W50/M50 한 파일 수신 full SHA/size 및 selected weight/history/entry5000/RNG schema 검증. S1 완료 Historical128/Wiki128 작은 봉인파일 세 개 수신. MMLU 기존 데이터 rows10:110을 재사용해 outcome-independent32/68 분리. 고정10k 순서 유지. 과거 live E01·다른 task 조회/변경0.

Core: N4/S875/S75/FULL8/RES8/REFIT4. 공유 첫 N4 request-z100과 fresh second fit300, M8 공통 We 과거5000 events 재인코딩1회. 성능에 따른 탈락·rescue·후속 자동제출0. 초기 실제 RES8 partial-fit/restore gate 뒤 사용자 recall까지 agent 중지.

운영 예상(실측 아님): GPU2–8시간, 추가 저장20–40GiB; 사용자 GPUh cap은 null. 단일1GPU/8CPU/60416M/12h 프로그램으로 준비와 core를 수행한다. Scheduler max30일보다 작은 요청이며 과거48GPUh를 상속하지 않았다. cap2 내 admission은 제출 직전 재확인한다.

Raw/control: `/data/janghj/ODE-edit/local/low-cost-write-donor-pilot/20260913-v1/`. 원본 BLUE·공유 source 수정0. 선택 input transfer 외 `NO_BROADCAST_NOT_REQUIRED`. P0 CPU는 실제 GPU gate PASS가 아니다.
