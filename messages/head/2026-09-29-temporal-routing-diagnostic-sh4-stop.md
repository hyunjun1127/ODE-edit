# GH → SH4: 단층 temporal-routing task 즉시 중단

Instruction / nonce: `ODEEDIT-GH-SH4-TEMPORAL-ROUTING-STOP-20260929-R1`.
Supersedes execution authority: `ODEEDIT-GH-SH4-TEMPORAL-ROUTING-DIAGNOSTIC-20260929-R1`.
Target server4 / session `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`.

사용자 원문: “실험 설계 중단시켜. 단층 EDIT이 아닌 여러 층 EDIT으로 설계를 해야하는데 잘못설계했다.”

## 최신 권한

현재 단층 고정-layer 설계의 구현·준비·실험·신규 제출/release·입력 전송을 즉시 중단한다.
15branch/1500fits/30snapshot 실행 권한과 누락 CP pull 승인은 이 task에서 철회한다.
초기 gate 도달, GH의 추가 ACK 또는 새 main fetch를 중지 선행조건으로 삼지 않는다.
본 문서 게시 전 전달한 같은 nonce의 direct override도 즉시 유효하다.

본 task에 등록된 job이 있으면 해당 submission receipt/owner/source/argv로 정확히
확인한 본 task GPU job/array 및 dependent collector만 취소하고 terminal cancellation을
한정 확인한다. 다른 task/job은 조회·변경하지 않는다. 미제출이면 취소할 job이 없음을 기록한다.
본 task에 속한 CPU worker/전송도 중단하되 기존 공유 자산은 보존한다.

원 parent CP/model/기존 및 partial source/raw/로그/비용/실패 기록은 KEEP, 삭제0.
새 snapshot 저장 의무를 충족하려고 계산을 계속하지 않는다.
기존 design/contract/CSV/dispatch SHA는 역사 자료로 보존하고 실패한 과학 결과로 바꾸지 않는다.
이는 사용자 설계 방향 정정에 따른 실행 철회이며 기술·성능 실패가 아니다.

다층 EDIT을 지원하는 재설계가 필요하다. 그러나 본 중단 지시는 새 다층 layer 조합,
writer/target/residual/history 규칙·실험 예산을 확정하거나 실행하는 권한이 아니다.
재설계·재실행은 새 사용자 지시 전 자동으로 시작하지 않는다.

nonce ACK, 실제 제출·취소 ID 또는 job0, worker/전송 중지 및 자료 보존 상태를 회신하고
`STOPPED_USER_DESIGN_WITHDRAWAL / WAITING_USER`로 종료한다.
monitoring_active=false, automatic_resume=false; polling/자동 recall/후속 제출0.
GH는 취소 확인에 필요한 한정 응답만 받으며 실험 감시를 재개하지 않는다.
