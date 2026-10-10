# Server1·2 cap2 적용 완료

사용자 nonce: `USER-GH-S1-S2-CAP2-LANE-ADJUST-20261010-R1`.
정본 authority `ae2f3eb4`에서 server1/server2 cap을 각각 2로 변경했다.
server3=1/server4=2 및 기존 memory/node/QoS 제한은 그대로다.
두 담당자의 직접 OWNER_ACK와 실제 적용 영수증을 회수했다.

| 서버 | 적용 시 실제 할당 | 후속 lane 조정 | 담당 게시 |
| --- | --- | --- | --- |
| SH1 | GPU2: 62529, 62583 | 기존 62529→62530 / 62583의 두 lane 유지. scheduler 변경 없음 | d5d02084 |
| SH2 | GPU3: 62531, 62864, 62865 | 기존 RUNNING 보존, 미시작 PENDING resource edge 세 곳 추가 | 01b45a1d |

SH2의 기존 GPU3 할당을 2로 허위 표기하지 않는다. 기존 작업은 자연 종료하며,
새 job 시작이 총량 2를 넘지 않도록 실제 dependency가 제한한다.
추가된 afterany edge는 62532←62864, 62869←62864, 62866←62538이다.
이전 dependency는 모두 유지했다. 62864 종료 이후의 두 lane은 다음과 같다.

- 62532→62538→62866→62867→62868→62870→62873
- 62869→62871→62872→62874→62875→62876

SH2는 정확 PENDING 13개를 잠시 hold한 후 검산·release했다. GPU0 collector62877
coverage는 그대로이며, 임시 hold 잔존0, 취소0, 신규 제출0이다.
두 서버의 ignored local admission도 cap2로 적용했고 frozen source/config/script,
원본 checkpoint/raw/W&B 및 다른 서버 job은 보존했다.

GH는 저장된 before/after 영수증에서 원 dependency 보존·정확 세 edge 추가와
모든 가능한 시작/종료 순서 228상태를 독립 CPU 검산했다. 초기 legacy3만 예외이며
새 시작으로 cap2 초과 불가, cycle0이다. SH1 저장 graph도 width2로 검산했다.
이는 자원 DAG 검산이지 모델/GPU qualification이나 실험 완료 증거가 아니다.
README 성능 셀은 이 자원 정책 변경으로 수정하지 않았다.

## 근거

- [SH1 실제 영수증](../../../audits/servers/server1/cap2-lane-adjust-20261010/receipt.json)
- [SH2 전후 영수증](../../../audits/servers/server2/cap2-lane-adjust-20261010/after.json)
- [GH 직접 전달/ACK](../../../audits/global/cap2-lane-adjust-20261010/dispatch.json)

신규 recurring monitor/자동 retry/장기 GPU 완료 대기 없음.
