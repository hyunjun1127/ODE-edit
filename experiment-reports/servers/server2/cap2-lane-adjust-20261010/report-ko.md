# server2 cap2 실제 적용

Nonce: USER-GH-S1-S2-CAP2-LANE-ADJUST-20261010-R1. Accepted turn: 01a12500-fa47-7fc0-85f7-1204a696bdf6.

정본 ae2f3eb4 및 envelope full SHA 일치 확인. server2/owner janghj의 전체 프로젝트 queue와 실제 Command/WorkDir/원 제출 receipt/source/config/script를 대조했다. 기존 RUNNING 62531, 62864, 62865의 GPU 3개는 grandfathered legacy 초과 할당으로 보존한다. 현재 할당을 2라고 표시하지 않는다.

PENDING 13개를 후속부터 일시 hold한 뒤 아래 세 resource-only afterany edge를 기존 dependency에 합집합으로 추가했다. 전량 재검산 후 release했다.

| Job | 이전 dependency | 추가 dependency |
| --- | --- | --- |
| 62532 | afterany:62531 | afterany:62864 |
| 62869 | afterany:62865 | afterany:62864 |
| 62866 | afterany:62864 | afterany:62538 |

62864를 전환 시 종료되어야 하는 기존 root로 두고 이후 두 lane은 다음과 같다. 전체 job 완료를 기다리는 barrier는 없다.

- 62532 → 62538 → 62866 → 62867 → 62868 → 62870 → 62873
- 62869 → 62871 → 62872 → 62874 → 62875 → 62876

가능한 시작/종료 순서 228상태 CPU 전수검산: cycle0, 기존 최초 allocation3만 예외, 이후 새 시작으로 cap2 초과 불가. 기존 세 RUNNING root까지 포함한 정적 graph 폭은 여전히3이며, 이를 future 두 lane과 혼동하지 않는다. GPU0 collector62877 원 dependency 그대로 유지. 임시 hold잔존0, 취소0, 신규submit0. 별도 reviewer/GPU 검증 없음.

root 및 현 own prep 5개 checkout의 ignored server2 cap행을2로 적용했다. node server2, memory ceiling60416MiB, patterns 및 다른 서버행 불변. frozen registration source/config/launcher/archive 및 raw/CP 변경0. 파일 source/config/script SHA 전후 대조 PASS. 다른 서버/owner 작업 변경0.

원자료: 같은 task audit의 before.json/held.json/after.json/proof.json/apply.py/local-caps.json. 단발 전후 snapshot이며 장기 monitor/retry 없음. README는 GH sole 통합. NO_BROADCAST_NOT_REQUIRED: 소형 자원조정 receipt만 Git.
