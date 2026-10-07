# server1 공통 GPU cap2 적용

최신 사용자 nonce `USER-GH-ALL-SH-GPU-CAP2-20261007-SERVER1` 수락·적용했습니다.
정본 SHA와 server1 로컬 cap2를 확인했으며 root의 기존 행은 변경할 필요가 없었습니다.
전용 작업경계의 ignored 설정도 같은 server1 cap2/node/memory로 결속했습니다.

단발 owner resource metadata 조회에서 GPU job 0개, 실제 할당 0GPU, admitted DAG 폭0이었습니다.
기존 job 취소/수리/재시작 및 pending scheduling 변경은 없었습니다.
W0-only 과거 추가GPU 예외는 새 admission에서 사용하지 않습니다.
새 과학실험 허가는 이 정책이 아니라 각각의 원 task 권한에 따릅니다.

큰 자료 전송 없이 `NO_BROADCAST_NOT_REQUIRED`; 반복 모니터링 없음.

이후 별도 승인된 CAKE/AlphaEdit-BLUE60739/60740 및 후속 PRUNE/RECT60757/60758을 등록했습니다.
등록 후 단발 metadata에서 실제 server1 할당은 CAKE의 1GPU이며 admitted DAG 최대폭은 2입니다.
후속 두 run은 모두 afterany60739:60740로 연결돼 추가 동시폭을 만들지 않습니다.
기존 pending job dependency/hold/throttle 조정은 필요 없었고 기존 job mutation은 없습니다.
