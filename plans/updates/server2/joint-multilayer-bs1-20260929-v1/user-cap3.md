# 최신 사용자 운영 override — cap3

Parent/nonce `ODEEDIT-GH-SH4-SH2-JOINT-BS1-MIGRATION-20260929-R1`.

현재 SH2 turn에서 사용자가 **“CAP 3으로 해”**라고 명시했다. 동일 task의 Server2 project/task cap을3으로 갱신한다. 기존 array55116의 throttle만2→3으로 변경했고 실제 scheduler `ArrayTaskThrottle=3`을 확인했다. 원9경로·BS1×100·save25·각1GPU/8CPU/60416MiB·exportNONE/Requeue0는 그대로다. 새 job/arm/중복실행0, 다른 job변경0.

실행 source `2a7762a4`와 configuration/execution.lock은 제출 당시 cap2를 보존한다. 최신 운영cap은 local `attempt-s2-r1/user-cap3.json`으로 결속한다. 이력의 cap2를 소급 수정하지 않는다. 세번째slot은 실제 scheduler 자원이 확보될 때만 사용할 수 있으며 GPU 공유·선점은 하지 않는다.

GH registered peer-direct 수신 ACK: local `cap3-peer.json`. 대표 joint 첫 step→두번째entry 또는 승인된 실제 자원대기 인계 후 pause 경계는 유지한다. 다른 task STOP/PAUSE 및 전역 서버 정책 파일은 이 task에서 수정하지 않는다.
