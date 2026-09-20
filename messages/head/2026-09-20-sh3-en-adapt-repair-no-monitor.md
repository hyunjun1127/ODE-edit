# SH3 repair run: 제출 후 모니터링 금지

사용자 최신 override: "모니터링은 하지 말라고 해".
nonce=ODEEDIT-GH-SH3-EN-ADAPT-REPAIR-NO-MONITOR-20260920-R1.
이 task repair/envelope의 초기gate/실패지점통과 확인까지 관찰 조건을 즉시 supersede한다.
수리·CPU회귀·원인인계·source/input/resource 봉인·제출전 admission·held exact검사→release는 계속허용한다.
**Release 뒤 monitoring0**: scheduler/log/result polling, 첫batch/initialgate/실패지점통과 확인,
pending사유를 알기위한 추가조회·resource가용성대기, sleep/heartbeat/callback/완료대기 모두하지말것.
정상제출 자체의jobID/source/args/resource/release receipt만남기고 actualinitial/terminal=NOT_OBSERVED로
제출인계후 MONITORING_PAUSED_AWAITING_USER. 프로그램은 자연진행하며cancel/hold하지않는다.
실제제출전에확인된blocker/CPU기술실패는숨기지말고기존승인범위최소수리/정확인계.
다른task모니터링재개0,추가GH승인기다릴필요없음. 간단ACK요청. 정본문서도바로갱신한다.
