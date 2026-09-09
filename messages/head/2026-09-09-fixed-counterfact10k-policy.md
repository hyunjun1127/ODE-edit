# 고정 CounterFact 10k 자산 배포 / baseline 3 job 지시

사용자 승인으로 GH가 고정 데이터셋 추출 코드와 PROTOCOL 정책을 직접 구현했다.
원본 전체 데이터와 BLUE 실제 sample.lock으로 10,000개 원래 레코드를 동일 순서로 추출했다.
Server1·2·4의 `local/datasets/counterfact-fixed-10k-v1/`에 별도 배포했으며,
각 서버에서 데이터 SHA, 10,000개 개별 record hash/순서/중복 여부,
평가 inventory 10,000/20,000/100,000을 검증했다.

전용 데이터 SHA256은 `3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1`,
BLUE sample root는 `5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729`다.
원본 전체 데이터셋과 과거 실행 locks는 수정하지 않았다. raw 레코드는 Git에 넣지 않았다.

- 새 CounterFact 실험은 이 자산만 사용하고, N개 실험은 앞 N개를 사용한다.
- 1,000개 prefix는 기존 JVP sample 순서와 실제 대조하여 일치했다.
- 3,000개 prefix도 loader에서 그대로 잘리는지 검증했다.
- 3개 focused tests: 기존 파일 덮어쓰기 차단, prefix 범위/순서, record 변조/순서 오류 검출.
- 10k 초과와 독립 audit가 필요하면 별도 사용자 지시를 받는다.

SH4에는 Llama PRE_EDIT 평가 1개, base AlphaEdit 및 base MEMIT B100x100 각각 1개를
같은 자산으로 cap2 pending 제출하도록 지시했다. base는 BLUE 저장소의 실제 `blue=False`
원본 경로 및 non-blue config이며, BLUE config와 다른 L2 등의 값을 명시한다.
원본 baseline의 설정을 BLUE에 억지로 맞추지 않는다. 초기 submission 확인 후 모니터링은 중단한다.

SH1·2·4는 공통 정책을 수신 후 향후 runner의 입력을 이 자산으로 결속한다.
이미 실행 중인 job/source/config는 수정하지 않는다. 재추출·셔플·결과 기반 교체를 하지 않는다.
