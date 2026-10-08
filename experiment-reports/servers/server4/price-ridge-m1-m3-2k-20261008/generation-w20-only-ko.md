# W20-only fluency/consistency

2026-10-08 사용자 지시에 따라 이번 세 run의 fluency/consistency는 W20 전체2,000요청에서만 측정한다. R/P/N current pre/post 및 W5/10/15/20 all-seen은 변경하지 않는다. 생성 텍스트 한 번으로 두 지표를 계산하며 기존 식·샘플링·분모·seed는 유지한다.

B1–B19 generation은 실행하지 않고 NOT_SCHEDULED를 기록한다. W&B에 미측정 값을0으로 올리지 않는다. W20 current100/first500은 전체2K 생성 결과의 CPU subset이다. 정규 생성 요청-시점은6,600→2,000이며 이것을 시간 실측 배수로 주장하지 않는다. KV MB4 그대로이며 배치 크기는 확대하지 않았다.

원 actual KV 정합 확인은 첫 생성 시점인 W20 편집 후에 수행한다. 원 고정8prompt의 reference/singleton/batch 비교 및 비용은 별도 기록되며 정규2K 생성 건수와 구분한다. 별도 W0·qualification job·새 baseline은 없다.

사용자의 명시 취소/재제출 답변에 따라 기존 collector61401을 먼저, 이어61400/61399/61398을 exact identity 검산 후 취소했다. NoCP이므로 세 모델 모두 cold 재시작이고 기존 source/raw는 보존한다. baseline60917–60923 hold는 그대로다.

일정/제어 mock2개와 기존 좁은 regression5개(실제 저장행 검산 포함), 총7개 CPU 검사가 통과했다. 실행 중 원 archive는 수정하지 않았다. collector와 producer를 동일 source에 봉인한다. 신규 실제GPU/원격W&B/W20 완료는 제출과 구분한다.

등록 결과는 후속 receipt로 남긴다. 합산cap3·GPU1 독립3lane·GPU0 afterany collector를 유지하고, GPU가용성/완료를 반복 대기하지 않는다.
