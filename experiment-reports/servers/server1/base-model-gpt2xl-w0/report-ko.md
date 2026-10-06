# GPT2-XL cold W0-only 평가

fresh cold W0-only GPU와 CPUcollector를 실제 등록·held 검사·release했다. 이 문서는 제출 인계이며 현재 모델 성능, 완료 또는 online 검증을 주장하지 않는다.

first2000 cold W0의 R2000/P4000/N20000을 새로 관측한다. 편집·fit·solve·H append·C0/P 준비·checkpoint 저장은 없다. 기존 ours/native baseline은 그대로 유지한다. W0-only 한 GPU에 한해 명시 사용자 cap 예외를 적용하되 원 method cap2 및 물리 scheduler 제약은 유지한다.

준비 및 평가 정의는 [제출 전 점검](../../../../audits/servers/server1/base-model-gpt2xl-w0/preflight-ko.md)에 기록했다. CPU 입력/스칼라 검사는 actual GPU 모델 검증과 구분한다. 실제 W&B는 신규 job의 startup/finish에 확인하며 raw local KEEP/noCP/원 소스 보존을 유지한다.

| 역할 | 실제 job ID | dependency | 등록 직후 단발 snapshot |
|---|---:|---|---|
| W0_BASE_MODEL GPU | 60156 | 없음 | PENDING, reason (None) |
| 독립 CPUcollector | 60157 | afterany:60156 | PENDING, reason (None) |

source `de31540487643b3c4187ffad5c09554d18c69e94`, tree `60fcc0d7ac96cba16639a1dc809ceb960aee659f`, config SHA `6700bdcaf5969abbc26cfd1805d1beb4e0453b9779fc47b98d0e0db42fed7135`, lock SHA `e943a596075a3e7823f98cc73f313d33f506cfc9cd0673b3a17b68477b915e45`다. 36개 exact source member를 freeze했다. [실제 제출 receipt](../../../../audits/servers/server1/base-model-gpt2xl-w0/submission-receipt.json)는 원 local lock/config/held/release를 SHA/size로 연결한다.

GPU1/CPU8/65536MiB/4h, CPUcollector GPU0/CPU8/24576MiB/2h, devbox/gpu/lab_gpu_s1/exportNONE/Requeue0를 held 상태에서 검사했다. source/archive/config/runtime/input/model 및 launcher bytes를 재확인한 뒤 collector→GPU 순서로 release했다. 물리 자원 부족시 scheduler PENDING이며 기존 arm 종료 dependency는 넣지 않았다. USER_SCOPED_CAP_EXCEPTION receipt를 남기고 global/local cap 파일 및 기존 job을 변경하지 않았다. 4h는 ETA가 아니다.

owner CPU10개 실패0/오류0, 독립 reviewer는 source와 이전 CPU8개를 확인했다. 실제 scheduler/모델/GPU/online을 reviewer가 실행한 것은 아니다. 현재 INITIAL_NOT_OBSERVED / W&B NOT_YET_OBSERVED이며 실제 결과·성공/완료를 추정하지 않는다. 제출 snapshot 뒤 agent polling/heartbeat/automatic retry를 중단했다. 등록 runner와 afterany collector는 자연 진행한다.

자동 collector는 저장 row의 R/P new<true, N true<new, tie=failure와 TF desired=N true를 독립 집계한다. 기존 W0 raw는 원 row/token/source/runtime identity를 확인한 CPU paired 대비로만 사용하고 신규 actual 측정을 대체하지 않는다. parent allocation은 collector에서 NOT_QUERIED이며 program wall/RSS/VRAM을 가능한 저장 counter로 분리한다. 전체 cold W0 모델 byte 비변이와 교차 hardware 수치적 동등성은 주장하지 않는다.

NoCP/exact_resume=NOT_AVAILABLE, no edited weight/H/optimizer 저장, NO_BROADCAST_NOT_REQUIRED다. raw/scalar per-case는 task local에 남고 Git/W&B에는 허용 소형 코드/identity/scalar만 기록한다. 원 root dirty와 다른 source/raw는 보존했다.
