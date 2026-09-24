# 기록 전용 수치 재현 정책 — 제출 전 owner 검사

권한: `ODEEDIT-GH-SH4-HISTORICAL-TIMEAXIS-RECORD-ONLY-20260925-R1`.
정본 전문 SHA `9d2cebee7da6d70ceb69f55aca4a15ef81acfbb653b9027af0f76989acfc6240`을 읽었다.
기존 PROTOCOL/설계/실행 입력은 이전 exact 읽기·CPU 검산에 연결하며 원본은 변경하지 않았다.

## 검사 사실

- 좁은 CPU 회귀 및 기존 state/reducer 회귀 32/32 PASS. 실제 GPU 수치 검증이 아니다.
- 기존 오류 0.0005242824554443359, 큰 유한 NLL 차이, boundary 밖 flip을 경고로 전달한다.
- 반복/MB16↔MB1/과거 평가/복원/상태 sentinel 모두 동일 기록 전용 비교 경로다.
- NLL 0.00025 / margin 0.0005 기준값은 불변. 초과값·초과량·문항 위치와 flip을 보존한다.
- 실제 신규 진단 raw는 numerical 판정 전에 atomic 저장한다. NaN/Inf는 명시 태그로 근거를 남긴 후 여전히 실패한다.
- source/input/token/order/cardinality/finite/whole-five-weight/정확한 restore byte는 차단 검사로 유지한다.
- peer/collector는 `PASS`의 의미를 구조적 완료로 확인한다. 경고는 전달되고 `numerical_certification=NOT_ESTABLISHED`는 유지된다.
- collector는 경고가 있어도 완결된 두 family를 처리하며 실제 구조적 실패는 계속 차단한다.
- 원 evaluator/model/dtype/kernel/MB16/FP64 제거→FP32 materialization/패널/수식은 변경하지 않았다.
- ACTUAL+force_removal 선분기와 report/inventory 뒤 atomic terminal 수리는 유지한다.
- source 변경 범위는 해당 task namespace뿐이다. 원 frozen 실행/source/raw/실패/T0/24CP는 보존한다.
- 신규 GPU 진단·native fitting·history append·checkpoint 저장 0.
- launch의 release 이후 scheduler snapshot을 제거했다. release 명령 성공만 제출 receipt로 남긴다.

CPU receipt: `/data/janghj/ODE-edit/local/historical-update-timeaxis/20260924-v1/record-only-20260925-r1/cpu-tests.json`, SHA `4f09b24747eb48d03b5207a911830a8a4a17ed1a6c1c9d483191a06c63068b09`.
local CPU log SHA `b792959d5550e44a3963cbb68d7dc52bfe2dde53101207aa6bb890da96e97df5`; 로그 Git 미등록.

## 재사용·자원 경계

9개 endpoint GPU raw + Alpha의 12개 diagonal receipt는 SHA/입력/weight binding 검산 후 explicit reuse bridge로만 연결한다.
Alpha diagonal 원 raw는 미저장으로 독립 raw 재검산을 주장하지 않는다. 기존 PASS와 실패 기록은 덮지 않는다.
MEMIT W100과 MEMIT diagonal 및 본 score task는 새 source의 실행 대상이다.
기존 1569 parent GPU초는 신규 할당과 별도다.

제출 전 관측: server4 본인 queue 비어 있음, 디스크 가용 74,469,453,824 B, inode 여유 225,234,556.
기존 20 GiB reserve 유지. 각 GPU job 1GPU/8CPU/60416MiB, 두 family 합 cap2. CPU collector 8CPU/24576MiB.
실제 제출 직전 admission을 다시 봉인한다. 계획 wall7일은 partition 상한30일 이내이며 실측 완료시간이 아니다.
기존 실제 host peak 약33,835MiB는 이전 시도 실측이고 새 peak는 미관측이다.

별도 red agent는 사용하지 않았다. owner source 감사와 CPU reducer/회귀검사다.
전용 child session boundary PASS; shared Git identity/config는 변경하지 않았다.
NO_BROADCAST_NOT_REQUIRED: 같은 host 입력과 local-only score를 사용한다.

이 문서 작성 시 `IMPLEMENTED_NOT_SUBMITTED`; 실제 제출 인계는 별도 source/lock/job receipt로 남긴다.
