# Alpha JV D/S CPU 준비 및 비용 보고 — GPU 결과 아님

상태: CPU_PREPARATION_COMPLETE / GPU_HOUR_BUDGET_UNASSIGNED. scientific_promotion=false.
이 패키지는 실행 결과나 PRE-GPU PASS가 아니다. 신규 model/GPU/Slurm/과학 endpoint 모두 0.

## 범위와 담당

SH1은 공통 config/fixture/trajectory 통합, D owner는 artifact/normalization/requestwise,
S owner는 sample/grid/driver, 독립 resource reviewer는 cap/budget 및 공통 경로 검사를 담당했다.
신규 소스는 `project/run_scripts/alpha_jv_llama_diagnosis/`로 제한하며 기존 runtime 77358b1546d1baf83b3e251afcce663b08d7bfd7의
세 재사용 패키지 72개 tracked member를 byte 비교했다. 수정 0.
main은 GH 검토 후에만 통합한다. 전용 branch push는 CPU 준비물만 포함한다.

## FULL_READ / CPU 검증

계약 SHA `98ee832e597b1f79c79ca48c62838af8f49f5306c08b3480b990d494f42f15f5` 및 operational addendum을 포함한 전체 읽기/재해시는
`full-read-inputs.json`에 기록했다. focused CPU 67/67 PASS,
py_compile/session/diff PASS. Actual GPU fidelity는 아직 NOT_RUN이다.
기존 source N0/.1/T2/N4와 새 공통 루프는 CPU fixture에서 node 계수/q/V/E/endpoint가
일치했다. Prefix observer 및 raw/terminal sink의 RNG 개입은 fail-close하고 W0/RNG를 복원한다.
이 CPU 일치는 모델 fidelity 또는 과학적 효능의 증명이 아니다.

## D availability

`HISTORICAL_EXACT_STATE_UNAVAILABLE_ON_SERVER1`. Publication은 W1/W5/W10만 기재하고 W9/fixed-z tensor/journal exact replay는
검증되지 않았다. 명시된 W5/context 로컬 경로는 부재이며 Server2 연결/복구는 하지 않았다.
W10을 W9로 대체하거나 W0부터 재실행하지 않는다. D의 exact-state blocker는 S dependency가 아니다.
Raw가 없는 원인 attribution, NRMS intervention, old900 변화는 결과를 만들지 않고 NOT_RUN으로 둔다.

## S outcome-independent seal

양 모델 동일 DEV100, 별도 disjoint AUDIT300을 Git-only exclusion inventory와 hash-rank로
봉인했다. 모델 결과/target 길이로 replacement하지 않았다. DEV RS/PS/NS 분모는 실제 입력의
100/200/1000이며 임의 locality 중복/제거 0이다. ID·order·hash만 공개하고 raw prompt는 Git 제외.
다섯 λ `.01,.03162277660168379,.1,.31622776601683794,1`은 모두 actual 예정이며
양 모델 각 7 paths/34 nodes/170 main JVP, 9 JV endpoints + Official + entry다.
아직 관측 endpoint는 0/18이며 표의 NOT_RUN은 과학 실패 또는 endpoint로 집계하지 않는다.
T4/h.5의 T1/T2 prefix는 중간 평가만 수행하며 append/capture 0, 이후 trajectory 불변을 검사한다.
감사 표본 300개는 예약만 했으며 실행 권한/후속 선택 규칙을 자동으로 발명하지 않는다.

## 비용과 첫 wave

| 범위 | 과거 기록 기반 GPUh proxy | 제외/한계 |
|---|---:|---|
| S Llama | 1.803274 | 추가 FD·복원·reserve 미계측 |
| S Qwen | 1.807715 | 위와 동일 |
| S 합계 | 3.610990 | 모델 로드/z/기존 writer/eval 성분 포함, 상한 아님 |
| D 기본 current100 | 0.662568 | old900·capture·controls 제외 |
| 선택적 W5→W9 replay | 1.749726 | 파일 부재, 예약/실행 0 |

비용 산식/입력 SHA/미계측 항목은 `cost-estimate.json`에 전부 있다. 과거 host/state의 성분별
min/max는 신뢰구간·실행 상한이 아니다. 사용자 GPU-hour cap은 null이며 과거 8h/48h를 상속하지 않는다.
첫 GPU wave는 새 budget·source/asset·정상 신호 fidelity 결속 후 양 모델 S_DEV 비교를 cap2 내에서
시작한다. 먼저 양 모델 BASE/time/coarse λ, 이후 양 모델 intermediate λ actual을 수행할 계획이다.
기존 job은 그대로 두며 RUNNING/CONFIGURING/미해제 COMPLETING 포함 재계수, scheduler 실패는 HOLD다.
task 메모리 182272M/1GPU, local 상한183296M. 자원 조회 성공은 예약이 아니다.

## 남은 경계

GPU-hour authority 및 exact launch/source/asset/fidelity 결속은 아직 없다. production launcher는
이번 CPU 패키지에서 봉인하지 않았다. D 실제 historical 재생/old900 endpoint, S actual 표와
PNG/효능/원인 판단은 NOT_RUN이며 추정하지 않는다. source/CPU/GPU 검증을 구분해 인계한다.
Server4 live source/process/output, 기존 원본/원격 raw 변경·접근 0. Git은 code/tests/이 raw-free 준비물만 포함한다.
