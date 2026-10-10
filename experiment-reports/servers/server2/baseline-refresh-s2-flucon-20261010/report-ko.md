# 완료 결과 갱신 및 미측정 CF Flu/Con 평가

지시 `USER-GH-S1-S2-BASELINE-REFRESH-S2-FLUCON-20261010-R1`, 접수 turn
`01a124db-95bd-7231-981d-429bbc98305c`. README 편집은 GH만 수행한다.

## 결과 검산

단발 own accounting 및 원 raw/terminal SHA 검산에서 완료 수치 적격 23행,
불일치 0. `table-rows.json/csv`에 원 source/config/순서/분모를 결속했다.
zsRE는 저장 predicted/target token correctness의 요청별 평균 후 요청 간 평균이다.
Loc은 loc_ans이며 W0 agreement/token micro가 아니다. 기존 CPU query proof는
모델 GPU numerical parity를 의미하지 않는다.

새 완료 Qwen CF AlphaEdit62075: E99.05/G97.55/Loc63.36/Score83.03373678537004.
zsRE AlphaEdit62083: E85.05212842712842/G78.18345598845599/Loc30.792555000914508.
zsRE MEMIT_FE62085: E0/G0/Loc0.033333333333333326. 이 0은 실제 저장 correctness
재집계 결과이며 결측 placeholder가 아니다. 각2000 요청, zsRE token 분모6691/6691/11476.
author62531/62532 및 B9 재개62538은 이 snapshot에서 미완료로 유지한다.

## 평가 구현과 보존

`official.runners.server1.flucon_eval`의 공통 생성/채점/실시간 tracking/collector를
그대로 호출한다. SH2 adapter는 server/task identity 및 기존 SH2 checkpoint의
selected-weight-only restore만 연결한다. 공식 payload는 원 identity로 검증한다.
역사 payload는 원 metadata/source/lock/selected tensor hash 및 동일2000 순서를
검증하는 별도 adapter로 복원하며 official checkpoint로 위장하지 않는다.
과거 runtime과 현재 평가 runtime은 별도 기록하고 bitwise parity를 주장하지 않는다.

새 fit/edit/W0/zsRE generation/qualification은 없다. 최종 CF2K 한 번 생성하여
Flu/Con을 함께 산출한다. 원 raw/W&B는 변경하지 않으며 표시만 raw 평균×100,
decimal half-up2다. CP/H/context/RNG는 원본에 보존하며 평가에 필요한 selected W만 복원한다.
공통 generation/logger 코드 수정·복제0. root dirty와 frozen job은 유지한다.

CPU adapter/기존 restore fixture15 PASS, official source166 SHA 및 external imports0.
별도 GPU qualification은 NOT_RUN_USER_DISABLED. CPU 및 held 검사를 실제 GPU 완료나
W&B 원격 전달 PASS로 표현하지 않는다.

## 중복 제거 및 제출 단계

Qwen FT61898/BLUE61962 CF generation은 기존 완료 raw를 유지한다.
SH1 history 평가62581/62582 완료,62583 기등록 실행 중이므로 중복 제출하지 않는다.
SH1 native Llama FT/SPHERE/FE62259/62260/62261도 기존 완료 결과를 사용한다.
W20 없는 author CP는 제출 대상으로 만들지 않는다.

현재 server2 cap3. 기존 FE author→zsRE→SPHERE-resume 자원 lane은 변경하지 않는다.
미평가 CP는 exact SHA/bytes/20commit/2000/source/config 검산 후 새 immutable eval-only
등록 경로를 사용한다. GPU1/CPU6/59392MiB/48h, GPU0 collector CPU6/24576MiB/4h.
남은 두 lane에 필요한 afterany edge만 추가한다. 전체 held inspection 후 release한다.
실제 IDs와 상태는 후속 `submission.json` 영수증에 기록한다. 이 문단 자체는 제출 완료가 아니다.

## 실제 등록/release 완료

실행 source `417a12e7`(main 게시), 새 attempt
`/mnt/raid5/janghj/ODE-edit/local/baseline-refresh-s2-flucon-20261010/registration-r1`.
아래 13 GPU 평가와 GPU0 collector62877을 전량 held 등록·exact inspection 후 release했다.
초기 단발 snapshot은 전부 PENDING이다. 사전 GPU qualification/결과 완료/원격 W&B PASS는 아니다.

| 모델 | 방법 / 원 편집 job | 새 평가 job | afterany |
| --- | --- | --- | --- |
| GPTJ | FT61650 | 62864 | 없음 |
| GPTJ | MEMIT61725 | 62865 | 없음 |
| GPTJ | AlphaEdit61778 | 62866 | 62864 |
| GPTJ | BLUE61779 | 62867 | 62866 |
| GPTJ | FE61780 | 62868 | 62867 |
| GPTJ | SPHERE61781 | 62869 | 62865 |
| Qwen | MEMIT62073 | 62870 | 62868 |
| Qwen | AlphaEdit62075 | 62871 | 62869 |
| Qwen | FE62077 | 62872 | 62871 |
| Qwen | SPHERE62079 | 62873 | 62870 |
| Llama historical | MEMIT42658 | 62874 | 62872 |
| Llama historical | AlphaEdit42657 | 62875 | 62874 |
| Llama historical | BLUE39283_1 | 62876 | 62875 |

collector62877은 새 평가13개 전체 afterany. 각 실제 name/config fullSHA/원 CP fullSHA/출력경로는
own `submission.json`에 수록했다. 기존62531→62532→62538 lane과 함께 DAG폭3이며
기존 job mutation0. 현재 가용169,822,842,880B 관측, 제출 시 fresh 용량과 64GiB reserve+
13×2GiB 관측 파일 여유를 재검산했다. CP/모델 대형복사 없음.

추가 CPU actual payload 검산: GPTJ6/Qwen4 모두 schema/batch20/원identity/FP32 finite PASS.
역사 Llama3도 원 payload fullSHA, 원 execution.lock/local-source.tar, 모델·토크나이저 fullSHA,
selected tensor dtype/shape-envelope SHA 및 ordered2000을 검산했다. 과거 4.44.2와 현재
generation 평가 runtime은 분리 기록한다. 이관된 CP의 편집 출처를 최신 official로 바꾸지 않는다.
독립 reviewer는 사용하지 않았으며 owner CPU audit이다. 시작 전 W&B는
`NOT_OBSERVED_BEFORE_STARTUP`; 실시간 scalar logging은 실행 job에서 수행한다.

대형 전송·삭제0, `NO_BROADCAST_NOT_REQUIRED`. 모든 모델/CP/관측 raw는 기존 local에
남으며 Git에는 소형 source/집계/영수증만 게시한다. 장기 monitor/자동 retry 없음.
