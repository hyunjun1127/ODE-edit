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

대형 전송·삭제0, `NO_BROADCAST_NOT_REQUIRED`. 모든 모델/CP/관측 raw는 기존 local에
남으며 Git에는 소형 source/집계/영수증만 게시한다. 장기 monitor/자동 retry 없음.
