# PRICE ridge M1–M3 실행 준비

**최신 2026-10-08 generation 수리:** 이전61207/61208/61209는 사용자 지시로 취소·보존했다. 수리 source `e310c38b`의 Llama61356/GPTJ61357/GPT2XL M1+M2 61358/collector61359를 release했다. 실제 KV 확인은 본 B1 이후이며 아직 NOT_OBSERVED다. [최신 수리 보고](generation-repair-ko.md)와 [현재 status](../../../../../tasks/status/price-ridge-m1-m3-2k-20261008/server4.json)가 아래 최초 제출 기록보다 우선한다. GPT2XL 통계·W0 결속은 수리됐고 M3의 저장 K 문제만 남았다.

아래는 최초 source8f39b226/attempt-r1의 보존 기록이다.

권한: `USER-SH4-PRICE-RIDGE-M1-M3-20261008-R1`, 정본 `be0917136eb46d9840aafecb960d0c9ed897c159`.
Llama/GPT-J와 CPU collector의 실제 held 등록·검사·release를 완료했다. 단발 초기 snapshot은 모두 dependency PENDING이었다. 실제 GPU 확인이나 새 2K 완료를 주장하지 않는다.

## 0단계와 구현

[저장 자료 검산](phase0-ko.md): Llama/GPTJ B1 최하층 median diag(M)는 각각 0.6214084218579502/0.735572161233262로 native lambda 15000 유지. GPT2XL은 0.34110647934119653이나 저장 K가 없어 M3 calibration은 NOT_RECORDED다. hash만으로 K를 복원하지 않았다.

M1은 canonical lookup 0일 때만 같은 forward의 prefix rewrite 5개 norm 평균을 anchors/anchor_star에 연결한다. 기존 entry helper에는 opt-in 분기만 추가했다. M2는 actual max updates 절반, GPT2 M3-only grace12 유지. M3 CPU 함수의 좁은 fixture와 분기·설정 검사 5개가 통과했다. 이는 모델 실험이나 실제 calibration PASS가 아니다. GPTJ 네 slot은 기존 tokenizer/context로 CPU 재포장한 hash가 저장 pack과 일치했다.

| 구성 | 현재 상태 | grace | lambda |
|---|---|---:|---:|
| LLAMA_REPRO | 61208, RELEASED/PENDING | 12 | 15000 |
| GPTJ_M1 | 61207, RELEASED/PENDING | 12 | 15000 |
| GPT2XL_M1_M2 | L14–L17 C0/전체 native input 미결속 | 9 | 20000 |
| GPT2XL_M1_M3 | 저장 B1 K 부재 | 12 | 미확정 |
| GPT2XL_M1_M2_M3 | 저장 B1 K 부재 | 9 | 미확정 |

Llama B1은 기존 payload/W/H hash와 실제 비교하여 다르면 transaction rollback한다. W20의 기존 RS99.80/PS91.18/NS84.94는 역사 결과이며 새 실행 결과가 아니다. 기존 writer/terminal 마지막 평가 payload/H once/20배치 controller를 재사용한다. 낮은 성능만으로 중단하지 않는다.

## W0·자원·보존

신규 W0 forward/evaluation/generation/job은 0. 저장 26,000행/model의 identity와 reducer를 CPU로 확인하고 참조만 재사용한다. 실제 runtime identity가 다르면 새 평가로 fallback하지 않고 중단한다. generation은 post-edit만 수행하며 state/RNG 비변이를 확인한다. 원모델 cold initialization은 유지한다.

기존 OURS 60107 종료 후 두 신규 GPU1 lane을 시작하는 `afterany:60107`로 등록했다. CPU collector61209는 `afterany:61207:61208`이다. 전체 cap2, 각 CPU8/59392MiB, hard60416MiB, 최대48h; CPU collector는 GPU0/CPU8/24576MiB/4h. 48h는 요청 상한이지 실측 ETA가 아니다. 기존 baseline60917–60923 hold와 취소61121/61122는 변경하지 않았다. 다른 job 변경도 0이다.

기존 메모리 계획에 generation reference 2.5GiB를 더한 host 추정은 Llama40.415GiB/GPTJ56.862GiB다. 실제 GPU/host peak는 아직 미측정이다. 두 run 보존 여유 계획은 33,126,350,848 bytes이며 준비 시 free78,756,012,032 bytes였다. 실행 batch 경계에서 기존 로그 예산과 generation reserve를 재검사한다. 부족하면 축소·삭제 대신 typed block한다.

## 후속 단계와 한계

2단계는 최종 profile과 matched identity 확정 후 BLUE/MEMIT baseline을 계획·구현한다. 동일한 기존 held baseline은 `WAITING_EXISTING_BASELINE_HOLD`이며 복제/해제하지 않는다. 3단계 FLAT/최하층+H는 선정 profile 의존으로 아직 제출하지 않았다. GPT2 자료 부재를 신규 W0나 stat 생성으로 메우지 않는다.

Owner source audit이며 별도 독립 reviewer는 사용하지 않았다. 공통 tracking/generation 구현은 읽기전용 재사용했다. 원 source/raw는 KEEP, NoCP/exact resume NOT_AVAILABLE. `NO_BROADCAST_NOT_REQUIRED`: 같은 서버의 기존 자산/원자료를 참조하며 작은 source/report만 own branch에 게시한다. main 통합은 기존 GH 절차다.

로컬 시도: `/data/janghj/ODE-edit/local/price-ridge-m1-m3-2k-20261008/attempt-r1`.
실행 source: `8f39b226f93c027880ec0b941d3f2630ee298b0d`.
Config SHA256: `107662dc6c1483b1870dd8966cedc58147d750c1bcd650c13035f89907078ad6`.
Lock SHA256: `2c1b662770bdbcf6e18c4545ecd0855e3901613f74d681d10590d5e30c67117d`.
[실제 등록 receipt](../../../../../runs/price-ridge-m1-m3-2k-20261008/submission.json)에 job/source/dependency와 원 receipt SHA를 기록했다. W&B run ID/URL/원격 검산은 실제 job startup 전이므로 NOT_OBSERVED다. 새 recurring monitor/heartbeat/자동 retry는 없다. 실행 source와 이후 보고 commit을 구분하며 own branch만 게시한다.
