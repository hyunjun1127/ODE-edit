# Server2 완료 결과 및 Qwen 이전 준비

- 승인: USER-GH-S4-S2-QWEN-BASELINES-MIGRATION-RESULTS-20261009-R1.
- GPT-J CF6/zsRE6 전체: 원 W20 raw SHA/bytes, 20개 native commit 영수증, 순서가 있는 2,000 case, 원 source/config/stream 결속 및 CPU reducer 일치 확인. 모델/CP load와 새 forward 0.
- CF E/G/S는 request-macro strict NLL preference, Score는 harmonic. zsRE S는 W0 prediction agreement이며 loc_ans와 별도(JSON에 보존).
- CF 실제 prompt-pair 분모 R2000/P4000/N20000; zsRE teacher-forced decision 분모 E5557/G5557/S9694/loc_ans9694. 점수는 token-micro가 아닌 request-macro.
- CF Flu/Con은 DEFERRED, zsRE NOT_APPLICABLE. 결측을 0으로 만들지 않음. 모든 checkpoint future consumer pending KEEP. 본 점검은 저장된 raw에 대한 CPU 검산이며 온라인 전송 재검증이 아님.
- W0_CF61723의 과거 logging rejection 이력은 보존. 정상 main 또는 raw를 수정/재실행하지 않음.

| dataset | method | job | E | G | S | Score |
|---|---|---:|---:|---:|---:|---:|
| cf | FT | 61650 | 88.7500 | 66.7500 | 40.6100 | 58.9700 |
| cf | MEMIT | 61725 | 97.9000 | 95.3750 | 60.5750 | 80.6281 |
| zsre | FT | 61726 | 23.1463 | 17.9478 | 0.5184 | 1.4793 |
| zsre | MEMIT | 61728 | 93.6070 | 88.9547 | 55.9245 | 75.3657 |
| zsre | ALPHAEDIT | 61730 | 99.7887 | 96.5511 | 84.7581 | 93.2353 |
| zsre | ALPHAEDIT_BLUE | 61732 | 99.8483 | 95.8058 | 86.4919 | 93.7068 |
| zsre | MEMIT_FE | 61734 | 29.2282 | 27.6713 | 29.4563 | 28.7630 |
| zsre | SPHERE | 61735 | 99.7677 | 96.3865 | 86.2276 | 93.7635 |
| cf | ALPHAEDIT | 61778 | 99.7000 | 96.3250 | 73.5550 | 88.2174 |
| cf | ALPHAEDIT_BLUE | 61779 | 99.5500 | 97.5250 | 75.0550 | 89.2258 |
| cf | MEMIT_FE | 61780 | 87.1500 | 85.7000 | 57.2150 | 73.8534 |
| cf | SPHERE | 61781 | 99.7000 | 95.7250 | 74.2650 | 88.3861 |

## Qwen 입력 상태

SH4 ba784fa3 cessation 수신: FT61783 launcher storage failure, 나머지23 pending GPU/archive 취소. 성공 main/W0 없음. handoff20 metadata84488B를 결속할 대상이며 새 SH2 job은 아직 미제출.
원 Qwen CF 일정은 W0_AND_W20_FIRST2000, zsRE 생성 없음. GPTJ의 DEFERRED를 Qwen에 상속하지 않음. BLUE L2=1 및 원 source/config/hparams 유지.
현재 단계는 LOCAL_ASSET_IDENTITY_AND_HOST_BINDING_IN_PROGRESS. SH2 local 모델/C0/P 존재(metadata)만 확인했으며 content identity 및 제출 PASS로 주장하지 않음. server2 cap4 또는 stricter, 기존 다른 job 불변. README는 GH 단독 통합.
