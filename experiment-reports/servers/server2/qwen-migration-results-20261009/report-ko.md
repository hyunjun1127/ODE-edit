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
최종 단계: REGISTERED_INSPECTED_RELEASED. 원 config12/stream2 일치, 모델/token/C0/P/reference full SHA 일치, CPU5 및 source157/Python298/import0 확인. 실제 GPU qualification은 NOT_RUN_USER_DISABLED.

원 runner의 저장공간 오류 raise 뒤 도달 불가능한 편집 본문은 SH2 복사본에서 unindent만 수리했다. 원 본문의 연산 AST/순서 일치 테스트 및 runner-exact.diff를 보존한다. 공통/원 SH4/frozen source는 수정하지 않았다.

실행 source `69bfbb2cdffe24072733950c671e47597ff9fbd5`, source-lock SHA `2c368505e34e89e51e178ef59410dad5a44d9cde488ec3a0810758e930442b70`.
release snapshot 2026-10-09T08:16:34.896906+00:00: 전량 PENDING, 과학 시작/완료 및 실제 W&B run readback은 미검증. 인증된 W&B 프로젝트 읽기만 확인했다.

| method | CF GPU | zsRE GPU |
|---|---:|---:|
| FT | 61898 | 61900 |
| MEMIT | 61902 | 61904 |
| AlphaEdit | 61906 | 61908 |
| BLUE | 61910 | 61912 |
| FE | 61914 | 61916 |
| SPHERE | 61918 | 61920 |

archive CPU 61899/61901/61903/61905/61907/61909/61911/61913/61915/61917/61919/61921; collector61922. 모든25개 held owner/Command/WorkDir/자원/source/input/dependency 검사 후 release. 실제 afterany 목록은 submission.json/CSV에 있다. 모델별 W0는 각 dataset FT producer 내부 원 필수 관측 한 번이며 후속이 exact receipt를 검산한다. 네 lane 상한, 기존 다른 job 변경0.

현재 source freeze에는 SH1 server2 trust READY가 없으므로 archive 단계는 ARCHIVE_PENDING_KEEP_SOURCE. server4 trust를 재사용하지 않고 전송/삭제0. 전체12 CP 보존+네 lane atomic 복사+32GiB reserve 필요115580338176B, 관측 여유321153998848B. receiver 준비를 science 승인 gate로 삼지 않았다. 기존CP 및 새CF/zsRE payload KEEP; 검증되지 않은 보존/삭제를 주장하지 않는다.

root/log: `/mnt/raid5/janghj/ODE-edit/local/qwen-baselines-server2-20261009/registration-r1/` (`logs/`, `registrations/`, `inspections/`, `released.json`). 새 장기 monitor/자동 retry/기존 job 취소 없음. README는 GH 단독 통합.
