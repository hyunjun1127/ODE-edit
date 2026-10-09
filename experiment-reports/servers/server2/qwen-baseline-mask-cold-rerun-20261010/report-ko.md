# Qwen mask cold rerun — SH2

권한: USER-GH-QWEN-BASELINE-MASK-COLD-RERUN-20261010-R1. 접수 turn `01a1214b-7e89-7673-90eb-8e7877f56555`.

## 준비 및 보존

영향받은 8개 CF/zsRE MEMIT·AlphaEdit·MEMIT_FE·SPHERE만 새 cold W0/빈 native history로 준비한다. GH 공통 generator SHA `35506690c41ecb7d59f11660da41dde50338f5a2ba9613735e045a8c7ab98db4`를 채택했다. Qwen registry의 네 방법 모두 EasyEdit generator에 결속된다. 별도 sphere/BLUE 공통 구현은 수정하지 않았다.

새 CF 4개는 **DEFERRED_CHECKPOINT_EVALUATION**: 생성 호출·FLU/CON placeholder 없음, factual 및 native 편집·checkpoint 일정은 유지. 최종 W20 CP는 후속 평가 소비자 대기 상태로 KEEP. 기존 FT/BLUE의 평가 일정/source는 변경하지 않는다. 첫 실제 native 호출의 context는 새 local 경로에 저장하고 source/config/실제 module SHA 및 context SHA를 기록한다. 사전 GPU qualification은 NOT_RUN_USER_DISABLED.

정확 owner/source/Command/WorkDir/state 확인 후 downstream-first 취소한 15개: 61974/73/72/71/70/69/68/67/66/61/59/58/57/55/54. 이미 완료된 61956/61960은 취소하지 않았다. FT61898/61900 및 BLUE61962/61964, 모든 OURS는 KEEP. 보호 GPTJ 평가 61942..47의 source/ID는 유지하고 취소된 resource frontier만 4개 lane으로 재결속한 뒤 hold 해제했다. 원 영수증은 ignored control-r1에 보존한다.

FT61900은 기존 20commit/최종 W20 pointer/CP 전체 SHA를 검증했다. 별도 Qwen FT eval-only 중복 등록은 발견되지 않았다. 새 평가 한 번만 허용하며 native 편집·W0·CF·generation은 실행하지 않는다. 원 CP는 271,632,161 bytes, SHA `55657ef22edfc8b8eead2912bdd716bbef7957989a384fcb32d19de435b2f932`.

## 검증

최소 CPU 52 PASS: own exact scope/native AST/DEFERRED caller→fake SDK/최종 factual+CP evidence, 기존 Qwen migration, GH 실제 tiny generator mask 3개. 과거 AST fixture는 이번 명시적 context receipt 및 DEFERRED 조건 두 차이만 정규화했다. 최초 CPU 실패 및 수정은 실제 native 실행 실패가 아니다. 독립 reviewer는 사용하지 않았다.

Official source 166 SHA PASS, external task imports 0. Qwen 2K public query CPU: 24,858 queries, E6691/G6691/Loc11476, input/target mismatch 0. 이는 pretrained GPU/수치 parity 또는 W&B online PASS가 아니다.

## 실제 등록·release

실행 source/main publication은 `7b5097aa447946e35de42229c22b0c0feabd11ae` (own 구현 `5438fd21`)이다. Source lock SHA `fafaeac54b2ccaa1fd52839c8d2f0b740e0d9a35fc99f5ed74b8dd49deaf190e`, input lock SHA `4ae67a5f6a9ff6ee05100483f5f3d931b58a279dd1bd8800229c5bcdb580ddae`. 이 보고서의 후속 게시 commit은 실행 source와 구분한다.

2026-10-10 00:46:20 KST: 아래 전량 held 검사/release 완료, snapshot은 전부 정상 dependency PENDING이다.

| 범위 | old | new GPU | afterany |
|---|---:|---:|---:|
| FT zsRE final2K eval-only | 61900 CP 보존 | 62072 | 61945 |
| CF MEMIT | 61954 | 62073 | 61946 |
| CF AlphaEdit | 61958 | 62075 | 61947 |
| CF MEMIT_FE | 61966 | 62077 | 61944 |
| CF SPHERE | 61970 | 62079 | 62072 |
| zsRE MEMIT | 61956 완료 역사 보존 | 62081 | 62073 |
| zsRE AlphaEdit | 61960 완료 역사 보존 | 62083 | 62075 |
| zsRE MEMIT_FE | 61968 | 62085 | 62077 |
| zsRE SPHERE | 61972 | 62087 | 62079 |

GPU0 archive/KEEP jobs: 62074/76/78/80/82/84/86/88, collector **62089**. Receiver binding 없음 및 CF deferred consumer 미완료이므로 원 CP KEEP; 실제 transfer/delete 0. Archive 완료로 보존 검증을 대신하지 않는다.

GPU 각각 1 A6000/CPU6/59392MiB, cold main 48h·FT eval-only 4h 요청(ETA 아님). CPU jobs 2CPU/4096MiB/4h. QoS `lab_gpu_s2` cap4 및 정확 whole-owner frontier를 검사했다. protected FT/BLUE 3개와 GPTJ eval을 먼저 유지하는 네 lane이며 canceled ID는 새 DAG에 없다. 필요 보존+atomic replacement+32GiB reserve 합계 108,175,294,464B; 당시 가용 약248GB. 실제 8개 source/config/script/resource/dependency/CP policy 검사 영수증은 registration-r1에 있다.

경로: `/mnt/raid5/janghj/ODE-edit/local/qwen-baseline-mask-cold-rerun-20261010/registration-r1/`. GPU 로그 `logs/<cell>-gpu-<job>.out/.err`, main CP `runs/<cell>/checkpoint/`. 첫 실제 context SHA는 `runs/<cell>/native-context-identity.json`에서 생성 예정이며 현재 미관측이다. FT62072의 config SHA는 actual job 기반 source 함수로 결정되며 `93f630caf6d4ed0c6ef47d5bc1e148d624c3bc36cf6e9364650c09f2372e6b3c`; 별도 CPU schema projection을 submission receipt에 기록했다.

W&B 공통 online logger/실제 job ID/name/source/config를 연결했으나 새 run은 아직 시작 전이다. 프로젝트 접근 `ONLINE_PROJECT_READ_VERIFIED`와 run-level remote readback `NOT_STARTED_PENDING`을 구분한다. 새 W20 성능·context·GPU PASS를 주장하지 않는다.

GPU 완료 대기/반복 monitor/자동 retry 없음. 원 raw/CP/source 삭제·대형 broadcast 없음 (`NO_BROADCAST_NOT_REQUIRED`). README는 GH 단독 통합.
