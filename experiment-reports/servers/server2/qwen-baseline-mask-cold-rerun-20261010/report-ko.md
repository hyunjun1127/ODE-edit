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

현재 source 준비 단계이며 실제 등록 ID/held 검사/release는 후속 submission 영수증으로 구분한다. 승인 cap4 및 더 엄격한 node/QoS/storage를 실제 등록 직전에 검산한다. GPU 완료 대기/반복 monitor/자동 retry 없음. 원 raw/CP/source 삭제·대형 broadcast 없음 (`NO_BROADCAST_NOT_REQUIRED`). README는 GH 단독 통합.
