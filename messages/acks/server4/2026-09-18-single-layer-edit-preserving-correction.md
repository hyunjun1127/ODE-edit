# SH4 ACK — ENFC M reuse-first

Instruction: `ODEEDIT-S06-SINGLE-LAYER-EDIT-PRESERVING-CORRECTION-M-SH4-V1`.
Nonce: `ODEEDIT-GH-SH4-ENFC-M-REUSE-INITIAL-GATE-20260918-R1`.

원문·설계·contract·원 CRLF cells·positioning·BLUE audit·CPU 참조 및
receipt를 FULL_READ하고 task-local authoritative에 create-once 결속했다.
Full-read receipt SHA256:
`96aab038b4d8de8eace27d17961ee31db5933873e3c2fb6c2aa197328aaf7ac3`.
시작 origin/main은 `3d76fe9312e91f997e60f708581a09efc51c3e3c`이다.

최신 권한은 M 재사용 우선, 필요한 actual T와 M 누락 실행/평가만이다.
S/R/L 제출·자동 확대0, Server4 project cap2. T PENDING/T_PASS/source 게시만으로
종료하지 않는다. 모든 필요한 M 정상 제출 후 실제 M initial gate 또는 검증된
main GPU-resource shortage에서만 monitoring pause한다. 새 M의 모든 final L4
endpoint를 보존하며 과거 noCP/FD-skip/mean plateau는 상속하지 않는다.

첫 재사용 감사: 10개 독립 cold100 batch, 8arm/episode. B1의 실제 cold native
WN/target/key/zeroM/context를 보존 proposal에서 확인하여 native fit을 재사용한다.
다른 9개 과거 sequential batch는 cold M 대체가 아니므로 shared native fit을
각1회만 준비한다. B1 N4 canonical pair 및 W0 first1000의 같은 case 관측은
재사용하며, 미보존 greedy32/Dev·새 correction 결과만 추가한다. 기존 CP 경로
부재를 전체 fit 재실행 이유로 사용하지 않았다.

원본 raw/source/teacher/다른 task를 변경하지 않는다. 독립 CPU 담당의 geometry,
all-token/optimizer 및 observer 경계 검토는 실제 Llama T와 구분하여 기록한다.
실제 T49928 및 후속 필요한 M의 상태는 별도 create-once 실행 receipt에 기록한다.
이 ACK는 T_READY, M 제출 완료 또는 효능 PASS가 아니다.

## 2026-09-18 최신 T 생략 recall

nonce ODEEDIT-GH-SH4-ENFC-SKIP-T-ALL-M-20260918-R1 적용. T 신규0,
M10 independent cold episode만 cap2, reuse-first/B1fit재사용/80final L4보존,
S/R/L0. T=SKIPPED_USER_DIRECTED/full_numerical_validation=NOT_ESTABLISHED.
정본은 prior FULL_READ SHA로 재결속했고 새 envelope/override 전체를 읽었다.
새 waiver routing4CPU 검사 및 import 확인, 원 method 수치/source 불변.
실제 M_INITIAL 또는 검증된 GPU shortage 후 pause 규칙은 유지한다.
이번에는 제출전 storage reserve72GiB 미충족으로 M0/10등록, 새GPU0이다.
이는 M PENDING/초기gate 통과가 아닌 RESOURCE_BLOCKED_STORAGE_PRE_SUBMISSION이다.
