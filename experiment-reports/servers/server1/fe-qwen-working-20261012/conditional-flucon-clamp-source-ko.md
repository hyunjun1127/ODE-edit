# clamp 0.5 조건부 최종 CF 생성 adapter

USER-FE-QWEN-WORKING-CURRENT-Z-20261012-R1 동일 작업의 source-only 후속.
추가 파일만 게시하며 기존 e87 generation 및 59c531d8 editing 파일은 불변.

새 flucon_clamp entrypoint는 rescue_clamp의 실제 SELECTED 증거와 양 dataset
최종 W20의 독립 raw/query/commit 재검산을 재사용한다. 추가로 CF/zsRE 각각
runtime/CP identity 및 native config L2=100/clamp=0.5를 검산한다.
실패/QUALITY_EARLY_STOP/부분/옛 clamp1 증거는 거절한다.

CP validator는 새 명시 분기이며 옛 guard를 해제하거나 clamp1로 relabel하지 않는다.
fullSHA 이후 mmap/선택 BF16 weights만 정확 FP32 복원하며 H 복제/소비 없다.
기존 NativeGenerationObserver/고정 참조/seed/원 CF2000/query/독립 raw 채점/
paper x100 표시/공통 eval_only W&B schema를 그대로 사용한다.

CPU 38 PASS: 새6 fixture + 기존32 회귀. 새 fixture에는 양 dataset runtime/CP
clamp 불일치 및 실패 receipt 거절, 미완료 proof 선차단, 실제 tiny mmap 복원,
unselected/RNG 보존, 옛 gate 불변, 생성/query/reducer/schema 재사용이 포함된다.
source171 SHA PASS/Python446/external imports0. 실제 pretrained/GPU/성공 proof는
NOT_RUN이며 실제 checkpoint와 최종 결과를 만들지 않았다.

API/입력: official/diagnostics/fe_qwen_current/FLUCON_CLAMP.md.
Root가 실제 proof/경로/resource를 나중 결속한다. 현재 heldout 실행 gate가 아니다.
SH1/SH2 제출0, rent 조회/전송/삭제/README/Overleaf 변경0.
NO_BROADCAST_NOT_REQUIRED: 소형 source/docs/report만 Git 게시.
