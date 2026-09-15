# EP47962 CPU 완료 검토·추가 설계 분석

Instruction ODEEDIT-S06-EP-TW1-47962-COMPLETED-DETAILED-REVIEW-SH4-V1.
Addendum ODEEDIT-S06-EP-TW1-47962-DESIGN-CONFORMANCE-ADDENDUM-SH4-V1.
Host server4; owner server4-server-head; session 01a04939-b5c7-7a03-ba2d-ef3343d62cfd.

실행 6d317bdb2660d7e9919bc3a9fb878564e9729e37와 분석 코드/보고 commit은 구분했다.
지정47962 단발 sacct: COMPLETED0:0,7694 allocated GPU-sec.
Actual1000requests/10commit/10history/9links와 selectedW4/M4 CPU검산 완료.
최종 독립 NLL: RS998/1000,PS1942/2000,NS8056/10000.
C1=5/C05=3/RAW=2; RAW B5/B10은 모든 corrected E>Ep라 선택.

설계 추가 장은 15requirement→실행 file/function/line/SHA→artifact→확인수준 표,
10batch q/gradient/projection/ball/trust/actual action, correctedB1/RAWB5 사례를 포함한다.
Halfspace projection9/10; post-ball <gE,C>positive8/10;
selected correction/native0.00675–0.02355%; neural FD 검증 완료라는 뜻은 아니다.

검토 범위:116rawmembers13407776774B fullSHA/size/stable stat,
10CP10570873202B weights_only CPU finite/shape/3hash conventions bridge,
65내부file references,164실행source members,42generic reductions.
독립 JSON reducer12files 재실행 byte일치, CSV-only MatplotlibPNG6개 byte일치.
분모/paired정수보존/40selector/15design rows/10CPUfixture/9syntax files 확인.
부정확한 native/map solve 합산을 피한다: nativeRHS10counter와 mapA10source-count는 별개.
Timer중첩과 purewriter/IO 미분리는 보고서에 유지했다.

Raw-free package: experiment-reports/servers/server4/ep-tw1-c4-2026-09-15-v1/gate-skip-r1/completed-review-v1/
Report SHA66cc456a3fce2925e23565e4098b6381d3233021ab4e921f9c03fd7e9556c121.
검증 status SKIPPED_USER_DIRECTED / numerical_validation=NOT_ESTABLISHED.
새GPU/model/forward/eval/Slurm mutation/rsync/삭제/CAKE작업0.
다른paused task와 source/공유dirty는 불변, scientific_promotion=false.

초기fetch timeout 기록 후 최신0043108(main)의 review/addendum를 ff 수신했다.
기존PROTOCOL 동일SHA 및 이전full-read evidence를 재사용했다.
본scope만non-force통합하며 global/다른SH/source runtime은 변경하지 않는다.
Generic access script의 runs/* 제외는 본explicit envelope가 허용한 compact
runs/odeedit_ep_tw1_47962_review_s4_v1/server4.json 한 파일에만 예외로 기록한다.
공용 access policy는 수정하지 않는다. Final main HEAD/tree는 post-push GH handoff에 결속한다.
