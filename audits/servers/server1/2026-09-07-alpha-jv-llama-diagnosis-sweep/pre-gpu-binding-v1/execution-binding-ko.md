# Alpha-JV D/S 실행 결속

D+S 전체 예산은 사용자 addendum SHA `54789342ea78151442a09ec07696db9ce80caae6f39a665c2ca4568b881a3c15`의 48 GPUh다. Server1 cap2, 1GPU/process, mem182272M이다. 첫 S wave의 최대 예약은 2모델 × 8시간 = 16 GPUh이며 queue 대기는 과금하지 않되 load·FD·평가·technical attempt·COMPLETING residency는 모두 과금한다. 제출 전 별도의 fresh resource admission과 held inspection을 요구한다.

CPU/compile/session/source/asset 결속은 실제 GPU fidelity 통과를 뜻하지 않는다. 각 모델에서 먼저 fixed stock-z와 cold W0/M0를 봉인한 뒤 정상 신호 raw JVP/FD와 한 실제 joint-step의 overlay/materialized parity를 검사한다. FD의 logits 관측은 결과와 무관하게 첫 S_DEV request의 모든 위치/전체 vocabulary이며 raw activation FD는 전체 S_DEV다. 허용 오차는 기존 v3.1 그대로다.

S는 각 모델의 동일 cold W0/M0, frozen z·context·qref·N0를 공유하는 7개 독립 trajectory와 9개 endpoint다. Official은 별도로 1회 실행한다. T4 parent의 T1/T2 prefix는 실제 물리 endpoint를 임시 관측하되 history append 및 persistent-history promotion은 0이다. 모든 완료 node의 training semantic observer는 decision 영향0이며 별도 forward/time으로 기록한다. Raw tensor와 실제 W/M endpoint 및 chronological factor journal은 ignored local namespace에만 저장한다. 저장 자체를 exact replay로 부르지 않는다.

이전 `77358b15`의 native/ordered runtime 세 package는 byte-unchanged다. 실제 실행은 새 source commit을 별도 detached clean worktree에 고정한다. 추가 분석 source 변경은 실행 worktree에서 하지 않는다. 과학 성능 저하/near-stall은 설정 변경이나 제외 근거가 아니다.

D historical exact W9/fixed-target bytes의 Server1 가용성 문제는 그대로 보존하며 S를 막지 않는다. S를 historical B10의 재현·수리로 주장하지 않는다. Reserved audit300은 outcome을 열지 않는다. Main 통합은 GH review 후이며 이번 push는 전용 branch만 허용한다.
