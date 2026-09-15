# GH → SH4: 진행 중 EP-TW-1 상세 리뷰에 설계 대비 실제 동작 분석 추가

Instruction ID: ODEEDIT-S06-EP-TW1-47962-COMPLETED-DETAILED-REVIEW-SH4-V1 (동일 task 추가)
Amendment ID: ODEEDIT-S06-EP-TW1-47962-DESIGN-CONFORMANCE-ADDENDUM-SH4-V1
Nonce: ODEEDIT-GH-SH4-EP-TW1-47962-DESIGN-CONFORMANCE-20260915-R1
수신: server4/session01a04939-b5c7-7a03-ba2d-ef3343d62cfd/CWD /data/janghj/ODE-edit/repository hyunjun1127/ODE-edit.
발신 GH/session01a04939-8873-7673-8dca-4c7fc5e31af0.
사용자: “그리고 SH4의 현재 리뷰부분에 ours가 어떻게 작동했는지(우리가 설계한 대로 잘 작동한건지)도 자세히 리뷰에 넣으라고 해”.

현재 리뷰를 중지하거나 처음부터 다시 시작하지 말고 아래를 최종 보고서에 통합한다. 기존 수치/원raw/실행source는 불변. 같은 task의 직접 관련된 사용자 추가요청이므로 현재 활성 리뷰 turn에 전달한다. 별도 단계 승인·GH중복감사 대기0.

## 필수 추가 장: 설계 → 실제 코드 → 실제 실행 증거

최종 diagnostic-report-ko.md에 “EP-TW-1 설계 대비 실제 동작”을 충분히 상세하게 추가한다. 단순 source hash PASS나 final score만으로 설계대로 작동했다고 결론 내리지 않는다.

설계 v3/계약 + 사용자 gate-skip override를 함께 기준으로 삼고 실제 executed source6d317bdb2660d7e9919bc3a9fb878564e9729e37를 읽는다. main의 최신 함수가 실제 실행 소스와 다르면 별도다. 다음을 요구사항·설계식·실제 함수/줄·저장artifact·batch별 관측값·확인수준의 표로 연결하라. 원래 필수였다가 user-directed skip된 항목을 silent 구현누락이나 PASS로 재분류하지 않는다.

1. **Native와 실제 출발점.** Wentry→own native Vp, target100/원본 fit·solve, L4-only/coldM0, C0=0의 실제 Vp anchor를 확인한다. Vp를 factorized RA 근사로 대체하지 않았는지, 후보 scale이 native 전체 update가 아닌 correction C에만 적용됐는지 코드·저장state로 구분한다. 각 batch fixed z/K/P/M/A와 현재 자기 trajectory state를 사용했는지, 새 compute_z/중복 native solve/미승인 layer update가 없었는지 기록한다.

2. **목적함수와 입력.** E는 각 request의 desired-answer token 평균 후 current100 평균인지, D는 고정 W0 teacher의 S64 full-vocabulary KL(p0||pV)·128 scored positions·문서평균인지 reduction/mask/normalization을 확인한다. gE/gD 별도 sweep, C0 평가점, stop-gradient 영역, gC=gW A^T 연결 source를 설명한다. 실제 추가 backward 횟수·저장 gradient hash/norm을 근거로 하되, FD 미실행을 derivative correctness 증명으로 대체하지 않는다. Dev128/old/official P-N/미래sample이 controller 입력으로 들어가지 않았는지 source/data-flow를 확인한다.

3. **진척 보호 방향의 실제 역할.** d=-gD+min(<gE,gD>,0)/||gE||²*gE와 zero-gE 분기가 코드와 맞는지 점검한다. 10batch q=<gE,gD>, gradient norms/alignment, projection coefficient, projection active 여부, <gE,d>/<gD,d>, 저장 KKT 잔차를 표로 만든다. 저장 tensor로 필요한 CPU scalar 산술을 독립 재계산하되 GPU와 CPU 반올림차이는 기록하고 bit-exact GPU gate를 사후 요구하지 않는다. 반공간 방향이 실제로 자주 개입했는지/전혀 필요 없었는지를 수치로 설명한다. 이 일차 조건은 finite-step 개별 request·PS·old retention 보장이 아니다.

4. **Step·target ball·trust 처리 순서.** alpha_cap/epsilon/zeta=.25, actual native norm을 기준으로 한 alpha, Zp+alpha*d의 원래 request별 anchor/radius projection, C=projected−Zp, C A trust retraction 순서를 대조한다. alpha/clamped request count/target-ball excess/retraction ratio/수정 전후 C·CA norm과 <gE,C>/<gD,C>를 batch별 표로 기록한다. Project 후 일차 부호가 바뀐 경우도 숨기지 않는다. 저장되지 않은 원시 중간tensor는 계산해서 실제 저장 관측인 것처럼 만들지 않는다.

5. **실제 finite screen/선택.** 네 후보를 동일 Vp에서 독립 materialize했는지, duplicate byte 처리, actual FP32 trust, E≤Ep(양의 허용량0), raw TF-strict의 정확 ID-set subset, finite/trust/quality 탈락 사유를 전 후보·전 batch 표로 공개한다. Feasible 중 D64 선택과 사전 lock의 D numerical tie→RAW priority→실제 correction norm→고정 ID 순서를 확인한다. Strict count만 같은 경우를 ID 보존으로 부르지 않는다. C1=5/C05=3/RAW=2라는 첫 집계를 batch/선택근거와 연결하고, RAW 두 batch에서 feasible 없음/개선 없음/동일bytes/numerical tie 등 실제 이유를 밝힌다. 후보별 성능이 좋아도 사후 후보를 다시 고르지 않는다.

6. **실제로 무엇이 weight에 쓰였는가.** 선택 ID·route의 C/A·raw Vp·실제 selected checkpoint를 연결한다. 가능한 저장 근거로 ||Vp−Wentry||, ||Wselected−Vp||, ||Wselected−Wentry||, correction/native 비율·방향 cos를 보고한다. 저장tensor가 충분하면 correction의 native 방향 평행/직교 성분을 CPU로 분해해 단순 native strength 축소와 동일한 write인지 수치로 표시할 수 있다. 이는 새 alpha sweep/반사실 평가가 아니며 성능 원인 확정은 하지 않는다. FP32 CPU 재구성의 차이와 원래 저장된 actual weight hash 일치를 구분한다.

7. **Commit/history/ledger/다음 entry.** inner history0/마지막 실제 selected endpoint에서 native finalizer1, L4 M append1, candidate 평가 nonmutation/RAW복원·선택commit 순서, 10commit/9links와 accepted ledger를 대조한다. 평균 E·TFstrict 보호가 canonical RS/PS/NS와 다른 정의라는 점을 표기하고, RAW→selected 및 at-write→최종 유지·손실·회복을 연결한다. 실패한 현재 요청이 과거 accepted label을 덮어썼는지 여부 등 ledger 규약을 실제 로그/저장state로 확인한다.

## 실행 과정 설명과 판정 방식

- 전체10batch 요약표 외 실제로 correction이 선택된 batch와 RAW가 선택된 batch를 각각 최소1개 골라, 설계식의 변수에 관측 수치를 대입해 출발점→direction→projection/trust→candidate screen→selected weight→history/next-entry를 읽기 쉬운 사례로 설명한다. 선택은 대표 동작 유형을 보여주기 위한 것으로 전10batch 자료를 숨기지 않는다.
- 모든 batch가 동일 selection이 아니므로 “method가 한 번이라도 호출됐다” 수준에서 끝내지 않는다. 실제 correction 크기와 effect, local RAW 대비 E/D/strict 그리고 관측전용 PS/NS 변화의 산술 값을 함께 보여준다.
- 판정은 SOURCE_CONFIRMED / STORED_EVIDENCE_CONSISTENT / DEVIATION / NOT_RECORDED / SKIPPED_USER_DIRECTED 등을 구분한다. 범위 안에서 “설계 규칙을 실제 적용했는가”는 답하되, numerical_validation=NOT_ESTABLISHED와 model-level derivative/FD/GPU replay 미검증 경계를 유지한다.
- 설계와 실행이 어긋난 사실이 발견되면 해당 batch/함수/자료/수치/영향받는 표를 명시한다. source/과거 결과를 고치거나 새실험으로 대체하지 않는다. 가능한 상세 CPU 분석은 계속하고 technical mismatch와 효능 비교를 섞지 않는다.
- 사용자 요청에 따라 source-backed 동작 설명과 설계 준수 판정을 보고서에 포함하는 것은 허용한다. 과학적 우월성·인과 기여·새 방법/후속 tuning 제안은 GH 소유로 유지한다.

## 권한과 완료 경계

기존 리뷰 envelope의 CPU-only 분석/write/report/main 권한 그대로다. review_nogate/ 하위 분석/테스트/plots와 같은 report package에 design-conformance.csv, per-batch-mechanism.csv 등 필요한 작은 표를 추가한다. 새 model/GPU/FD/ULP/forward/backward/Slurm/repair/rerun0. 생략된 gate를 다시 prerequisite로 부활시키지 않는다. runtime와 raw 불변.

PNG는 코드 생성만, 최종 상세 보고서·분석 source·표·figure·manifest/receipt를 기존 완료리뷰 package에 통합하고 own-scope nonforce main에 게시한다. 추가 장의 coverage와 한계까지 끝나야 이번 expanded review를 최종 완료로 보고한다. 진행 중인 CPU 작업을 중복 계산하거나 GH에게 다시 검사해 달라고 요청하지 않는다.

CAKE baseline은 사용자가 “현재 리뷰 완료 보고를 GH가 받은 뒤 지시”하도록 했으므로 **아직 task 전달/clone/제출 대상이 아니다**. 이 추가 리뷰까지 마친 최종 handoff를 GH가 받으면 GH가 별도로 CAKE 지시를 전달한다. 첫표/M0는 CAKE 활성화 조건이 아니다.
