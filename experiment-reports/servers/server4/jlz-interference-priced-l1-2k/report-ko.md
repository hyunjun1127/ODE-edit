# Interference-priced group-L1: SH4 실행 기록

Instruction/nonce: `USER-GH-SH4-JLZ-INTERFERENCE-L1-20261006`. Task: `jlz-interference-priced-l1-2k`.

## 2026-10-06 USER recall: 1,500-edit 중간 결과

PRICE59768의 B1–B15를 CPU 검산하여 게시했다. 15 commits/14 own-state joins/75 history appends, 누적 R1500/P3000/N15000 관측 완료. W15 RS **99.867% (1498/1500)**, PS **91.467% (2744/3000)**, NS **85.667% (12850/15000)**.

[W15 상세 보고서·CSV](w15/report-ko.md). 원본 row/token identity·분모와 저장 집계 일치, 가격/projection/controller 및 W/H/RNG/ledger 연결을 검산했다. 새로운 모델 평가 없음. 한정 1회 scheduler snapshot은 PRICE RUNNING/collector PENDING이다. B16 이후는 검토하지 않았으며 W20 완료를 주장하지 않는다. 기존 job/봉인 source/취소한 대조군은 변경하지 않았다.

## 59721 실패 및 사용자 교정 요청 — 등록 당시 기록

2026-10-06 사용자 recall: `59721 fail되었으니 교정해`. 후속 요청에 따라 동일 조건의 기존 W0 데이터가 검증되면 첫 2,000개 W0 평가를 새로 수행하지 않는다.

PRICE 59721은 `PROJECTION_SORTED_BREAKPOINT_ROOT_UNAVAILABLE`로 FAILED(exit1), 부모 GPU 할당 1,250초였다. B1 commit 0회이며 batch-entry rollback receipt의 verified=true다. 원 source/config/raw/log는 유지한다. 이 오류는 품질·수렴 실패가 아닌 sorted-breakpoint 구현 경계 오류다.

실제 저장 피연산자에서 shared spend가 beta=.75인 plateau에 인접한 두 근이 FP64 반올림으로 해당 구간을 약2ULP 벗어났다. 기존 코드의 strict interval/equality 검사가 후단 KKT 검사 전에 `root unavailable`을 냈다. 교정은 최초 breakpoint에서 기존 block-length 식을 그대로 사용하고 기존1e-10 shared/KKT 기준을 통과할 때만 해당 경계 해를 반환한다. 계수·목적·projection metric·FP32 cap/공유허용오차는 변경하지 않았다.

원 실패 피연산자 회귀에서 원 오류를 재현했고, 교정 결과 tau=11.167240484457963, spend−beta=1.1102230246251565e−16, complementarity=1.2398127507371418e−15로 기존 기준을 통과했다. 이는 실제 실패 scalar의 회귀이며 새로운 toy suite나 GPU/모델 재검증이 아니다. FP32 전체 tensor는 noCP이므로 재생하지 않았고 새 실행의 기존 gate를 유지한다. [회귀 receipt](../../../../audits/servers/server4/jlz-interference-priced-l1-2k/repair-59721-regression.json).

최초 recall 시 원 FLAT 59722 RUNNING, REVERSE 59723 PENDING, collector 59724 PENDING이었다. 이후 사용자 명시 요청으로 FLAT/REVERSE만 exact owner/source/argv 확인 뒤 취소했고 CANCELLED 및 할당 해제를 확인했다. 부모 GPU 비용은 각각565초/0초다. 원 collector/source/raw/log는 보존했으며 대조군 재등록은 없다. [취소 receipt](../../../../audits/servers/server4/jlz-interference-priced-l1-2k/user-cancel-controls.json). 교정은 새 immutable PRICE-only attempt와 해당 CPU collector로 분리했다.

위 repair의 W0 재사용은 원 PRICE59721의40chunks/26,000rows, R2000/P4000/N20000을 CPU 독립 재집계하고 model/input/token/native/evaluator/runtime/cold identity를 결속했다. 수정된 projection/source HEAD 전체가 같다고 주장하지 않고 W0 관측에 관여하는 unchanged byte closure와 setup/observer AST를 구분했다. 새 설정의 W0 상태는 QUALIFIED_EXACT_REUSE다. 원 930.845초는 이전 비용으로 유지하고 새 W0 관측 시간/forward는0으로 기록한다. 원 raw 복제·hardlink·symlink·checkpoint 재사용은 없다. 실제 새 모델 cold state/device/CPU8/runtime 일치는 실행 중 재확인하며 실패 시 새 W0 forward로 fallback하지 않는다. [결속 receipt](../../../../audits/servers/server4/jlz-interference-priced-l1-2k/repair-59721-binding.json).

Repair 등록/release 완료: PRICE **59768**, CPU collector **59769**. 최종 제한된 초기 snapshot은 각각 PENDING/Resources, PENDING/Dependency였다. 대조군 dependency는 없으며 새 PRICE부터 실행한다. 등록 직전 cap/owner/source/CPU8/mem59392MiB/GPU1/48h/exportNONE/Requeue0 및 전량 held fullargv/script bytes 검사를 통과했다. Collector는 GPU0/CPU8/24576MiB/4h, afterany:59768이다.

실행 source `0415aba3c160170d306be8196792f198dad4d122`, config SHA `26096236ba0fe1a683c98d954904dbf0a048d4611f03cd62b1aef77f7c00091f`. 실행 원본은 `/data/janghj/ODE-edit/local/jlz-interference-priced-l1-2k/repair-59721/`. [새 제출 receipt](../../../../runs/jlz-interference-priced-l1-2k/repair-59721.json), [변경분 검토](../../../../audits/servers/server4/jlz-interference-priced-l1-2k/repair-59721-review.json). 위 source와 이후 보고서 게시 commit은 구분한다.

등록 당시 새 GPU/B1 및 실제 W0 runtime assertion은 NOT_OBSERVED였다. 이번 USER recall의 W15 검산이 이 초기 관측 상태를 갱신한다. PRICE-only 전체 기대치는 20commit/19join/100H이며 controls는 NOT_REQUESTED다. 봉인된 PRICE/collector는 미변경이며 이번 중간 보고 후에도 능동 monitoring/automatic resume/retry는 중단한다.

## 최초 등록 기록 — 아래 상태는 당시 snapshot

현재 상태는 SUBMITTED_RELEASED_RESOURCE_PENDING이다. PRICE/FLAT/REVERSE 각각 독립 cold W0/H0, fixed first2000, BS100×20을 구현·등록했다. 기존 v12-R 실행 source `635798ba276957312ec1aceda686ad563c906957`은 읽기 전용 재사용하며, W15 결과 게시 `782c4c7a`와 구분한다.

정본 `project/proposals/jlz-interference-budget-v1/`의 5파일을 전부 읽고 manifest SHA `b7c7e7644bd1878143ce6df43f58b384978b10e01a235617d29081f5052fb78b` 및 4개 member의 bytes/SHA를 검산했다. 원 bytes는 변경하지 않는다.

새 toy/synthetic 수치 suite, 별도 qualification job, small-B fit/B1 pilot은 없다. 실제 각 arm B1 c0의 기존 P/K/A/M에 대한 고정 LOO matvec와 첫 실제 proposal 사영 검사는 봉인 runner 내부에서 수행한다. Source/static/import 확인은 actual GPU PASS가 아니다.

이전 MAIN B16 ENOSPC의 쓰기 실패와 공간 소모 주체 NOT_IDENTIFIED를 구분한다. 새 저장 경계는 entry-price once, candidate JSONL once, fit summary의 hash/line-count 참조와 전체 serializer 상한 및 batch 경계 공간 guard다. Model/weight/H/P/K/M/optimizer 복원 bundle을 저장하지 않는다.

제출 직전 owned GPU queue는 비어 있었으나 server4 node의 GPU 8개는 할당 상태였다. 기존 작업 변경 없이 PRICE 먼저, FLAT/REVERSE는 afterany 자원 순서와 최대2GPU로 모두 등록·release했다.

| 역할 | Job ID | 제한된 최종 초기 snapshot |
|---|---:|---|
| PRICE | 59721 | PENDING / ReqNodeNotAvail, May be reserved for other job |
| FLAT | 59722 | PENDING / Dependency afterany:59721 |
| REVERSE | 59723 | PENDING / Dependency afterany:59721 |
| CPU collector | 59724 | PENDING / afterany:59721:59722:59723 |

실행 source는 `a9905b9fccbdb48b9b17e768368afa9e663f0bf5`, config SHA `ad6ca0f4987ecfd400c4b637682ba2dc1a8a2626f22de8594b8d7cadb1b7511d`, lock SHA `c66173da14839e078660a51166fa2169d8d5220f403d72192f1289411bb49f7a`다. 실행 원본은 `/data/janghj/ODE-edit/local/jlz-interference-priced-l1-2k/attempt/`에 봉인했다. Scheduler test-only는 GPU/collector 모두 return0이며 실제 allocation은 만들지 않았다. 실제 등록된 위 4job은 owner/fullargv/source script bytes/GPU/CPU/explicit memory/wall/exportNONE/Requeue0/dependency를 전량 held 상태에서 확인한 후 release했다.

각 GPU job은 1GPU/8CPU/59392MiB/48h, collector는 GPU0/8CPU/24576MiB/4h다. 현재 projectcap3·task최대2를 적용하며 요청 wall은 ETA가 아니다. 별도 reviewer의 좁은 source 검토는 PASS_WITH_WARNINGS/확정 blocker0이다. [검토](../../../../audits/servers/server4/jlz-interference-priced-l1-2k/source-review.json), [제출 결속](../../../../audits/servers/server4/jlz-interference-priced-l1-2k/submission-binding.json), [전체 argv/등록](../../../../runs/jlz-interference-priced-l1-2k/submission.json).

정식 resource pending 초기 인계에 따라 monitoring_active=false/automatic_resume=false/automatic_retry=false로 종료한다. 봉인 DAG만 승인 3arm/W20/collector를 계속한다. 실제 B1 LOO/사영/GPU 검증, B1→B2, W20은 NOT_OBSERVED이며 새 recurring monitor를 등록하지 않았다.

새 method 성능/ETA/실측 peak는 NOT_MEASURED다. 신규 GPU 실행 결과와 W20은 NOT_OBSERVED다. noCP / exact resume NOT_AVAILABLE. 원 raw와 실패 source는 KEEP이다.

최종 syntax/import/config 확인은 PASS이며 toy/수치실험/model load/forward는 0이다. 실제 20개 native pack·2,000개 순서·26,000개 관측 row identity를 재결속했다. Source 검토와 실행 중 실제 B1 검사는 별개다.

저장 계획은 3arm 전체 최대 8,738,832,384 bytes(collector·atomic·error reserve 포함), 다음 최대 batch 181,665,792 bytes와 error reserve 134,217,728 bytes다. 결속 시 free 119,460,401,152 bytes였다. Quota 명령은 없어 NOT_AVAILABLE이며 공유 파일시스템의 미래 여유를 보장하지 않는다. 매 fit 전 guard를 다시 수행한다. Host 31.778GiB, GPU 66.821GiB는 구현·구조·workspace reserve에 따른 계획값이지 실측 PASS가 아니다. Python console 상한과 native FD pre-batch 확인 범위는 구분했다.

정적 확인/입력·저장 계획: [preflight](../../../../audits/servers/server4/jlz-interference-priced-l1-2k/preflight.json), [binding](../../../../audits/servers/server4/jlz-interference-priced-l1-2k/binding_summary.json). 원 local 입력/config/receipt: `/data/janghj/ODE-edit/local/jlz-interference-priced-l1-2k/preparation/`.

NO_BROADCAST_NOT_REQUIRED: 같은 S4 local 자산을 읽기 전용 사용하며 Git에는 승인 scope source·소형 receipt·보고서만 게시한다. 대형 raw/model 전송 또는 기존 자료 삭제는 수행하지 않는다.
