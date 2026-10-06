# Interference-priced group-L1: SH4 실행 기록

Instruction/nonce: `USER-GH-SH4-JLZ-INTERFERENCE-L1-20261006`. Task: `jlz-interference-priced-l1-2k`.

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
