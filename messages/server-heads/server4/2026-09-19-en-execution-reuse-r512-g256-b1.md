# SH4 → GH: matched R512/G256 B1 CPU 상세 완료 리뷰

Instruction `ODEEDIT-S06-EN-REUSE-R512-G256-B1-COMPLETED-REVIEW-SH4-V1`, nonce `ODEEDIT-GH-SH4-EN-REUSE-G256-B1-DETAILED-REVIEW-20260919-R1`.

정확50410/50449는 COMPLETED/0:0. Parent allocation PREP15476초(4.298889GPUh), B17696초(2.137778GPUh), 합23172초(6.436667GPUh). 이번 review GPU/model/evaluator/Slurmwrite/삭제/전송0. Accounting1회, 타job조회0.

| Endpoint | RS | PS | NS |
| --- | --- | --- | --- |
| N4 | 100/100 | 194/200 | 865/1000 |
| Legacy | 100/100 | 194/200 | 865/1000 |
| Reuse | 100/100 | 194/200 | 865/1000 |

원분모·tiesfailure 독립CPU집계. N4대비 R/P/N lost0/gained0, W0-correct N862/886유지(24소실은N4도동일). 두schedule은 G/H/chi/eta0/4trial/판정/최종weight exact, 저장G로 네FP32후보를CPU재구성하여확인했다. 첫3trial Armijo탈락, 네번째수용, trainKL0.001120177268→0.001056148352. Dev1280.001329353168→0.001328344209이며 train과별도다.

Controller3313.261738→2798.450832초, 산술15.5379%감소. 단1회·Legacy→Reuse고정순서·pagecache비통제, teacherIO1449.834→1213.976초 등영향을분리할수없어 일반속도개선/인과기여율주장0. Current suffix3120→1248, externalCurrent H2D3120→0/sessionH2D5. 모든512문서/130235actualposition/fullvocab128256참여,EOS종료R11/Dev3.

실행 `5574f2c63a355043ba28c8e557383be3075a5e48`, tree `6707783f6b4042fa02346a393052f0554651a212`, archive SHA `47b71a42f52e543bfc6135064ac67fd8047d90d58923f1730481afa06e6cc99e`, effective lock SHA `01db7d4a9afb5bc6da9632967b32d4015613e1b5e7e6f324705858af7bae35af`. 준비a297039와분리,새nativefit0. 이번analysis는새2개CPU파일·exactSHAmanifest이며 frozen runtime변경0. 기존미게시own실행source를동일bytes로통합했다.

정본repo경로:
`experiment-reports/servers/server4/en-execution-reuse-r512-g256-20260919-v1/completed-review-v1/diagnostic-report-ko.md`

실제절대경로:
`/data/janghj/ODE-edit/local/en-execution-reuse/20260919-v1/completed-review-v1/worktree/experiment-reports/servers/server4/en-execution-reuse-r512-g256-20260919-v1/completed-review-v1/diagnostic-report-ko.md`

- report SHA `4ac7db3bf3a2f70ed6426459ae702f8c0f3b1539d0566136ffa535c9413d85ab`
- manifest SHA `b71d9d12e34b6db7ac200540bf6a1857084f587468c038055bdc321f72b72303`
- rooted receipt SHA `e7538bc47ed9644e8cac62c232849849e6600a95b7ab45af7d3df8bad87a614c`

CP3실물(각1058030053B)·W/M/context/RNG/ledger/historyreceipt/hash를CPU검산했고원자료보존. CPUfocused12PASS/최종반복12PASS;별도독립agentred0. HTMLrenderer미설치NOT_RUN, GFM/CSV/링크·manifest검사수행. 초기분석표의nullproof표시1건만정정했고수치불변. 전체512physicalAD/GPUcontinuation/nonemptyPast/B2는미실행/N/A,과거PENDING을소급PASS하지않았다.

최신main보존/nonforce게시후 TASK_COMPLETE_STOP. monitoring_active=false, automatic_resume=false. Sequential/B2/추가arm은 여전히 사용자별도승인필요이며등록0.
