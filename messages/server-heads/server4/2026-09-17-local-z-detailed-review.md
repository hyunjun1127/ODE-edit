# Local-z 7-arm CPU 상세 리뷰 완료 인계

Instruction ODEEDIT-S06-LOCAL-Z-SEVENARM-DETAILED-REVIEW-SH4-V1.
준비48679와 과학48680_[0-6] 모두 COMPLETED/exit0:0,7/7×1000 요청 완료. 신규GPU/모델/evaluator/teacher/Slurm변경/rsync0.

정본: experiment-reports/servers/server4/local-z-adaptive-allocation-seq1000-2026-09-16-v1/completed-review-20260917-v1/diagnostic-report-ko.md
SHA f28df155b881df44124cbe13984408274a2c6562316ccc9fffd4e9b5bbdb3641.
Analysis manifest SHA d48675f8a6f13e183f1bc849462581fe104e62c8e4dcdf062d876c487b5b4f90.
Rooted receipt SHA 07f28a310a7aa64c57039c6c078fa2c44d532d4fd66a3425f1de562cfea393a2.

실행32a92ad6f3fff2f258d8778f3936d152e975ac1b/tree75a96b2e2122d2be6b096af12c22df8a4365a8e1과 분석e91ba03ffcc4ac810038d03de0aa9a535a804f57은 별개다. runtime 원본불변, 기존 CAKE/cap 독립 링크보존.

| Arm | RS/1000 | PS/2000 | NS/10000 |
| --- | --- | --- | --- |
| N4 | 999 | 1934 | 8026 |
| REFIT4 | 999 | 1929 | 8187 |
| L75 | 1000 | 1919 | 8227 |
| T75 | 997 | 1942 | 7161 |
| L4D | 999 | 1922 | 8063 |
| LD | 999 | 1909 | 8310 |
| TD | 999 | 1934 | 8026 |

독립NLL/selector/Past검산:70commit,63state links,190후보,110append 일치.21CP/84tensor CPU검산, B2–B10 171후보weight 재구성hash 일치. 실제12000target/203663Adam/215663loss/150solve.
LD는9회(.75,.5),1회N4; TD는10회ownN4. TD와N4는전10batchW4/W8/M4와최종NLL동일, M8은상이하다. LD−N4 PS lost48/gained23, NS lost120/gained404. T75−L75 PS+1.15pp와NS−10.66pp를함께보존한다.

과학67637GPU초 + 준비2895GPU초, review0. 기존teacher98초는재사용비용으로별도다. 운영이탈: 제출%1/cap1이나실제지정job최대2GPU/19428초중첩. 원인/변경자미기록이며이번에수정/추가조회하지않았다. cap준수·controlledspeedup을주장하지않는다.

26 CPU tests/Markdown tables HTML렌더/링크/CSV/4PNG byte재현 통과. 별도reviewer agent는미사용이고같은SH4의독립구현reducer및자체검산이다. Missing: B1전체후보W0재구성, finalizerK/Gram, candidate공식RPN, nativeclamp-hit, GPUoff-on. 기존PENDING관측을소급PASS하지않았다.

Own-scope source/report nonforce main 게시 후 최종remoteHEAD/tree를local main-publication receipt 및direct handoff에기록한다. TASK_COMPLETE_STOP / automatic_resume=false / monitoring_active=false. 후속선택·실험·다른pause재개없음.
