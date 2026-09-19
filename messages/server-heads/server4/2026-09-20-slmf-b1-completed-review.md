# GH direct handoff — SLMF B1 완료 CPU 리뷰

Instruction `ODEEDIT-S06-SLMF-B1-COMPLETED-DETAILED-REVIEW-SH4-V1`
Nonce `ODEEDIT-GH-SH4-SLMF-B1-COMPLETED-REVIEW-20260920-R1`.

정확51058 COMPLETED/0:0, B1_COMPLETE_USER_LIMIT. 네 arm raw 독립집계 모두 RS100/100·PS194/200·NS865/1000; N4 대비 성공 ID lost/gained0. EN-KL-Q trial4 수용, 실제 norm0.00849720634. DEC-LINE/CUM은 네 trial 모두 FP32_NO_MOVE, native fallback. CUM B1→S3 gate FAIL(비영보정·choice/Phi개선 실패), S3/S10 제출0.

W0 N886→865: grosslost24/gained3, 보정후 회복0. EN R512 KL0.00112039035013→0.00105820016472; selected R512 choice/Phi는 NOT_MEASURED. Dev 신규flip2/회복2/Phi증가를 함께 보존했다. 수치검증 NOT_ESTABLISHED, FD WAIVED_USER_DIRECTED; noCP/exactresumeNOT_AVAILABLE.

비용: 새51058 parent3629 GPU-sec, program3597.552280초. Prior failed/T0 parent1113초는 별도. 원 writer 총timer invalid7403588.902935초는 source-backed RCA로 제외하고 유효한 상위기전216.898656초를 사용했다.

보고서 repo: `experiment-reports/servers/server4/single-layer-mechanism-first-20260919-v1/completed-b1-review-20260920-v1/report-ko.md`.
절대경로: `/data/janghj/ODE-edit/local/single-layer-mechanism-first/20260919-v1/completed-b1-review-20260920-v1/worktree/experiment-reports/servers/server4/single-layer-mechanism-first-20260919-v1/completed-b1-review-20260920-v1/report-ko.md`.
보고서 SHA256 `f1b15aa5103923be7228863a124237f471ce8537808732a99352a9b7586e1064`.
Analysis manifest SHA256 `c59400bacef6f88c0d385a556287cde4f85caac934fd9dee7ea5465653cb6d82`.
Rooted receipt SHA256 `9d999df531faeb3f87558f217aa8fdea943c8ed9bb84833f20cac9b37ad43234`.

실행 source5f790856/treea5b08922/lock6a14ebaf는 불변. 분석은 새 dedicated branch, base0150da2; GH ACK commitb56c8f7을 보존해 통합했다. 최종 mainHEAD/tree는 게시 완료 direct 회신에서 제공하며 이 파일에 자기참조 hash를 넣지 않는다.

CPU20 PASS, 원raw598파일fullSHA불변, tensor522파일검산, 표16개HTML실제렌더/링크/열, PNG2장byte재현·육안확인, raw-free 검산 완료. Sharedhelper는 새WT local session설정부재 및 명시허용runs경로 거부를 NOT_PASS로 기록했고 수정하지 않았다.

이번 GPU/model/eval/Slurmwrite0, 다른arm/task/scheduler조회0, SH2held51071불변. `TASK_COMPLETE_STOP`, monitoring_active=false, automatic_resume=false. 별도재승인/중복GH감사요청0.

