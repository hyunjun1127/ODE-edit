# Server4 job 상세 리뷰 완료

ACK nonce: `ODEEDIT-GH-SH4-ALL-JOBS-DETAILED-REVIEW-20260924-R1`.

CPU-only 리뷰를 마쳤다. Snapshot 2026-09-24 00:52:05 KST에서 현재 queue0,
최근 exact parent17개 및 과거 게시보고97개를 대조했다. Alpha94개 승인 family가
완료 산출물에 대응하며 후속7개는 FOLLOWUP_NOT_SUBMITTED다.
신규 model/GPU/evaluator/Slurm write/repair/삭제/원격전송0.

보고서: `experiment-reports/servers/server4/completed-jobs-review-2026-09-24-v1/report-ko.md`

- Report SHA256: `8f552b44482a809681370366c93590f947c1a7c8c1517593953a4f777ff09385`
- Analysis source: `29f3052b89e1bc85374bbcacd4ef98b17e115604`
- Analysis manifest SHA256: `442f25081e005f8238d3495f7791d6c28fc4cae2e67320fc432addf42fd5b616`
- Rooted receipt SHA256: `06cd1e0ca8e4ba1923487077d85cee4d7c225852ee6b68893c2ecd2422d813e1`
- Exact jobs: 50974, 50983, 51055, 51056, 51057, 51058, 51260, 52527, 52528, 52529, 52530, 52563, 52564, 52565, 52566, 52575, 52576.
- Alpha GPU allocation50,605초(14.056944GPUh), 최근 총61,565초; 최대동시2.

Native Current R/P/N: W50 100/190/620, W70 100/190/547,
W80 100/180/550, W90 100/186/523 (분모100/200/1000).
H56의 NS 차이는 각 entry 0/−1/0/+4개이며 PS/RS는 동일하다.
TF·joint·문항 lost/gained·geometry·KR·history·cost는 보고서/CSV에 분리했다.
최신 수치차 observer-only 정책을 parity PASS로 바꾸지 않았다.
Full numerical equivalence NOT_ESTABLISHED, GPU continuation NOT_TESTED,
HTML renderer 미설치 NOT_RUN. Owner audit+independent reducer이며 별도 agent red0.

원 runtime/raw/input12CP/실패/waiver/타task 유지.
Own-scope nonforce publication 외 추가 작업 없음.
Access helper는 core source/report에는 PASS였으나 명시 허용된
`runs/odeedit_server4_jobs_review_20260924/receipt.json`에서 exit7/NOT_PASS였다.
명시 envelope의 좁은 예외로 기록했고 공용 helper/정책/identity 수정은0이다.
TASK_COMPLETE_STOP; monitoring_active=false; automatic_resume=false.
