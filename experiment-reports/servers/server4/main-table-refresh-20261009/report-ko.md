# Server4 완료 실험 본표 갱신 검산

nonce: `USER-GH-ALL-SH-COMPLETED-MAIN-TABLE-REFRESH-20261009-R1-SERVER4`.
accepted turn: `01a120fb-33b8-7c12-a301-020a9c507251` (own local session turn_context 확인).
2026-10-09 23:08 KST 한정 관측. 기준 main `99ecdba4`; README는 GH 소유이며 수정하지 않았다.

## 결과

새 eligible W20 완료 행 **0**. 기존 명시 예외 **1개** 재검산, 기존 본표 대비 수치 변경 **0**.

| 행 | job | 사실 상태 | 본표 조치 |
|---|---|---|---|
| Llama PRICE FREE100 / MEMIT / CF | 60103 | 실제 W20/2000, 20 commit, 26,000 raw 행 재집계 일치 | 기존 § 예외 값 유지 |
| Qwen CF FT | 61783 | FAILED, 1초, 원 output 없음 | server2 최신 replacement를 유지; 실패를 완료로 기록하지 않음 |
| Qwen 나머지 CF 5개 / zsRE 6개 | 61785/87/89/91/93, 61755/57/59/61/63/65 | CANCELLED, 최종 raw 없음 | server2로 이관된 행을 SH4로 중복 집계하지 않음 |
| 옛 Llama native 6개 | 60917–60922 | CANCELLED | 완료 행으로 복원하지 않음 |

## 기존 FREE100 예외의 독립 CPU 검산

원 final chunk 40개를 읽어 N은 true NLL < new NLL, R/P는 new NLL < true NLL의 strict preference를 직접 계산했다. 요청 순서가 fixed10k 앞 2000 case ID와 일치하고, endpoint W20 및 20 commit/terminal/state 일치를 확인했다. 원 raw는 local에 그대로 두고 파일별 SHA만 게시한다.

- Eff: 1994/2000 = 99.7%
- Gen: 3711/4000 = 92.775%
- Loc: 16441/20000 = 82.205%
- Score: 반올림 전 세 성공률의 조화평균 = 90.98196945746398%
- 기존 표 표시 90.98 / 99.70 / 92.78 / 82.21과 동일. old→new delta 전부 0.
- Flu/Con, zsRE는 미관측이며 null/빈칸 유지. 이 legacy prompt-pair 점수를 새 official request-macro나 새 cold rerun 완료로 재명명하지 않는다.

summary SHA `5acf7176ef23aef08a5d3708d600bb16621b0823a0fc0cb08f1bf83f1ad9a331`.
source `5226337121fd2c90c595f26297c9927400f8f0af`.
정확 config/cohort/terminal/commit/raw SHA 및 실제 job name은 audit JSON에 있다.

## 범위와 보존

known own official12 + 옛 Llama6 + 명시 FREE100 예외에 대한 exact accounting 1회, 원 output 존재·config SHA 확인, published own historical inventory를 사용했다. 전체 filesystem의 모든 미등록 실험을 전수 점검했다는 뜻은 아니다. 다른 historical/튜닝/Q3/heldout/GPT2/W0/qualification/collector는 새 main으로 승격하지 않았다. own 완료 zsRE는 없으며 다른 서버 replica 및 재평가 job을 SH4 결과로 보고하지 않는다.

`audits/servers/server4/main-table-refresh-20261009/table-rows.json` 및 CSV는 GH 통합용이다. Qwen 행의 상태는 **원 SH4 job** 상태이며, server2 최신 job 상태를 덮어쓰라는 요청이 아니다.

GPU/forward/restore/fit/Slurm 변경/CP 전송·삭제/온라인 history 수정 0. source/raw/CP/기존 jobs KEEP. NO_BROADCAST_NOT_REQUIRED: compact reducer/metadata만 Git 게시하며 raw는 local 보존. 새 monitor/자동 retry 없음.
