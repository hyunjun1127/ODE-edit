# Server3 완료 main-table 갱신 회수

최신 origin/main `99ecdba4` README 및 control/main-results-policy.json 기준, nonce USER-GH-ALL-SH-COMPLETED-MAIN-TABLE-REFRESH-20261009-R1-SERVER3. 관측 2026-10-09T14:05:57.047896+00:00. 본표 신규 완료 적격 행 **0개**, Q3 별도 완료 행 1개, L1 실행 중 1개다. README는 GH 단독 소유이므로 수정하지 않았다.

| 실행 | 상태 | Score | Eff | Gen | Loc | 본표 처리 |
|---|---|---:|---:|---:|---:|---|
| Qwen 61813 qwen-price-q3-eot-2k | W20/2000 COMPLETE | 83.770638 | 98.050000 | 89.950000 | 68.985000 | 기존 Q3 별도 보고 유지 |
| Llama 61821 llama-price-L1-2k | RUNNING; W20 없음 | — | — | — | — | 상태만, 자동승격 없음 |

Qwen 원 raw26000행(R2000/P4000/N20000)을 CPU 독립 재집계하여 이전 보고와 네 지표 delta=0을 확인했다. case/row/token identity·finite·strict/count·20commit/19join/100H 검산 유지. full raw SHA/bytes는 qwen/inventory.json, source/config/cold/cohort는 table-rows.json. Flu/Con은 DEFERRED이며 W20 snapshot을 복원하거나 삭제하지 않았다. Q3-selected final은 tuning 자체는 아니지만 일반 갱신 지시로 기본 PRICE에 승격하지 않는다.

Llama는 W0/initial/profile만 확인했고 단발 snapshot에서 commit0, result/terminal 없음. 실제 accounting RUNNING을 성능 완료로 표시하지 않았다. 향후 종료를 기다리거나 polling하지 않는다. 선택 L1 source 50a9110b86a0846710804d58bf3ea783762ff25f, config f38a21d6682d22a6a22c6d73a0ec605a1f0ef9f6ce6a88b0c11225730861eb83. 기존 Llama60103 FREE100 본표 예외는 그대로 유지한다.

own 완료 zsRE checkpoint/공개-query 재평가 결과는 없다. 다른 서버 baseline migration·replica를 재집계하지 않았다. 과거 MEMIT-H/MEMIT-HJ/JLZ-v10/v12/pilot는 inventory에 제외 이유를 남겼으며 current official/PRICE로 재명명하지 않았다. 미제출 baseline과 W0/collector/qualification을 main으로 대체하지 않았다.

재현: `python3 /data/janghj/ODE-edit/local/official-baselines-20261008/worktree/audits/servers/server3/main-table-refresh-20261009/qwen/reduce.py` 뒤 `python3 /data/janghj/ODE-edit/local/official-baselines-20261008/worktree/audits/servers/server3/main-table-refresh-20261009/build_rows.py`. 원 raw와 모델/CP local KEEP. owner CPU audit, 별도 reviewer 없음. no GPU/model-forward/newjob/Slurm mutation/online history rewrite. compact Git만 공유(NO_BROADCAST_NOT_REQUIRED). 기존 실행은 그대로다.
