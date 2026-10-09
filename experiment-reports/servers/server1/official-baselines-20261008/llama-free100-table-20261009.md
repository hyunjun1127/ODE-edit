# Llama Ours CF: 이전 FREE100 결과 반영

사용자 지시: “llama ous의 CF table엔 이전에 돌렸던 Free100 으로 놓자”.
새 official 실행이 아닌 이전 server4 PRICE MEMIT writer 결과를 명시적으로 사용한다.

- task: `jlz-price-cap-base-repair-2k`, arm `LLAMA_FREE100`, cell33, job **60103**.
- 정본 inventory: `experiment-reports/global/2026-10-08-ours-baseline-results/inventory.csv`, W20_COMPLETE/20 commits.
- endpoint: batch20/post/all_seen, state_edits=2000, requests=2000.
- 원 summary: server4 `/data/janghj/ODE-edit/local/jlz-price-cap-base-repair-2k/method-metrics-20261007/LLAMA_FREE100/batch-20/post/summary.json`.
- 원 SHA256: `5acf7176ef23aef08a5d3708d600bb16621b0823a0fc0cb08f1bf83f1ad9a331`.
  2026-10-09 SSH read-only full hash가 기존 published metrics.csv의 SHA와 일치했다.

| 지표 | 분자/분모 | 정확한 성공률(%) | 표 표시 |
|---|---:|---:|---:|
| Eff (R) | 1994/2000 | 99.7 | 99.70 |
| Gen (P) | 3711/4000 | 92.775 | 92.78 |
| Loc (N) | 16441/20000 | 82.205 | 82.21 |
| Score | 반올림 전 세 성공률의 조화평균 | 90.98196945746398 | 90.98 |

기존 strict NLL preference의 prompt-pair 집계이며 새 official request-macro 재평가로
relabel하지 않는다. 표시만 decimal half-up. Flu/Con 및 zsRE 결과를 추정하지 않고 빈칸 유지.
원 source/raw와 모든 job은 변경하지 않았다. 새 forward/fit/복원/전송/삭제는 없다.
