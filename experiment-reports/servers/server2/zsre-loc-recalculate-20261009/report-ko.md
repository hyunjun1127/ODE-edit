# server2 zsRE Loc CPU 재집계

GPT-J 완료 zsRE 6종의 W20/2000 원자료를 독립 재집계했다. Loc은 loc_ans target token 정확도의 요청별 평균이며 W0 prediction agreement는 보조지표로 보존한다. summary 단순 rename이나 token micro 평균이 아니다.

| 방법 | job | Eff | Gen | 이전 W0 agreement | 새 Loc |
|---|---|---:|---:|---:|---:|
| FT | 61726 | 23.146349 | 17.947837 | 0.518387 | 0.624961 |
| MEMIT | 61728 | 93.607033 | 88.954734 | 55.924492 | 30.868701 |
| ALPHAEDIT | 61730 | 99.788690 | 96.551140 | 84.758091 | 27.993357 |
| ALPHAEDIT_BLUE | 61732 | 99.848333 | 95.805754 | 86.491917 | 28.900281 |
| MEMIT_FE | 61734 | 29.228211 | 27.671343 | 29.456310 | 8.459409 |
| SPHERE | 61735 | 99.767665 | 96.386534 | 86.227639 | 28.055999 |

각 행 2000 requests, Eff/Gen 각각 5557 tokens, Loc 9694 tokens. 실제 predicted/target ID 비교를 saved correctness bits와 대조했고 source/config/checkpoint identity, ordered cohort, 원 raw fullSHA/bytes를 결과 JSON에 결속했다. Specificity_loc_ans summary는 독립 계산의 교차검산에만 사용했다.

범위: 기존 server2 official GPT-J 완료 zsRE 6 main. 이번에 등록한 Qwen12는 완료 결과에 포함하지 않았다. 저장소 server2 결과/제출 메타데이터에서 추가 완료 zsRE ours 결과는 확인되지 않았으며, 다른 서버나 tuning을 main으로 승격하지 않는다. 과거 gptj-results.json은 역사 영수증으로 보존하며 해당 표의 Loc만 본 결과로 대체한다.

CPU macro/micro control PASS 및 6종 raw 검산 PASS. CF 변경0, GPU/model load/forward/job mutation/online history rewrite0. 원 tokenization의 논문 완전동등 또는 paper reproduction PASS를 주장하지 않는다. GH가 README를 통합하며 SH2는 README를 수정하지 않는다.

Reducer SHA256: 0c17f3648ed8de5119e61088954e6b7000f3ef58895277d503256f83f9793ef5
