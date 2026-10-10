# Server2 완료 본표 갱신용 CPU 검산

nonce `USER-GH-ALL-SH-COMPLETED-MAIN-TABLE-REFRESH-20261009-R1-SERVER2`를 수락했다. accepted turn `01a120f9-c8e6-7063-afc3-9d5e32e1a0dc`, session `01a0493a-074c-7f91-9a13-769116326fef`. CWD `/mnt/raid5/janghj/ODE-edit`, origin `hyunjun1127/ODE-edit` 확인, 동일 nonce 선행 결과 없음. 전용 non-main WT에서 main `99ecdba4`의 README 및 main-results-policy를 읽었다. README는 GH 단독 편집이다.

## 결과

관측 시각 **2026-10-09 23:05:48 KST**. 정확한 own job 목록에 대해 accounting 1회 및 queue 1회만 읽었다. 최신 replacement를 우선한 본표 행 24개와 구 GPT-J zsRE 편집 완료 provenance 6개를 별도 inventory했다.

새로 반영 가능한 Qwen zsRE 결과는 다음 두 개다. 단위는 %이며 Loc은 W0 agreement가 아니라 **loc_ans 정답 token의 요청별 평균**이다.

| Method | job / actual name | Eff | Gen | Loc |
|---|---|---:|---:|---:|
| MEMIT | 61956 / s2-qwen25-zsre-memit-gpu | 37.606352813852816 | 36.68829906204906 | 30.025379311583595 |
| AlphaEdit | 61960 / s2-qwen25-zsre-alphaedit-gpu | 85.14393217893218 | 78.60837662337663 | 30.03645365808373 |

두 행 모두 source `5503935821b0ececb4aef09a5bccb5308879a6b5`, W20/2000 완료 receipt·20개 commit fullSHA·config·source·ordered stream을 검산했다. 실제 저장 predicted/target ID 비교를 token correctness와 대조하고 각 요청 내부 평균→2000 요청 평균을 독립 재계산했다. rewrite/paraphrase/neighborhood token 분모는 **6691/6691/11476**, 총24858. 고정 CPU query-proof SHA와 실제 W20 work의 query SHA가 일치하며 원 summary와 차이 <1e-10pp. 이전 표의 PENDING에서 실측 완료로 바뀌는 것이며 이전 native 수치와의 수치 delta를 꾸미지 않는다. 논문 pretrained 출력 bitwise parity 주장은 아니다.

기존 GPT-J CF FT/MEMIT/AlphaEdit/BLUE/FE/SPHERE `61650/61725/61778/61779/61780/61781`은 실제 raw fullSHA·20 commit·2000 ordered occurrence·strict true/new NLL 비교를 재검산했다. 기존 E/G/S/Score 값과 delta0, prompt-pair 분모2000/4000/20000. Flu/Con은 계속 **DEFERRED**, 미측정0 금지. 총 numeric-eligible 8개 중 신규 숫자 2개, 기존 확인 6개다.

## 숫자로 승격하지 않은 행

- GPT-J zsRE eval-only `61942–61947`: 모두 PENDING, 새 공개-query 결과 아직 없음. 옛 편집 완료 `61726/28/30/32/34/35`의 token-prefix/W0 agreement 수치를 새 점수로 재명명하지 않았다.
- Qwen zsRE FT `61900`: scheduler 및 raw/20commit 편집 완료는 확인했으나 구 token-prefix source `69bfbb2c`이다. **공개-query 재평가 필요/미등록**으로 구분하고 이번 본표 numeric-eligible 제외. Slurm PENDING인 새 평가가 있다고 표현하지 않는다.
- Qwen CF FT `61898`, CF BLUE `61962`, zsRE BLUE `61964`, zsRE FE `61968`: RUNNING, 최종 수치 없음. 특히 CF BLUE dependency는 앞선 사용자 지시에 의해 해제된 상태이며 이번 refresh에서는 변경하지 않았다.
- 나머지 Qwen CF MEMIT/AlphaEdit/FE/SPHERE `61954/61958/61966/61970`, zsRE SPHERE `61972`: PENDING.
- W0/collector/qualification을 새 본실험으로 세지 않았다. PRICE/tuning/Q3/history 변형/타서버 replica/GPT2 역사 자료를 본표로 새로 승격하지 않았다.

정확 source/config/cohort/raw SHA·분모·상태·jobname·timestamp·old→new는 `audits/servers/server2/main-table-refresh-20261009/table-rows.json` 및 CSV에 있다. 재현 가능한 CPU reducer `review.py` SHA도 JSON에 봉인했다. 상태는 이 한정 snapshot이며 이후 자연 진행을 최신 조회한 것으로 주장하지 않는다.

## 보존·제약

GPU/forward/모델 로드/checkpoint payload 로드/복원/fit/평가 재실행/온라인 조회·history 수정/Slurm mutation 모두0. 원 CP/raw/frozen source 및 실행 중 작업 KEEP. raw/text/token/tensor/secret Git0. `NO_BROADCAST_NOT_REQUIRED`; compact JSON/CSV/report만 게시한다. 실제 W&B remote history 전달 성공은 이번 CPU 결과 검산으로 주장하지 않는다. 반복 monitor/heartbeat/자동 retry 없음.
