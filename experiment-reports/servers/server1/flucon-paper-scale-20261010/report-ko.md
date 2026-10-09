# SH1 Flu/Con 논문 배율 및 저장 W20 평가 등록

Instruction: `USER-GH-FLUCON-PAPER-SCALE-TABLE-REFRESH-20261010-R1`.

## 실제 등록

GPU 평가 5개와 GPU0 collector를 전량 held 검사한 뒤 release했다. 초기 snapshot(제출시각 2026-10-10 02:51:52–53 KST)은 모두 PENDING이다. GPU 완료/온라인 readback을 주장하지 않는다.

| 모델/방법 | 원 편집 job | 새 평가 job | 자원 dependency |
|---|---:|---:|---|
| Llama FT | 61771 | 62259 | 없음 |
| Llama SPHERE | 61770 | 62260 | 없음 |
| Llama MEMIT-FE | 61773 | 62261 | 없음 |
| GPT-J MEMIT-FE-HISTORY | 61927 | 62262 | afterany:62061 |
| Llama MEMIT-FE-HISTORY | 61928 | 62263 | afterany:62262 |
| CPU collector | — | 62264 | afterany:62259:62260:62261:62262:62263 |

실행 source `61ab70384bc909d359537f14a3bbb3691f5c9ce5`, official tree `24aef23257f1a9ecc78703347da786b37c654604`. 문서 publication commit과 실행 source는 별도다. server1 직접 USER cap4/local cap4, 기존 allocation/admitted DAG 포함 폭4. GPU당 CPU8/65536MiB/48h, collector GPU0/CPU8/24576MiB/4h. wall은 ETA가 아니다. owner/Command/WorkDir/script/argv/source/config/input/CP/reference/resources/dependency/Requeue0/exportNONE를 held 상태에서 검사했다.

실제 원본 영수증: `/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1/flucon-paper-scale-20261010/registration-r1/{submission,execution-lock,held-inspection,admission}.json`.

## 구현 및 최소 검사

- 기존 W20 checkpoint fullSHA, 원 identity, 20 commit/2000 요청, ordered stream 및 base/tokenizer/reference를 결속했다. 원 CP는 제자리 보존한다.
- 선택 가중치만 복원하고 native CAKE case-batched KV/full-prefix mask/global RNG/total100 경로로 전체2K를 한 번 생성한다. 같은 텍스트로 Flu와 Con을 계산한다. W0 재생성/편집/fit/별도 GPU qualification은 없다.
- 새 CF eval-only authority/profile 및 CP/source/query/reference identity를 공통 schema에 좁게 추가했다. all_seen/post generation 및 W20 progress만 허용한다. factual/fit/W0/current 거부, 기존 zsRE/enabled/deferred 규칙 유지.
- CPU 52 tests PASS. source166 SHA/Python334/external task imports0 PASS. schema 수정에 맞춰 SOURCES의 현재 SHA만 재봉인하고 원 upstream/과거 이력은 보존했다. 실제 GPU 평가/온라인 readback PASS와 구분한다.
- runtime weight/RNG 비변이와 원 CP/pointer 불변, 전체 count 검사 및 CPU independent collector를 연결했다. progress와 최종 점수를 분리하며 미완료 평균을 최종값으로 올리지 않는다.

## 표시 및 완료결과 단발 refresh

GH `paper_display` API를 채택했다. 미반올림 raw 평균에 100을 곱한 뒤 decimal half-up 소수2자리로 표시한다. raw JSON/W&B 값/키는 변경하지 않는다. Flu는 백분율이 아니며 Con은 정답률이 아니다.

기존 Llama W0 실제 raw를 CPU 재채점: 2000 requests/20000 prompts, Flu count2000/Con count2000. Flu `6.352242334333923` bits → **635.22**, Con `0.24636896048599818` cosine → **24.64**. 새 generation 관측이 아니다.

`audits/servers/server1/flucon-paper-scale-20261010/completed-rows.json`에 완료 factual/public-query 11행을 별도로 제공한다: Llama CF FT/SPHERE/FE 3행, zsRE6행, GPT-J/Llama history2행. 기존 검산 CF3/GPT-J history는 원 raw/terminal SHA를 재확인해 재사용했다. 새 완료 Llama history는 원 frozen evaluator의 CPU token/NLL/count 검산을 수행했다. zsRE6은 predicted/target token bits로 request-macro E/G/loc_ans를 독립 재집계했다. 이전 PENDING AlphaEdit61934/MEMIT-FE61936도 실제 최종 raw/receipt를 확인했다. 추가 model forward/복원/스케줄러 반복 조회는 없다.

새 W20 Flu/Con은 아직 **EVALUATION_REGISTERED_NOT_OBSERVED**다. 이전 DEFERRED를 숫자0 또는 새 실측으로 바꾸지 않는다. 원 Qwen61975는 잘못된 context로 제외, 수정62061은 inventory 당시 미완료로 제외한다. Llama MEMIT/Alpha/BLUE historical 예외와 PRICE FREE100은 유지하며 remote/다른 cohort CP를 새 local 관측으로 relabel하지 않는다.

README는 GH 단독 편집. compact `table-rows.json/csv`와 `completed-rows.json`에 source/config/CP/raw/cohort/분모/미반올림 수치 및 표시값을 분리했다. raw/text/token/tensor/CP 업로드·이동·삭제 및 기존 job 변경은 0. `NO_BROADCAST_NOT_REQUIRED`: 같은 서버의 기존 CP/reference를 읽고 compact Git 증거만 공유한다. 초기 인계 후 장기 monitoring/automatic retry 없음.

결과 publication은 main `1fb4cef609f8ef8530f28256694ecba727b89018`이다. 이후 GH direct 전달은 app-server 응답 timeout으로 `COMMUNICATION_HOLD`/수신 UNKNOWN이며 자동 재전송하지 않았다. Git 게시와 직접 수신 ACK/README 반영은 구분한다. 정확 전달 상태는 같은 audit의 `delivery.json`에 남겼다.
