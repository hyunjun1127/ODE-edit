# SH1 완료 생성 지표 및 후속 평가 등록 통합

Nonce: `USER-GH-S1-S2-COMPLETED-TABLE-S1-FLUCON-CAP3-20261010-R1`.
SH1 accepted turn `01a1236d-9293-73b0-aff5-7e2789e84602`, publication
`f74dcdae5938b6ebab4a8a5944b5644ee948e079`의 owner 결과를 통합했다.

## 완료 지표

| Llama CF method | 생성 job | raw FLU / CON | 논문 표시 FLU / CON |
| --- | --- | --- | --- |
| FT | 62259 | 4.49504268731324 / 0.029030793469636437 | 449.50 / 2.90 |
| SPHERE | 62260 | 6.192038526176254 / 0.3364011388508934 | 619.20 / 33.64 |
| MEMIT-FE | 62261 | 4.200683524775806 / 0.08397526788376594 | 420.07 / 8.40 |

각 2,000 case / 20,000 prompt, FLU/CON 유효 분모 각각2,000, missing0.
GH는 owner compact JSON의 SHA, 원 미반올림 평균×100→half-up2 및 README 셀을 검산했다.
원자료 전량 채점은 기존 frozen CPU collector의 reference-bound 완료 증거를 SH1이
원 endpoint/terminal/source SHA와 결속해 재사용했다. GH의 새 전량 raw 채점이나 GPU 실행이 아니다.
zsRE6행은 저장 predicted/target request-macro 재검산 결과가 기존 값과 같아 숫자를 유지했다.
각2,000 requests, E/G/Loc token 분모6035/6035/12465이며 Loc은 loc_ans 정확도다.

## 후속 실제 등록

| 모델 / MEMIT_FE_HISTORY | 원 W20 job | 새 eval-only job | dependency | 초기 상태 |
| --- | --- | --- | --- | --- |
| GPT-J | 61927 | 62581 | 없음 | PENDING |
| Llama3 | 61928 | 62582 | 없음 | PENDING |
| Qwen2.5 corrected | 62061 | 62583 | afterany62581 | PENDING |

GPU0 collector62584는 세 평가 모두 afterany. 정확 job name은 submission 및 README에 기록.
실행 source `37094d92d70d0499304da9f7b82ab59340c57232`, lock
`ba5262d003d6c6c125f420885706189a874409ae442b83880407845de2ca4437`.
GPU당1GPU/8CPU/65536MiB/48h, collector GPU0/8CPU/24576MiB/4h.
held owner/source/config/checkpoint/resources/dependency 검사 후 release했다.
Server1 cap3, admission 기존allocation1 및 새 DAG폭3. 기존 job 변경0.
취소62262/62263은 보존하며 잘못된 context61975/실패61929를 재사용하지 않는다.
author CF62529는 10:30:05 KST RUNNING/W20 없음, zsRE62530은 PENDING이므로
새 generation 평가 대상이 아니다. 새62581–62583의 GPU/온라인/생성 완료는 아직 미관측이다.

## 근거 및 변경 경계

- [SH1 원 보고서](../../servers/server1/completed-table-flucon-cap3-20261010/report-ko.md)
- [원 table rows](../../../audits/servers/server1/completed-table-flucon-cap3-20261010/table-rows.json): SHA256 `9ff004351c8e6012d13284b9404a35d57c53cba2657f3deecb77f2aa0d0c3f2e`.
- [실제 등록/config·checkpoint SHA·출력 경로](../../../audits/servers/server1/completed-table-flucon-cap3-20261010/submission.json).

README 변경은 Llama FLU/CON6셀, history 평가상태6셀, author CF 상태1셀과 설명이다.
CF factual, zsRE, 다른 모델 수치 및 역사 예외는 불변이다. raw/W&B/checkpoint/frozen source 변경0,
GH scheduler 조작0, 장기 모니터0. Server2의 앞선 게시 결과와 cap3도 유지한다.
