# SH1 완료 결과 및 SH2 FLU/CON 입력 인계

승인 `USER-GH-S1-S2-BASELINE-REFRESH-S2-FLUCON-20261010-R1`.
정본 `92704580261c4f4b29d98d5a6870d2a967d0a34e`, envelope SHA `b060543a28f1a178c1c3862e4681fc2c33d2595657c6a7a7df03fcffae6fd884` FULL_READ.
session `01a04939-f93a-7b50-bca0-65438eab2062`, root `/mnt/raid5/janghj/ODE-edit`, origin `hyunjun1127/ODE-edit` 확인.
accepted turn `01a124dc-4d21-70f0-9fa7-94452f671f11`. 수락과 CPU 검산/게시/전달 상태를 구분한다.
전용 non-main `codex/server1-refresh-s2-flucon-20261010` WT에서 own audit/report만 변경했다. README는 GH sole integration.

## 단발 상태 / 중복 방지

관측 2026-10-10 08:10:08 UTC. server1 combined cap3 유지, actual allocation2 / admitted DAG폭2.

| job | 범위 | 단발 scheduler 상태 | 인계 판단 |
|---|---|---|---|
| 62581 | GPT-J FE_HISTORY CF W20 FLU/CON | COMPLETED | COMPLETE/raw CPU 검산 후 수치; SH2 중복 금지 |
| 62582 | Llama FE_HISTORY CF W20 FLU/CON | COMPLETED | COMPLETE/raw CPU 검산 후 수치; SH2 중복 금지 |
| 62583 | corrected Qwen FE_HISTORY CF W20 FLU/CON | RUNNING | 기존 평가 유지, SH2 중복 금지 |
| 62584 | 위3평가 CPU collector | PENDING | 기존 유지 |
| 62529 | Llama FE author CF | RUNNING | 18 committed batches 관측, W20 없음 |
| 62530 | Llama FE author zsRE | PENDING | 아직 완료/점수 없음, CF generation 비대상 |

기존 native Llama FT/SPHERE/MEMIT-FE 평가62259/62260/62261은 완료 raw/terminal/collector 검산 SHA가 같아 재사용한다.
현재 SH1에서 SH2로 새 평가 등록할 **유효하며 미등록인 local CF W20 CP는0개**다.
historical MEMIT/AlphaEdit/BLUE remote CP는 현재 local/current-cohort 복원 가능으로 승격하지 않았다. 기존 표의 명시 historical 예외는 유지한다.
별도 `historical-server2-references.json`에 이미 server2에 있는 본표 historical 42658/42657/39283_1의 정확한 기록 경로/bytes/SHA와 순서 대조 보고서를 연결했다. SH2가 실제 현물·원source/config/loader를 검산할 후보3개이며 SH1의 fresh payload PASS가 아니다. 이전 salted-hash campaign의 다른 네 CP로 바꾸면 안 된다.
원본 CP/context/config/raw와 기존 job은 변경하지 않았다. 새 GPU/forward/fit/CP load/submit/cancel/전송/삭제는 전부0.

## 원자료 검산

기존 완료 factual12행은 exact config/endpoint/terminal SHA와 이전 CPU 검산을 결속한다.
zsRE 6개는 frozen 공개-query evaluator/loader lock/전체2K parity evidence를 확인하고 저장 predicted/target token correctness에서 요청별 평균 후 요청 간 평균을 다시 계산한다.
Efficacy/Generalization/loc_ans macro는 기존 값과 일치. 요청2000, Llama E/G/Loc token분모6035/6035/12465, missing0.
W0 prediction agreement/token micro/옛 tokenizer-prefix 결과로 대체하지 않는다. CPU token/query 검산은 pretrained forward parity가 아니다.

새 완료 GPT-J/Llama generation은 원 checkpoint full SHA/bytes, final pointer/identity, 원20commit/2K 순서, source/config, frozen evaluator/reference, COMPLETE를 확인한다.
원 per-case text/token raw를 기존 CPU native verifier에 reference 자산과 함께 제공해 completeness/work/분모/점수를 검사한다. 별도 모델 forward는 없다.
Flu는 bits, Con은 cosine 원값을 유지하고 표시만 원평균×100 후 decimal half-up2자리. 결측/DEFERRED/진행중은0으로 채우지 않는다.
수치와 정확 path/bytes/SHA/provenance는 `table-rows.json` 및 `inventory.json`에 있으며 CSV는 GH 표 통합용이다.

검산 완료: factual/eval-only 완료12행과 author 미완료2행, 총14행이다. 새 완료 generation 두 개 모두 reference-bound per-case CPU 재채점 PASS이며 요청2000, prompts20000, Flu/Con valid2000 및 missing0이다.

| model / history 평가 job | Flu raw bits | Con raw cosine | Flu 표시 ×100 | Con 표시 ×100 |
|---|---:|---:|---:|---:|
| GPT-J / 62581 | 5.334287172759865 | 0.01054845862757061 | 533.43 | 1.05 |
| Llama / 62582 | 5.273934747386175 | 0.07580516374408931 | 527.39 | 7.58 |

이 표시는 백분율 정확도가 아니다. 원 raw/W&B는 변경하지 않았다. Qwen62583의 final 점수는 미관측이다.

## 입력/전달

`inventory.json`의 `SH2_inputs`에 CP exact member, 원 run/config/source/identity, cohort/stream, loader SHA, reference/protocol/seed 및 기존 eval job ID를 묶었다.
완료 또는 이미 등록된 CP는 `SH2_submit=false`; author CF는 `NO_W20_CHECKPOINT`다. transfer allowlist를 만들거나 payload를 복사할 필요가 없어 NO_BROADCAST_NOT_REQUIRED.
기존 W20 CP는 consumer 보존 정책대로 KEEP. 새 공통 helper/수학/과학 source를 변경하지 않았다.
SH2 중간 dedup 전달은 동일 accepted turn `01a124db-95bd-7231-981d-429bbc98305c`에 transport accepted; 실제 owner ACK와 구분한다.
historical 경로 전달도 같은 turn에 transport accepted이며 기록 source는 main `88083ae3` (own member commit `252b3c16`)이다. 독립 현물 검산 완료를 주장하지 않는다.
최종 compact 자료를 main 비강제 게시 후 SH2/GH에 전달한다. 장기 GPU 대기/recurring monitor/자동 retry 없음.
