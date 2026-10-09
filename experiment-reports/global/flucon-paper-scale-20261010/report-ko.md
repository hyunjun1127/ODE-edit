# FLU/CON 논문 표시 배율 및 완료 결과 통합

Instruction: `USER-GH-FLUCON-PAPER-SCALE-TABLE-REFRESH-20261010-R1`.
GH accepted turn: `01a121be-6160-7333-9b02-ec62ffef2bed`.

## 먼저 게시한 표시 교정

| 모델 / 관측 | 원 FLU (bits) | 원 CON (cosine) | README Flu ×100 | README Con ×100 |
| --- | ---: | ---: | ---: | ---: |
| Llama3 W0 / SH1 61768 | 6.352242334333923 | 0.24636896048599818 | 635.22 | 24.64 |
| Qwen FT W20 / SH2 61898 | 4.71017497777678 | 0.030072135827285053 | 471.02 | 3.01 |

위 4개 실제 셀과 모든 모델·FE_HISTORY 표의 열 단위를 교정했다. FLU는 정확도 %가 아니며 100 초과가 정상이다. CON도 정답률이 아니다. 반올림된 표 숫자를 확대하지 않고 원 summary에서 ×100 후 half-up 2자리로 표시한다. 원 metric JSON, W&B raw 키, source/가중치/참조 자산은 불변이다. 미측정은 DEFERRED/빈칸을 유지한다.

Qwen SH2 W0의 원값은 6.252105796227186 / 0.2591242773267912 (표시 625.21 / 25.91)이나, SH3가 제공한 선택 W0와의 조건 결속을 SH2가 확인하기 전에는 본표에 추가하지 않는다. factual W0 셀 출처를 변경하지 않는다.

[원 조사 기록](audit-source.md)은 수정 전 표를 기술한 역사 기록이다. 그 기록의 원래 상대 링크에 대응하는 게시 파일은 [재채점 코드](../../../audits/global/flucon-paper-scale-20261010/recompute.py)와 [Qwen 전체 CPU 재채점 결과](../../../audits/global/flucon-paper-scale-20261010/qwen-raw-recomputation.json)다. Llama W0 근거는 [기존 W0 보고](../w0-main-table-20261009.md)다. Qwen W0/W20 각각 2,000 case / 20,000 prompt 저장 텍스트의 재채점이지 pretrained generation 재실행 증거가 아니다. 배율 수정 뒤에도 FT의 471.02 / 3.01이라는 낮은 값은 남는다.

## 표시 API와 검산

`official.evaluation.generation.paper_display.paper_cell(raw_value, *, metric, raw_unit)` 및 `paper_generation(summary)`는 표시 전용이다. 후자는 raw 단위·유효 count를 검사하고 `Flu_paper_x100` / `Con_paper_x100` 문자열만 만든다. 이미 표시된 문자열/표시 단위는 입력으로 거부한다. scalar logger나 평가 수식은 변경하지 않았다. 기존 historical 보고 값은 raw provenance로 보존한다.

CPU `official.tests.test_generation_paper_display`: 5 PASS. `official.tools.verify`: 166 source SHA / 331 Python / external imports 0 PASS. GPU/forward/TF-IDF refit 0. 추가 owner 결과·등록은 아래 통합 상태와 별개다.

## 실제 담당 수락

| 서버 | accepted turn | 수신 상태 / 담당 |
| --- | --- | --- |
| SH1 | 01a121bf-d873-7a82-9654-01368ea03466 | 명시 OWNER_ACK; 완료 결과 검산 및 유효 저장 CF checkpoint 평가-only 실제 등록 진행 |
| SH2 | 01a121bf-dc12-7420-9754-c50086b87646 | 명시 OWNER_ACK; 완료 결과와 Qwen W0 호환성 검산 |
| SH3 | 01a121bf-d90c-7961-ac25-8e757c17d1a7 | 완료; 신규 적격 생성 값 0, 선택 W0 26,000행/47개 증거 검산·SH2용 provenance (069b9516) |
| SH4 | 01a121bf-d996-71d2-973c-3bb381fbe3ba | 완료; 신규 본표 적격 결과 0, Flu/Con 실측 0 (177dadf1) |

실제 전달 정본은 [envelope](../../../messages/head/2026-10-10-flucon-paper-scale-table-refresh.json)다. SH4 기존 60103은 raw/20commit을 재확인했지만 생성 미측정이다. migrated baseline·held-out500·tuning은 새 본표 값으로 승격하지 않는다. 현재 이 최초 게시 시점에 SH1 평가 job ID는 아직 회수하지 않았으므로 PENDING으로 표시하지 않는다. 이번 표시는 실험 완료 주장이 아니다. 후속 compact receipt에 따라 이 보고와 README 셀을 갱신한다.
