# FLU/CON 논문 표시 배율 및 완료 결과 통합

Instruction: `USER-GH-FLUCON-PAPER-SCALE-TABLE-REFRESH-20261010-R1`.
GH accepted turn: `01a121be-6160-7333-9b02-ec62ffef2bed`.

## 먼저 게시한 표시 교정

| 모델 / 관측 | 원 FLU (bits) | 원 CON (cosine) | README Flu ×100 | README Con ×100 |
| --- | ---: | ---: | ---: | ---: |
| Llama3 W0 / SH1 61768 | 6.352242334333923 | 0.24636896048599818 | 635.22 | 24.64 |
| Qwen FT W20 / SH2 61898 | 4.71017497777678 | 0.030072135827285053 | 471.02 | 3.01 |

위 4개 실제 셀과 모든 모델·FE_HISTORY 표의 열 단위를 교정했다. FLU는 정확도 %가 아니며 100 초과가 정상이다. CON도 정답률이 아니다. 반올림된 표 숫자를 확대하지 않고 원 summary에서 ×100 후 half-up 2자리로 표시한다. 원 metric JSON, W&B raw 키, source/가중치/참조 자산은 불변이다. 미측정은 DEFERRED/빈칸을 유지한다.

Qwen SH2 W0의 원값 6.252105796227186 / 0.2591242773267912를 **625.21 / 25.91**로 추가했다. SH2 최종 `fa770b9c` 검산에서 SH3의 모델·tokenizer 10개 파일 SHA/bytes, cold state 및 동일 first2K가 결속됐다. 생성 자체의 source/protocol/seed20261007/참조/2,000개 raw SHA는 SH2 provenance이며 SH3 factual W0 네 셀은 불변이다. cross-hardware generation parity는 주장하지 않는다. [최종 호환 영수증](../../../audits/servers/server2/flucon-paper-scale-20261010/w0-compatibility-final.json).

SH2가 단발 02:43:21 KST 검산한 완료 13개 factual/public-query 행은 현재 README 숫자와 다시 대조하여 불일치 0이다. 이미 반영된 결과이므로 새 완료 13개라고 세지 않는다. 미완료 Qwen CF 네 수정 chain의 생성 8칸은 `ING/PENDING`이 아니라 실제 승인 일정 **DEFERRED**로 바로잡았다. 해당 factual 4칸과 zsRE 상태는 같은 owner snapshot 기준이다. [24개 chain 및 별도 W0 입력](../../../audits/servers/server2/flucon-paper-scale-20261010/table-rows-final.json).

[원 조사 기록](audit-source.md)은 수정 전 표를 기술한 역사 기록이다. 그 기록의 원래 상대 링크에 대응하는 게시 파일은 [재채점 코드](../../../audits/global/flucon-paper-scale-20261010/recompute.py)와 [Qwen 전체 CPU 재채점 결과](../../../audits/global/flucon-paper-scale-20261010/qwen-raw-recomputation.json)다. Llama W0 근거는 [기존 W0 보고](../w0-main-table-20261009.md)다. Qwen W0/W20 각각 2,000 case / 20,000 prompt 저장 텍스트의 재채점이지 pretrained generation 재실행 증거가 아니다. 배율 수정 뒤에도 FT의 471.02 / 3.01이라는 낮은 값은 남는다.

## 표시 API와 검산

`official.evaluation.generation.paper_display.paper_cell(raw_value, *, metric, raw_unit)` 및 `paper_generation(summary)`는 표시 전용이다. 후자는 raw 단위·유효 count를 검사하고 `Flu_paper_x100` / `Con_paper_x100` 문자열만 만든다. 이미 표시된 문자열/표시 단위는 입력으로 거부한다. scalar logger나 평가 수식은 변경하지 않았다. 기존 historical 보고 값은 raw provenance로 보존한다.

CPU `official.tests.test_generation_paper_display`: 5 PASS. `official.tools.verify`: 166 source SHA / 331 Python / external imports 0 PASS. GPU/forward/TF-IDF refit 0. 추가 owner 결과·등록은 아래 통합 상태와 별개다.

## 실제 담당 수락

| 서버 | accepted turn | 수신 상태 / 담당 |
| --- | --- | --- |
| SH1 | 01a121bf-d873-7a82-9654-01368ea03466 | 완료 결과 11행 검산, 평가5개+collector 실제 release (1fb4cef6) |
| SH2 | 01a121bf-dc12-7420-9754-c50086b87646 | 완료 13행 재검산·W0 생성 분리 출처 호환 결속 (fa770b9c) |
| SH3 | 01a121bf-d90c-7961-ac25-8e757c17d1a7 | 완료; 신규 적격 생성 값 0, 선택 W0 26,000행/47개 증거 검산·SH2용 provenance (3a6c7ee9) |
| SH4 | 01a121bf-d996-71d2-973c-3bb381fbe3ba | 완료; 신규 본표 적격 결과 0, Flu/Con 실측 0 (177dadf1) |

실제 전달 정본은 [envelope](../../../messages/head/2026-10-10-flucon-paper-scale-table-refresh.json)다. SH4 기존 60103은 raw/20commit을 재확인했지만 생성 미측정이다. migrated baseline·held-out500·tuning은 새 본표 값으로 승격하지 않는다.

## SH1 실제 제출 및 완료 결과 통합

SH1 source `61ab70384bc909d359537f14a3bbb3691f5c9ce5`, execution lock SHA `6d3495b07337f48fd57dd50996b08ae76b7d86549223e7932dd56efcbf979db1`. 원본 submission과 공개 compact receipt를 직접 대조했다. held 검사/release 후 2026-10-10 02:51:52–53 KST 첫 snapshot은 전부 **PENDING**이다. GPU 결과/새 W&B readback은 NOT_OBSERVED이며 기다리지 않았다. cap4, 기존 job 변경0.

| 원 CF W20 checkpoint 소유 run | 평가 job name | 실제 ID | resource dependency |
| --- | --- | ---: | --- |
| Llama FT 61771 | official-s1-flucon-eval-llama3-ft | 62259 | 없음 |
| Llama SPHERE 61770 | official-s1-flucon-eval-llama3-sphere | 62260 | 없음 |
| Llama MEMIT-FE 61773 | official-s1-flucon-eval-llama3-memit_fe | 62261 | 없음 |
| GPT-J FE_HISTORY 61927 | official-s1-flucon-eval-gptj-memit_fe_history | 62262 | afterany:62061 |
| Llama FE_HISTORY 61928 | official-s1-flucon-eval-llama3-memit_fe_history | 62263 | afterany:62262 |
| GPU0 collector | official-s1-flucon-eval-collector | 62264 | afterany:62259:62260:62261:62262:62263 |

정확 CP 경로/전체 SHA/config/output 경로는 [SH1 등록 compact JSON](../../../audits/servers/server1/flucon-paper-scale-20261010/table-rows.json)에 있다. 별도 출력 root는 `local/official-baselines/server1/flucon-paper-scale-20261010/preparation-r2/runs/`이다. GH는 5개 평가의 Flu/Con 10칸만 PENDING으로 변경했고 해당 baseline factual 4칸은 보존했다. 평가를 이미 완료한 것으로 표시하지 않는다.

추가 완료 검산에서는 Llama 공개-query zsRE AlphaEdit61934의 E/G/Loc **95.18/91.40/31.10**, MEMIT-FE61936의 **14.81/13.57/0.56**으로 6칸을 채웠다. 다른 zsRE4행은 수치 불변이다. 별도 Llama FE_HISTORY61928의 factual 4칸은 Score/E/G/Loc **64.66/80.10/74.10/48.99**다. 이 variant를 native MEMIT-FE 행에 합치지 않았다. [11개 완료행·미반올림 수치·원 raw SHA](../../../audits/servers/server1/flucon-paper-scale-20261010/completed-rows.json).

미제출 범위: old Qwen61975는 context 오류로 제외, 수정62061은 owner inventory 당시 미완료. Llama historical MEMIT/AlphaEdit/BLUE CP는 이번 SH1 로컬 유효 후보에 없고 별도 원격 CP 전송은 수행하지 않았다. 미측정 값은 계속 DEFERRED/기존 빈칸이며 0이 아니다. 본표 historical 예외와 OURS 예외를 변경하지 않았다.

최종 변경량: 배율 교정 숫자4칸 + Qwen W0 생성2칸 + 새 Llama zsRE6칸 + 별도 Llama history factual4칸, 등록상태10칸, Qwen generation DEFERRED 정정8칸. SH2 완료13행과 SH1 기존 완료행은 원자료 근거로 재검산하되 중복 새 완료로 세지 않는다. [자동 표 검산](../../../audits/global/flucon-paper-scale-20261010/verify_display.py), [담당 수락·채택 기록](../../../audits/global/flucon-paper-scale-20261010/adoption.json).

SH1 공통 schema의 좁은 CF eval-only 변경을 GH가 읽고 display+caller CPU10 PASS를 재확인했다. SH1 회귀52 PASS와 중복 합산하지 않는다. 별도 GPU qualification은 여전히 비활성화이며 이번 평가 자체만 허용됐다. 원 CP/raw/W&B/모델·hparams 변경, 추가 삭제·취소, 신규 반복 monitor 모두 0이다.
