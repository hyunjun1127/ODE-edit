# 완료 baseline 및 zsRE 검산 — GH main 표 통합

Nonce: `USER-GH-ALL-SH-BASELINE-COMPLETED-ZSRE-AUDIT-20261010-R1`.
GH는 네 서버의 명시 owner ACK와 compact 결과를 회수하고 README 셀을 통합했다.
상태는 2026-10-10 약07:37–07:38 KST의 owner 단발 관측이며 이후 실시간 상태를 뜻하지 않는다.

## 새 완료 셀

| 서버 | 모델 / dataset / method | 실제 job | Score | Eff | Gen | Loc | Flu ×100 | Con ×100 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SH2 | Qwen CF MEMIT | 62073 | 62.23 | 68.50 | 65.83 | 54.30 | DEFERRED | DEFERRED |
| SH2 | Qwen CF BLUE | 61962 | 77.98 | 97.70 | 96.75 | 55.86 | 602.99 | 37.43 |
| SH2 | Qwen CF MEMIT-FE | 62077 | 50.59 | 51.05 | 51.23 | 49.53 | DEFERRED | DEFERRED |
| SH2 | Qwen CF SPHERE | 62079 | 83.73 | 99.40 | 97.70 | 64.37 | DEFERRED | DEFERRED |
| SH2 | GPT-J zsRE MEMIT | 61943 (원 편집61728) | — | 93.52 | 88.86 | 30.81 | 해당 없음 | 해당 없음 |
| SH1 | Qwen CF MEMIT_FE_HISTORY (별도 표) | 62061 | 49.43 | 48.45 | 48.18 | 51.83 | DEFERRED | DEFERRED |

본표 신규21개 numeric 셀 + 별도 history4개 셀이다. CF 비율은 owner가 검산한 균일 R/P/N 분모
2000/4000/20000의 정수 분자에 정합화한 뒤 decimal half-up 2자리로 표시한다.
Score는 반올림 전 raw 성공률의 조화평균이며 `Score_AlphaEdit_display`로 바꾸지 않는다.
BLUE Flu/Con은 raw `6.029855922995685` bits / `0.37431408204466576` cosine에 100을 한 번 곱한다.
W&B/원 JSON 단위는 변경하지 않았다. 과거 오류 context61956/61960/61968/61975는 복구·승격하지 않았다.

## zsRE 검산 범위

완료13개 행(Llama6, GPT-J5, Qwen2)은 실제 frozen evaluator/public AST lock/tokenizer/전체 stream의
query·target parity와 저장 predicted/target ID equality를 owner가 검산했다.
Eff/Gen/Loc은 **요청 내부 teacher-forced token accuracy → 요청2000개 평균 ×100**이다.
Loc은 `loc_ans` 정답이며 W0 agreement, 전체 token micro 평균, strict prompt 정확도로 대체하지 않는다.

| 모델 | 완료 행 | 요청/행 | E token | G token | Loc token |
|---|---:|---:|---:|---:|---:|
| Llama | 6 | 2000 | 6035 | 6035 | 12465 |
| GPT-J | 5 | 2000 | 5557 | 5557 | 9694 |
| Qwen | 2 | 2000 | 6691 | 6691 | 11476 |

입력/target CPU mismatch0은 독립 pretrained 모델의 수치 출력 parity나 전체 논문 재현 PASS가 아니다.
Qwen은 공개 non-Llama 분기 적용이다. W0 행은 기존 exact-token-prefix 관측으로 남고,
모든 대기·실패 행까지 공개-query 재평가가 완료됐다는 주장은 하지 않는다.

## 상태 및 제외

- Qwen zsRE MEMIT62081/MEMIT-FE62085 RUNNING, Alpha62083 PENDING.
- Qwen zsRE SPHERE62087 FAILED/W20 없음. 숫자 없음; 이번 통합에서 재제출하지 않았다.
- Qwen CF Alpha62075 및 GPT-J zsRE SPHERE61947 PENDING.
- Llama Flu/Con 평가62259/62260/62261 RUNNING: 해당6개 생성 셀만 ING로 변경. 아직 최종 metric 없음.
- History 생성62262/62263 PENDING, 수정 Qwen history62061 생성은 DEFERRED.
- SH3/SH4 적격 완료 baseline0, own 완료 zsRE0: 검산 NOT_APPLICABLE이며 GPU PASS가 아니다.
- 기존 historical 예외·W0 분리 출처·OURS/튜닝 제외를 유지한다. 새 jobs/forward/복원/삭제/원자료 수정0.

## 근거 및 재현

각 owner 파일에 source/config/ordered cohort/raw SHA/최종 checkpoint·commit 근거와 원 수치가 있다.

- [SH1 결과](../../../audits/servers/server1/baseline-completed-zsre-audit-20261010/table-rows.json), [보고서](../../servers/server1/baseline-completed-zsre-audit-20261010/report-ko.md)
- [SH2 결과](../../../audits/servers/server2/baseline-completed-zsre-audit-20261010/audited-final.json), [보고서](../../servers/server2/baseline-completed-zsre-audit-20261010/report-ko.md)
- [SH3 제외 inventory](../../servers/server3/baseline-completed-zsre-audit-20261010/report-ko.md)
- [SH4 제외 inventory](../../servers/server4/baseline-completed-zsre-audit-20261010/report-ko.md)
- [실제 direct 전달·owner ACK](../../../audits/global/baseline-completed-zsre-audit-20261010/dispatch.json)

GH 재검산은 compact-to-table 범위다. owner의 전체 raw 검산을 새로 수행했다고 주장하지 않는다.
`python3 audits/global/baseline-completed-zsre-audit-20261010/verify_table.py`는 입력 JSON SHA를 고정하고
완료30개 관측(기존 포함)의 numeric 셀 및 미완료 상태를 검사한다. raw 생성문·tensor는 Git에 추가하지 않는다.
NO_BROADCAST_NOT_REQUIRED: 이미 게시된 compact receipt만 통합. 장기 polling/자동 retry 없음.
