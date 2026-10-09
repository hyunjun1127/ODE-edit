# zsRE Loc 정정: GH 직접 official 구현 및 완료 12행 CPU 재집계

승인: `USER-GH-ZSRE-LOC-RECALCULATE-MAIN-20261009-R1` 및 사용자 추가 지시
“official에 존재하는 코드 자체도 갱신해. 이건 GH가 직접해”. 구현 commit `3a400ae5`.

## 정정 내용과 근거

기존 Specificity는 W0 예측 보존율이었다. 본표 Loc은 이제
`100 × mean_requests(mean_loc_ans_tokens(predicted_token == target_token))`다.
W0가 틀린 답을 그대로 출력해도 Loc 성공이 되지 않는다. `Specificity_loc_ans`는
동일 정답 정확도의 호환 별칭, `W0_prediction_agreement`는 별도 보조지표다.

공개 [AlphaEdit dataset](https://github.com/jianghoucheng/AlphaEdit/blob/main/dsets/zsre.py)은
`loc_ans`로 neighborhood target을 만들고,
[평가기](https://github.com/jianghoucheng/AlphaEdit/blob/main/experiments/py/eval_utils_zsre.py)는
예측 argmax와 target token을 비교한다. 이번 집계 정의는 이 정답 기준을 따른다.
기존 token-prefix 구현과 원 논문의 tokenizer 처리·실험 조건 완전동등은 별도 문제이며,
이번 정정만으로 paper reproduction PASS를 주장하지 않는다.

GH가 `official/evaluation/{reduce,factual}.py`, hparams 계약, W&B mapping/whitelist,
서버 caller의 보조지표 whitelist, server1 CPU audit, 문서·테스트·SOURCES를 직접 수정했다.
평가 forward/입력/편집 수학은 변경하지 않았다. W0 reference가 없어도 실제 측정한
loc_ans 정확도는 유효하며, 보조 W0 agreement만 미측정으로 분리한다.
새 public mapping은 서로 다른 Specificity와 Specificity_loc_ans를 받으면 명시적으로 거부한다.
옛 frozen source와 W&B history에는 hotpatch/rename/backfill을 하지 않는다.

## 실제 완료 산출물

SH1 Llama 6개 및 SH2 GPT-J 6개 W20/2,000건 raw의 predicted/target IDs에서 correctness를
독립 도출하고 저장 bits와 교차검산했다. 각 요청 안에서 먼저 평균한 뒤 2,000개 요청을
평균했다. 저장 `Specificity_loc_ans`는 계산 입력이 아니라 결과 교차검산용이었다.

| 모델 | 방법 | 이전 W0 보존율 | 정정 Loc |
|---|---|---:|---:|
| Llama | FT | 1.42 | 1.59 |
| Llama | MEMIT | 37.25 | 30.49 |
| Llama | AlphaEdit | 51.22 | 44.85 |
| Llama | BLUE | 66.69 | 42.31 |
| Llama | MEMIT-FE | 0.97 | 0.60 |
| Llama | SPHERE | 50.22 | 44.31 |
| GPT-J | FT | 0.52 | 0.62 |
| GPT-J | MEMIT | 55.92 | 30.87 |
| GPT-J | AlphaEdit | 84.76 | 27.99 |
| GPT-J | BLUE | 86.49 | 28.90 |
| GPT-J | MEMIT-FE | 29.46 | 8.46 |
| GPT-J | SPHERE | 86.23 | 28.06 |

각 행 request 분모 2,000. Llama E/G token 분모 각각 6,035, Loc 10,465;
GPT-J E/G 각각 5,557, Loc 9,694. 이 token 분모는 coverage이며 점수는 token-micro가 아니다.
Eff/Gen도 독립 재집계했으며 README 소수 둘째 자리 표시는 변하지 않았다.
CF 완료 8개 행의 수치는 이전 CPU receipt와 그대로 일치하며, 역사 예외 행도 변경하지 않았다.
Flu/Con 미평가/DEFERRED를 0으로 채우지 않았다.

- [통합 before/after JSON](../../audits/global/zsre-loc-correction-20261009/results.json), [CSV](../../audits/global/zsre-loc-correction-20261009/table.csv)
- [SH1 원자료 SHA·config/cohort·분자/분모](../servers/server1/official-baselines-20261008/zsre-loc-recalculate-20261009/report-ko.md), reducer SHA `c95182ca6dc72486e257c2d1af8b495b122950209f52916bcdadf61278627511`
- [SH2 원자료 SHA·config/cohort·분자/분모](../servers/server2/zsre-loc-recalculate-20261009/report-ko.md), reducer SHA `0c17f3648ed8de5119e61088954e6b7000f3ef58895277d503256f83f9793ef5`
- [SH3 범위 inventory: 완료 zsRE 없음](../servers/server3/official-baselines-20261008/zsre-loc-review-20261009/report-ko.md)
- [SH4 범위 inventory: 완료 zsRE 없음](../servers/server4/zsre-loc-recalculate-20261009/report-ko.md)

확인한 승인 완료 12행 중 누락/미검증 0. 승인 완료 OURS zsRE는 네 서버 inventory에서
발견되지 않았다. 미등록·전체 filesystem의 임의 결과까지 완전 조사했다는 뜻은 아니다.
Qwen 신규 12개는 제출 결과이지 완료 결과가 아니므로 Loc을 채우지 않는다.

## 검증 및 보존

- GH CPU 회귀 168개 PASS: reducer/평가기/W0 reader/실제 mapping·fake transport/서버 caller·audit.
- SOURCES 157 SHA, Python 300, external task imports 0 PASS.
- 표 checker: 완료 20 dataset-method 행, 정정 zsRE 12행, 신규 Qwen job 12개와 receipt 일치 PASS.
- 초기 검사 1건의 source lock stale 실패는 수정 후 manifest 재결속 및 전체 168개 재검사로 해소.
- 모델/GPU 평가·재편집·checkpoint 복원·온라인 history 변경 0. 기존 raw/source/CP/job KEEP.

병행 Qwen 이전은 SH2에서 source `69bfbb2c`로 이미 freeze/release했다. CF
61898/61902/61906/61910/61914/61918, zsRE 61900/61904/61908/61912/61916/61920,
archive 홀수 61899–61921, collector61922. 이 frozen source를 이번 정정으로 덮어쓰지 않는다.
이후 저장 raw를 새 정의로 후처리한다. archive trust가 freeze 후 도착하여 현재 실행은
`ARCHIVE_PENDING_KEEP_SOURCE`; 전송/삭제 PASS를 주장하지 않으며 원본을 보존한다.
