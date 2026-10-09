# W0 main table 직접 검산 (2026-10-09)

사용자 지시로 README의 세 모델 표에 W0(편집 전) 기준 행을 추가했다. GH에게 수치 확인을 위임하지 않고 server1 로컬 및 server2·3·4 SSH에서 실제 산출물을 조회했다. 기존 36개 편집 실험 행의 범위는 유지하며, 이번 추가는 각 base model의 zero-edit 기준점이다.

| 모델 | CF Score | CF Eff | CF Gen | CF Loc | CF Flu(bits) | CF Con(cosine) | zsRE Eff | zsRE Gen | zsRE Loc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama3-8B-Instruct | 13.36 | 8.20 | 10.95 | 88.56 | 6.35 | 0.2464 | 38.10 | 37.61 | 38.59 |
| Qwen2.5-7B-Instruct | 20.88 | 13.95 | 16.60 | 85.14 | DEFERRED | DEFERRED | 36.42 | 35.26 | 38.40 |
| GPT-J-6B | 24.44 | 17.00 | 19.30 | 82.48 | DEFERRED | DEFERRED | 27.83 | 27.15 | 27.59 |

## 실제 출처

| 모델·평가 | 서버 | job | 선택한 관측 |
|---|---|---|---|
| Llama3 CF 및 FLU/CON | server1 | 61768 / official-s1-cf-w0 | READY의 actual_model_edits=0, factual 2K와 generation 2K |
| Llama3 zsRE | server1 | 61710 / official-s1-zsre-w0 | reference-local.json 안의 전체 evaluation 2K |
| GPT-J CF | server2 | 61723 / W0_CF | READY_COLD_W0_COMPLETE, factual/W0.json |
| GPT-J zsRE | server2 | 61724 / W0_ZSRE | READY_COLD_W0_COMPLETE, factual/W0.json |
| Qwen CF | server3 | 61813 / qwen-price-q3-eot-2k | W0-result.json 및 W0/chunk-*.json, initial.cold와 state 일치 |
| Qwen zsRE | server2 | 61900 / s2-qwen25-zsre-ft-gpu | shared-w0/qwen25-zsre/w0-cases.json 및 w0-receipt.json |

Server4의 최신 Qwen baseline 배포는 server2로 이전됐다. server4의 관련 W0·배포 경로도 SSH 조회했으며 표에는 실제 완료된 server2 관측을 사용했다. Job 61900 전체 chain은 조회 시 RUNNING이지만 독립 W0 receipt는 2,000건 완료다. scheduler 완료 여부만으로 W0 수치를 채우지 않았다.

Qwen CF의 source는 93341767이며 이전 jlz observer이다. **선택된 Q3 편집 설정의 W20 결과를 승격한 것이 아니라, 편집 전 cold W0만 사용**한다. 준비 manifest의 모델 snapshot, 공식 CF stream SHA, 26,000개 observer row identity 및 W0 상태를 검산했다. 최신 official evaluator의 GPU forward 동등성을 새로 입증한 것은 아니다. server2 61898의 별도 W0 생성평가는 조회 시 진행 중이고 완성된 CF W0 receipt가 없어 그 부분 산출물로 값을 채우지 않았다. 이 보고서의 Qwen Flu/Con DEFERRED는 선택한 server3 W0의 상태다.

## 검산

- CF 각 모델: request 2,000, R/P/N 분모 2,000/4,000/20,000. 저장된 유한 raw NLL의 strict preference를 재계산했다. 요청별 평균 후 전체 평균, Score는 반올림 전 세 성공률의 조화평균이다.
- zsRE 각 모델: request 2,000. 모든 저장 predicted_token_ids와 target_token_ids의 일치를 직접 계산하고 token_correct 및 기존 요약값과 대조했다. 요청별 토큰 정확도의 평균이며 전체 토큰 micro 평균이 아니다. Loc은 loc_ans 정확도다.
- CF 및 zsRE 모두 공식 2K case ID 목록과 순서 SHA가 일치했다. 선택한 W0 receipt 또는 cold state와 연결하고 원파일 SHA·bytes를 기록했다.
- Llama3 생성: 2,000개 case의 fluency_valid/consistency_valid 및 유한 값을 확인하고 행별 평균을 다시 계산했다. Flu=6.352242334333923 bits, Con=0.24636896048599818 cosine. tracking의 raw 단위를 유지한다. 논문식 ×100 표시가 필요하면 각각 635.2242334333923 / 24.636896048599818이며 본표는 ×100하지 않았다.
- 표시는 decimal half-up, factual/Flu 소수 둘째 자리, Con 소수 넷째 자리다. 모든 산출물의 저장 summary와 재계산 값이 허용오차 1e-10 이내 일치한다.
- 이번 검산은 CPU read-only이며 새 GPU 평가, 편집, 학습, job 변경 또는 원자료 수정이 없다.

## zsRE 비교 범위

세 W0는 기존 official exact-token-prefix evaluator의 관측이다. Llama3에서는 원본 AlphaEdit/BLUE 추가 공백 처리, GPT-J에서는 Unicode decode-retokenize 때문에 Eff·Gen query 차이가 확인됐다. 따라서 W20 원본 호환 재평가 결과와 이 W0를 동등한 evaluator로 간주하지 않는다. Qwen Eff·Gen의 해당 2K 토큰 비교는 일치했지만 모델 forward 및 Loc 전체 native parity는 별도다. 표의 W0는 원본 호환 재평가 완료로 표시하지 않는다.

이 제한을 이유로 실측 수치를 삭제하거나 W0 Loc에 자기 일치율 100%를 사용하지 않는다. 원본 호환 W0가 별도로 관측되면 동일 provenance 검사를 거쳐 교체할 수 있다.

## 재현 근거

[collector](../../audits/global/w0-main-table-20261009/collect.py), [전체 수치·분모·원파일 SHA·서버/job 및 scheduler snapshot](../../audits/global/w0-main-table-20261009/results.json).
collector는 모델 라이브러리를 import하지 않는 표준 Python 프로그램이다. `python3 collect.py cf|zsre|generation|cf_chunks <원자료 경로>`로 실행하며 원격 서버에는 stdin으로 전달했다. 데이터셋·토큰 원문·생성문·가중치는 Git에 넣지 않았다.
