# SH3 FLU/CON 표시 배율 및 Qwen W0 provenance

Nonce: USER-GH-FLUCON-PAPER-SCALE-TABLE-REFRESH-20261010-R1. 정본 e99b61e3의 전체 envelope와 main-results-policy를 읽었다. README는 GH 단독 통합이며 이 보고서는 CPU 검산 및 소형 입력 전달이다.

SH3 자체 신규 실측 FLU/CON 셀은 0개다. 선택 Qwen W0는 job 61813 `qwen-price-q3-eot-2k`의 편집 전 cold 상태이고, 생성 평가를 수행하지 않았다. 따라서 두 생성 셀은 **DEFERRED**를 유지하며 0으로 채우지 않는다.

| 대상 | Eff | Gen | Loc | Score | Flu | Con |
|---|---:|---:|---:|---:|---|---|
| Qwen CF W0, 2000 requests | 13.95 | 16.60 | 85.14 | 20.88105346436503 | DEFERRED | DEFERRED |

저장 W0 raw 26,000행을 독립 stdlib reducer로 재집계했다. R/P/N 분모는 2,000/4,000/20,000, 성공수는 279/664/17,028이다. 요청별 macro, finite, 중복 key, 저장집계 일치, initial cold와 W0 state 일치를 검사했다. 기존 GH W0 inventory 47개 파일의 현재 SHA/bytes도 일치한다. factual 값은 그대로다.

SH2가 전달받은 생성 raw `6.252105796227186` bits / `0.2591242773267912` cosine의 표시 산술은 각각 **625.21 / 25.91**이다. 이는 Decimal half-up으로 반올림 전 값에 100을 곱한 결과이며 **SH3 생성 실측이나 호환성 PASS가 아니다**. Flu는 백분율이 아니며 Con도 정확도가 아니다. 원 raw/W&B 키는 변경하지 않았다.

SH2용 `qwen-w0-provenance.json`은 모델 revision a09a35458c702b33eeacc393d103063234e8bc28, source 93341767ccb41a4237e8a762421340622d3580f1, config a14e7acf1b208984a7c143f1b6d07af3aafa14b187fc098a27cf68e95b96178f, full CF stream SHA 66edc483a8d4bcadedd479e4c36759a686ad61a38741d8870a9052b795710e37, ordered IDs SHA, cold W/H identity, tokenizer/config의 기존 asset lock 및 runtime을 제공한다. 실행 seed 20261002는 factual/편집 실행 seed이며 SH3 generation seed로 해석하지 않는다. SH2는 자신의 generation source/protocol/reference/tokenizer/cohort를 별도 결속해야 한다. 호환 미확인 상태에서는 DEFERRED이며 SH2 생성 provenance를 SH3로 바꾸면 안 된다.

단발 endpoint 확인에서 Qwen 61813의 result는 COMPLETE/20batch이나 Q3 선택설정으로 별도 보고를 유지한다. Llama 61821 `llama-price-L1-2k`는 해당 local final의 result/terminal이 아직 없으므로 완료 점수를 만들지 않는다. 이 점검은 scheduler 상태 추측이나 반복 모니터링이 아니다. own 완료 zsRE는 기존 한정 inventory상 없으며 타서버 이관 replica를 중복 평가하지 않았다. 과거 변형/heldout/tuning은 본표로 승격하지 않았다.

재현: 저장소 준비 WT에서 `python3 audits/servers/server3/flucon-paper-scale-20261010/reduce.py`. 결과는 같은 audit 폴더의 table-rows.json 및 qwen-w0-provenance.json, 본 폴더 table-rows.csv에 있다. owner CPU audit이며 독립 reviewer/GPU 검증은 수행하지 않았다. 모델 load/forward/새 job/기존 job 변경/CP 이동·삭제/온라인 history 변경 0. NO_BROADCAST_NOT_REQUIRED; 작은 provenance만 Git/직접 전달한다. 별도로 보류된 context 전달은 재개하지 않았다.

GH 표시 API main15091336을 통합하고 official.tests.test_generation_paper_display CPU 5개 PASS 및 두 raw 값의 공통 paper_cell 결과 일치를 확인했다. HF asset의 blob 실제경로와 requested snapshot filename을 함께 보존하여 tokenizer/config 6개와 model shard 4개의 기존 SHA/bytes를 결속했다. 대형 모델 재해시는 수행하지 않았다.
