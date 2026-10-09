# Qwen 61813 W20 CPU 회수 검산

Job `61813` / `qwen-price-q3-eot-2k`는 cold 2k final 실행이다. 실제 arm은 `Q3-beta250`, cohort_role=eval2k_final, tier=final이다. Q3 튜닝 자체와 구별하되 Q3 선택 설정의 본표 승격 예외는 확인되지 않아 **README 자동 반영 대상에서 제외**하고 GH에 수치를 전달한다. README는 편집하지 않았다.

| 범위 | Score | Efficacy | Generalization | Specificity | Flu | Con |
|---|---:|---:|---:|---:|---|---|
| CF W20 2000 | 83.770638 | 98.050000 | 89.950000 | 68.985000 | DEFERRED | DEFERRED |

단위는 %. NLL strict preference(tie 실패), 요청별 동일 R1/P2/N10이므로 prompt-pair 비율과 request macro가 일치한다. 분모 R2000/P4000/N20000. raw 26000행의 case 순서·row/token identity·유한성·token/strict 분모와 저장 집계를 독립 표준 Python reducer로 검산했다. 기존 jlz observer 사용이며 최신 official evaluator 실제 forward 동등성 검증은 이번에 수행하지 않았다. TF accuracy는 summary.json에 별도 저장했다.

20 commit, 19 W/H/RNG join, history100회 및 observer state receipt 정합. 원 source `93341767ccb41a4237e8a762421340622d3580f1`, config `a14e7acf1b208984a7c143f1b6d07af3aafa14b187fc098a27cf68e95b96178f`, stream `66edc483a8d4bcadedd479e4c36759a686ad61a38741d8870a9052b795710e37`. beta_base/c/beta_max_scale=2.5, M1=false/EOT pos0, 기본 설정으로 relabel하지 않았다. 관측시각 2026-10-09T14:05:57.007396+00:00.

프로그램 시간 23255.990s, 단발 accounting COMPLETED/0:0, allocation 06:28:01 (1GPU/8CPU/58GiB). 서로 중복 합산하지 않는다. Scheduler 완료만으로 판정하지 않고 W20 raw/terminal/result/commit을 확인했다. W20 snapshot은 editable L4–L8 weights만이며 exact resume 불가; 복원/전송/삭제 없이 보존했다. 생성평가 수치는 없으며 0으로 채우지 않았다.

재현: 준비 WT에서 `python3 audits/servers/server3/main-table-refresh-20261009/qwen/reduce.py`. 원 raw는 local 유지, inventory.json에 읽은 파일 SHA/bytes/mtime 기록. owner CPU audit이며 별도 reviewer/GPU검산/새 job/기존 job 변경 없음. 현재 이 scope에서 검산한 새 본표 자동 적격 행은 0개이며 과거 타실험을 자동 승격하지 않았다.
