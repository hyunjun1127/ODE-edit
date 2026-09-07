# ORBODE cumulative analysis-only package

이 모듈은 완료된 `execution-r2 / job37649`를 읽는다. 모델/실험 runtime을
호출하지 않는다. 원래 runtime/source와 모든 raw artifact는 수정하지 않는다.
분석 원문 계약은 `ODEEDIT-S06-ORBODE-CUMULATIVE-RERUN-EXHAUSTIVE-REPORT-SH4-V1`.

각 명령의 출력 디렉터리는 create-once이며 기존 출력 재사용/덮어쓰기를 금지한다.

```bash
PY=/data/janghj/EasyEdit/.venv/bin/python
MOD=project.run_scripts.ordered_response_barrier_ode.analysis_cumulative_report
$PY -m "$MOD.integrity" --output <WORK>/integrity-v1
$PY -m "$MOD.performance" --output <WORK>/performance-v1
$PY -m "$MOD.mechanism" --output <WORK>/mechanism-v1
$PY -m "$MOD.comparisons" --performance <WORK>/performance-v1 --mechanism <WORK>/mechanism-v1 --output <WORK>/comparisons-v1
$PY -m "$MOD.tail" --mechanism <WORK>/mechanism-v1 --integrity <WORK>/integrity-v1 --output <WORK>/tail-v1
$PY -m "$MOD.plots" --performance <WORK>/performance-v1 --mechanism <WORK>/mechanism-v1 --comparisons <WORK>/comparisons-v1 --output <WORK>/plots-v2
$PY -m unittest -q "$MOD.test_focused"
$PY -m "$MOD.report" --work <WORK> --output <NEW_PACKAGE>
$PY -m "$MOD.verify" --package <NEW_PACKAGE> --seal
$PY -m "$MOD.verify" --package <NEW_PACKAGE>
```

`integrity`는 약258GB 원본의 SHA와 checkpoint tensor를 CPU에서 읽는다.
`performance`는 봉인된 26-prompt-kind/request row의 accepted canonical reducer를
재사용하며 새로운 evaluator forward는 없다. `comparisons`는 원래 dataset의
exact metadata hash로 충돌 한 쌍을 분류할 뿐 표본을 선택/제외하지 않는다.
`mechanism`은 current command와 historical fixed-target preservation을 별도 분모로
집계한다. Native C_reg unbound, nominal dynamic path와 stored-weight net 구별을 유지한다.

공개 재현 그림은 raw 없이 package CSV만으로 생성할 수 있다. `plots`는 Agg backend,
고정 순서/색/스타일/seed/DPI를 쓰고 각 PNG를 두 번 렌더하여 bytes exact를 검사한다.
`verify`는 table 산술, 분모, row SHA, plot input/output, source SHA 및 raw-free 범위를
검사하고 manifest/rooted receipt를 생성한다. CUDA/model/GPU/Slurm 호출0.
