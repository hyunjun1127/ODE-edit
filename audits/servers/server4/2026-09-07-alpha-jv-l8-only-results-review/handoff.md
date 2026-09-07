# SH4 L8-only 결과 검토 — GH handoff

instruction_id: ODEEDIT-S06-ALPHA-JV-L8-SERVER4-RESULTS-REVIEW-V1

상태: REVIEW_READY. 전용 분석 브랜치만 제출하며 main 통합은 GH 검토 후다.
기존 실행 source/raw/checkpoint/log 변경0, 새 model/GPU/evaluator/Slurm0,
promotion=false. 단 한 번의 scheduler 확인에서 38433_4/5는 COMPLETED0,
task-owned active/pending0이었다. 후속 polling0.

## Identity 및 검증 범위

- 분석 source: `6025d5f896150aa8d68ba54d722856f0955ccaca`, tree `95b44b1ea03bbc62b0118c1c5f7e06810b354d59`.
- 실행 source: `44602a1a80554c67da0ef9646b43d843104785f2`, tree `cdb089a139f180c822d1e6bf44fe3d27b858c1ce`.
- O/JV reference publication: `0d0a0131e4a6a2a645dfa6530377d420a084d136`의 48 members. Server2 원본 raw/checkpoint를 재검증했다고 주장하지 않는다.
- L8 raw 306 members / 41,600,058,801 bytes, 20/20 batches, 2,000 requests, W/M 연결18/18, checkpoint 파일 및 CPU tensor6/6; nonfinite/failure/duplicate0.
- 최종 비교6 arms 각각 final W10의 RS1000/PS2000/NS10000 분모. 서로 다른 host/backend이므로 controlled speedup/bitwise cross-host parity 주장은 하지 않는다.
- 원래 Server2 38306_4/5 partial prefix는 사용0, final denominator0. User-directed external cancellation이며 과학 실패가 아니다.
- Focused tests20 PASS, compile 및 whitespace PASS. 봉인 CSV의 표준 CRLF는 원래 bytes/SHA를 보존하며 Git whitespace 검사는 `cr-at-eol`로 수행했다. 전체 회귀/GPU gate0.

Canonical package:
`experiment-reports/servers/server4/alpha-jv-l8-only-sequential1000-review-2026-09-07-v1`

|항목|SHA256|
|---|---|
|factual-report-ko.md|a7b07d16ea36e0aefe7aa16fb2c259275ee1879d486f084a469f9f94dd414732|
|analysis-manifest.json|882b0d6ddca002bd55acc333fc9aa8b894847b686e1adcf9f9931ea1e5965d55|
|rooted-receipt.json|9ee3123ac283ab7313d288fd3fff0464f8db84cd72fb95549247609c71c4d2e4|
|member root|aeb70fcef75cdddf41fcbc2d96c4672ba5fa481acbb3ff97557377bed27f782d|
|receipt root|1e61e4c40eb61f4ac2cf17cac3eac91df26dbdfa56e03085283729447aca1200|
|plot-manifest.json|8337d75b49aa365073d4d4cf86dcea1ca8e53e7b4dfd5a49f5cc8db594dfa91b|

45 package files: 21 CSV (8,238 rows), 13 JSON, 9 PNG, 2 Markdown.
Manifest member inventory는 manifest와 rooted receipt 자체를 제외한43개다.
두 큰 scalar CSV는 로컬에만 남기고 external-scalar-table-identities.json에 결속했다.

## 재현 명령

아래는 완료된 frozen inputs를 사용하는 CPU 분석 명령이다. 실행 source와
live task를 건드리지 않는다. 기존 output은 create-once이므로 새 재현 디렉터리를
사용한다. 이 handoff에서는 추가 실행하지 않았다. Analysis worktree에서 실행한다.

```bash
review_python=/data/janghj/EasyEdit/.venv/bin/python
review_module=project.run_scripts.alpha_native_response_ode_v31_sequential.l8_takeover_analysis
review_raw=/data/janghj/ODE-edit/local/state/alpha-jv-migration-server4-20260907/tech-r1
review_sh1=experiment-reports/servers/server1/alpha-jv-sequential1000-layer-review-2026-09-07-v1
review_output=$(mktemp -d /data/janghj/ODE-edit/local/alpha-jv-l8-only-results-review/reproduce-XXXXXX)

"$review_python" -m "$review_module.integrity" --raw-root "$review_raw" --source-repo "$PWD" --output "$review_output/integrity"
"$review_python" -m "$review_module.integrity" --raw-root "$review_raw" --runtime-comparison-inventory "$review_output/integrity/raw-member-inventory.json" --output "$review_output/integrity/runtime-comparison.json"
"$review_python" -m "$review_module.performance" --root "$review_raw" --sh1 "$review_sh1" --output "$review_output/performance"
"$review_python" -m "$review_module.mechanism" --l8-root "$review_raw" --external-main "$review_raw/external-main" --sh1-package "$review_sh1" --output "$review_output/mechanism"
"$review_python" -m "$review_module.plots" --performance "$review_output/performance" --mechanism "$review_output/mechanism" --output "$review_output/plots"
"$review_python" -m unittest discover -q -s project/run_scripts/alpha_native_response_ode_v31_sequential/l8_takeover_analysis -t . -p 'test_*.py'
```

봉인된 최종 분석으로 package를 재생성하는 실제 명령은 다음과 같다.
`--output`에는 아직 없는 경로만 허용한다. 새 report commit의 HEAD는 manifest에서
분리 기록되므로 package manifest 자체의 SHA는 새 source provenance에 따라 달라진다.
PNG는 동일 sealed CSV+plot source에서 두 번 렌더링하여9/9 byte-identical을 확인했다.

```bash
/data/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.alpha_native_response_ode_v31_sequential.l8_takeover_analysis.publish \
  --repo /data/janghj/ODE-edit/local/alpha-jv-l8-only-results-review/analysis-v1 \
  --performance /data/janghj/ODE-edit/local/alpha-jv-l8-only-results-review/performance-v2 \
  --mechanism /data/janghj/ODE-edit/local/alpha-jv-l8-only-results-review/mechanism-v4 \
  --integrity /data/janghj/ODE-edit/local/alpha-jv-l8-only-results-review/integrity-v1 \
  --plots /data/janghj/ODE-edit/local/alpha-jv-l8-only-results-review/plots-v2 \
  --output <NEW_CREATE_ONCE_PACKAGE_PATH>
```

## 해석 경계와 다음 상태

L8−JV의 final RS/PS/NS 차이는 Llama −0.5/+0.6/−0.45pp,
Qwen −0.2/+2.0/−0.42pp이다. Qwen에서는 L8도 Official보다 NS가 높지만
Llama에서는 그렇지 않다. Llama B10 near-stall은 JV와 L8 모두 관측되며
상이한 W/M/z trajectory를 같은-state 인과실험으로 취급하지 않는다.

미기록은 report에 NR로 유지했다: L8↔reference 동일 prompt NS loss/recovery,
O/JV margin quantiles, 일부 library/driver versions, W0→W10 dense net displacement.
후속 normalization/sweep/새 run 권한은 부여하지 않는다.
다음 상태: GH_REVIEW_PENDING / TASK_COMPLETE_STOP.
