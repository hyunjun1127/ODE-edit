# Temporal routing diagnostic 실행

정본: `plans/global/2026-09-29-temporal-routing-diagnostic-v1/`.
원본 native/공유 환경은 읽기 전용이다. `prepare`와 `preflight`는 CPU 입력/토큰 준비,
`native`는 공식 body 호출/계측, `observations`는 observer-only TF,
`runner`는 parent/layer별 고정100 BS1, `reduce`는 afterany CPU 집계다.

```bash
python -m unittest project.run_scripts.temporal_routing_diagnostic.test_core -v
python -m project.run_scripts.temporal_routing_diagnostic.prepare --repo "$TASK_REPO" --output "$TASK_INPUT"
python -m project.run_scripts.temporal_routing_diagnostic.preflight --binding "$TASK_INPUT/input-binding.json" --output "$TASK_PREFLIGHT"
python -m project.run_scripts.temporal_routing_diagnostic.launch --repo "$TASK_REPO" --attempt attempt-v1 --configuration "$TASK_PREFLIGHT/configuration.json"
python -m project.run_scripts.temporal_routing_diagnostic.launch --submit-existing "$TASK_ATTEMPT"
```

환경 변수는 문서 예시이며 실제 launcher에는 봉인한 절대경로가 들어간다.
모든 branch를 upfront array0–14%2로 등록하고 CPU collector는 afterany다.
이미 작성된 source/attempt/행을 덮어쓰지 않는다. 신규 자동 retry/과학 분기 선택은 없다.
Snapshot은 n50/n100 실제 selected-layer weight만. Editor resume bundle은 저장하지 않는다.
실제 초기 또는 resource-pending 인계 후 agent monitoring은 중지한다.
