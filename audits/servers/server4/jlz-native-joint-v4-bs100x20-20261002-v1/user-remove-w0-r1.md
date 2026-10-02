# USER override: W0 신규 평가 제외 및 기존 결과 재사용

사용자 원문: “W0 실험은 중단하고 없애고 AB 실험올려. 이전의 실험 결과가 존재해서 해당 부분 재사용하면 된다”.

- W0 57282 및 해당 immutable graph의 미실행 57283–57287을 owner/name/source/argv 확인 후 SH4가 취소했다. 취소 actor는 SH4이며 타 task 변경0.
- 원 parent 비용332 allocated GPU초, A/B fit/write0. 원 source/부분 raw/실패 및 취소 자료는 삭제하지 않는다.
- 기존 v2 W0 full2k raw SHA `4436ce3de7889164398c189d4cb8be56716245499e1623bd924962b06243d4d8`를 read-only 재사용한다. 옛 실험을 재개하는 것은 아니다.
- R2000/P4000/N20000 모든 행의 identity, tokenization/count, active flag, finite, strict 및 집계 확인. 기존 W0와 이번 실제 initial-state의 W/H hash 일치.
- 기존 full-position head/category-major와 v4 selected-position head/request-major는 배치 배치가 다르다. 비트 단위 동일성은 미확립. 이미 저장된 중복16900 NLL의 최대 차이 `0.0000972747802734375`; 추가 GPU 평가0.
- 재사용 bridge: `local/jlz-native-joint-v4/20261002-compute-r1/user-remove-w0-r1/W0-reuse.json`, SHA `e4b75f19786291c14f6baaa4d6e1948eec127f4b530b261725dc6b2472719008`.
- 새 graph에는 shared W0 job/forward가 없다. A/B pilot → 두 pilot afterok → A/B timing 및 cold BS100×20 → CPU afterany collector. 두 독립1GPU lane/cap2.
- 각 pilot/main이 기존 W0/H0 hash를 확인한다. 재사용 raw/bridge 변조는 계속 hard failure. A/B 목적/예산/문항/평가/noCP는 그대로다.
- CPU 회귀7개 PASS(1.569초). 실제 GPU pilot/main PASS를 뜻하지 않는다. owner 검산이며 별도 독립 red 미사용.
- NO_BROADCAST_NOT_REQUIRED; 원 raw/model/tensor/prompt/fullstdout Git0. 상세 local cancellation 확인 receipt를 보존했다.
