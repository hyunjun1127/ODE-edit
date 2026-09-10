# 최종 raw-free 검산과 해석

## 검증 사실

- 실행12/12, logical steps104/104. Jobs44124/44143/44144/44158 모두 COMPLETED0; restore PASS, 기타 parameter 변경0. 추가 실험/재실행0.
- 최종 보고서422행/34699 bytes를 전체 검토했다. Report SHA256 `902ca04d2bb646bfc28fdb8def1a9bb84ea6134a3bcc79cf36fd7ed2960025c3`.
- Publication50 members 전수 재해시 PASS. 별도 raw385 members/691943610 bytes 전수 재해시 PASS.
- Raw에서 신규360 aggregate groups의 numerator/denominator와 NLL mean을 별도 `math.fsum` reducer로 검산했다. Mean의 1e-12 비교 허용치는 보고서 산술 검증용이며 controller/scientific threshold 변경이 아니다.
- CPU13 tests/compile/bash PASS. 세 PNG의 실제 시각 검토 및 동일 코드 재실행 후 exact SHA 일치 PASS.
- 124800 state-bound pair-rows, training10400 rows. Bootstrap2000/request-cluster. Imputation0. Source/runtime/data/model/parameter/denominator 경계를 별도 유지.
- 실제2.4119444444 GPUh. Raw/model/prompt/generation/tensor/log는 Git 제외. NO_BROADCAST_NOT_REQUIRED.

## 관측에 근거한 결론

H는 세 entry에서 N보다 Current rewrite NLL을 낮췄다. Early PS만195/200→196/200이고 Middle197/200, Late193/200은 유지됐다. EP의 추가 RS/PS/NS 성공률 이득은 주 비교에서 관측되지 않았다. Early/Late는 barrier가 비활성이고 H/R/EP의 실제 최종 weight와 평가가 동일하다. Middle은 첫 step만 활성이다.

Middle 같은-state matched-risk probe 대비 EP의 train request mean NLL은 약1.35e-6 낮았지만, 일부 요청은 악화됐고 Curve 성공 count는 같았다. Nominal shadow 대비 첫 step mean NLL은 약3.76e-6 높고92/100 요청이 악화됐다. 따라서 일차 progress-null 구현 성립과 실제 요청별/유한-step 보장은 구분해야 한다.

Free는 보정0/8, H와 weight SHA 및 Full3900 rows가 exact 동일하다. J4의 Current rewrite NLL0.007888969803898362는 EP-N8의0.00788899340594071과 매우 가깝고 성공 count 이득은 없다. N16은 rewrite0.007974483326834161/rephrase1.1684273219815804로 EP-N8의0.00788899340594071/1.1672448669568984보다 높았다. 모든 Middle 보조 arm은 Current PS197/200, Fixed NS696/1000, Past NS685/1000이다. N16의 추가 gradient는15회이며 N8은7회(공통 G0 별도)다. 이번 패널에서는 추가 feedback의 성공률 이득이 입증되지 않았다.

실용적으로는 native 이후 공통 refinement의 NLL 개선과 barrier의 추가 이점을 분리해야 한다. 이 campaign의 낮은 활성 빈도/동일 성공 count는 EP 도입의 추가 실용적 근거가 제한적임을 보여준다. 다만 이미 Current RS100/100인 개발 패널, 작은 변위, 세 entry에 한정되므로 보편적 무용성이나 다른 운영점의 결론은 아니다. Cold-z 포함 production-only latency는 측정하지 않았다.

## 한계 및 남은 미측정

- Free-energy는 저장 G0의 entry pre0 진단이며 이후 node 비중은 NOT_RECORDED.
- Matched probe risk는 reduced-coordinate 값이며 full-weight actual FP32 risk는 NOT_RECORDED.
- Production-only controller latency와 unique generation input-token count는 별도 NOT_RECORDED. Generation cached-context mask counter를 새 token 수로 대체하지 않음.
- 전체 cross-server bitwise model/gradient replay는 미검증. 동일 weight의 미세 FP64 scalar 차이는 physical change가 아님.
- 후속 sequential, 다른 layer/model, 새 benchmark, scientific promotion은 승인/실행하지 않음. Scientific promotion=false.

Canonical package: `experiment-reports/servers/server2/blue-l4-progress-barrier-2026-09-11-v1/`.
분석 코드와 원시 수치 링크는 위 보고서/manifest에 결속했다. 봉인 보고서/원본 결과를 이 검산 과정에서 수정하지 않았다.
