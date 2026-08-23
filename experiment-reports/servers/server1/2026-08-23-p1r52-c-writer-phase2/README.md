# P1R52 C-writer Phase 2 보고서 인덱스

Phase 2는 동일한 10개 B100 slice를 각각 FULL-FP32 W0에서 독립 실행하고, 각 B100 안에서 K1→K8마다 `accepted-z → writer → W`를 수행한 결과다. 비교 기준 final-v6는 동일 W0/slice/order에서 z를 K8까지 완성한 뒤 writer를 한 번 적용한다. 따라서 same-z equivalence는 주장하지 않는다.

- canonical report: [`exhaustive-v1/p1r52-c-writer-phase2-independent-kstep-full-fp32-exhaustive-factual-ko.md`](exhaustive-v1/p1r52-c-writer-phase2-independent-kstep-full-fp32-exhaustive-factual-ko.md)
- report SHA256: `7d34b40f4e1983061113dbce1afb771436c187ae54286e4cc349d47616e2a6b6`
- analysis manifest SHA256: `923d485574afcb8f5d8859b2dc8331168f2146418362266b5ded6f96792ec103`
- manifest member root: `e8964390b10963eb4f3b89b63f7f8e9b3dd49a06231e647bb851d0f21cf87ae1`
- rooted receipt identity: `01ae51f73877ab9d70ffaccf32e20f2e94f7d3eb35c4f19e2c505caec3776b7f`
- completeness: 30/30 independent cases, 3,000/3,000 requests, 240/240 K writes, failures/retries/imputations 0

`exhaustive-v1/`에는 method 정의, endpoint/NLL aggregate, per-unit/per-K, route, layer update, compute/overhead, paired comparison, artifact inventory와 receipts를 보존한다. 216,000행 prompt-level 원자료 CSV는 Git의 raw/large-result 제외 경계에 따라 싣지 않았으며, source manifest가 원본 SHA와 행 수를 계속 결속한다.
