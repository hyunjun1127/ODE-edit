# P1R52 C-writer Phase 3 보고서 인덱스

Phase 3는 FULL-FP32 B1→B10 W/Alpha-cache sequential continuity를 유지하면서 각 B100 안에서 K1→K8마다 writer를 적용한 결과다. 비교 기준 Phase 1은 같은 stream/cache sequential 경계에서 accepted-z를 K8까지 완성한 뒤 B100당 writer를 한 번 적용한다.

- canonical report: [`exhaustive-v1/p1r52-c-writer-phase3-kstep-cache-sequential-full-fp32-exhaustive-factual-ko.md`](exhaustive-v1/p1r52-c-writer-phase3-kstep-cache-sequential-full-fp32-exhaustive-factual-ko.md)
- report SHA256: `fbb790618655f7c975e0f5c42dff3860c8159880be44e38a258fe046a26047d8`
- analysis manifest SHA256: `17b1914cd387c353f01a843cc9b72f3ac711ce02f8282adbc96a535dd7993361`
- manifest member root: `04758fc0f0dfea4c15022f650a6474488354db6cae6b967872c17da5085f5f29`
- rooted receipt identity: `ae578ebc22ff8fb5c4dc288ba2494f8e9fcfcab8487357b9e1e716cad4587356`
- completeness: 30/30 sequential batches, 3,000/3,000 requests, 240/240 K writes, cache audit 30/30, failures/retries/imputations 0

`exhaustive-v1/`에는 method 정의, endpoint/NLL aggregate, per-unit/per-K, route, layer update, cache continuity, compute/overhead, Phase 1 paired comparison, artifact inventory와 receipts를 보존한다. 225,000행 prompt-level 원자료 CSV는 Git의 raw/large-result 제외 경계에 따라 싣지 않았으며, source manifest가 원본 SHA와 행 수를 계속 결속한다.
