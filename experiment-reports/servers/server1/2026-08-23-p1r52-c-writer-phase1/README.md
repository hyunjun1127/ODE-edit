# P1R52 C-writer Phase 1 보고서 인덱스

이 디렉터리는 FULL-FP32 Alpha-cache sequential Phase 1의 소형 canonical 분석 산출물만 보존한다. 원시 result/log/model/cache/tensor는 포함하지 않는다. 과학적 promotion은 하지 않았다.

## 읽는 순서

1. 사용자 최종 제시본: [`final-presentation-v5/p1r52-c-writer-phase1-full-fp32-sequential-tech-r1-exhaustive-factual-ko.md`](final-presentation-v5/p1r52-c-writer-phase1-full-fp32-sequential-tech-r1-exhaustive-factual-ko.md)
   - SHA256 `b93391e5c344bd5635fdd1c2f6532d41e683a88d85dabb3be21e949f05f3891b`
   - sequential 대표 endpoint를 `B10 종료 후 W10으로 B1~B10 전체 1,000 requests 재평가`로 맨 위에 둔다.
   - 공통 FULL-FP32 W0 pre-edit, rewrite/rephrase 분리, accepted/native-z와 immediate-post W를 구분한다.
   - C0/C1/C3는 각 B100에서 z를 K1→K8까지 최적화한 뒤 writer를 한 번 적용한다. K1→K8 표는 writer intermediate endpoint가 아니다.
2. GH accepted exhaustive 근거: [`accepted-exhaustive-v3/p1r52-c-writer-phase1-full-fp32-sequential-tech-r1-exhaustive-factual-ko.md`](accepted-exhaustive-v3/p1r52-c-writer-phase1-full-fp32-sequential-tech-r1-exhaustive-factual-ko.md)
   - SHA256 `cc5c210ad3a80e2237bb55a4a4cb5f91c260d3f368e745319dd5f2fb92daec3b`
   - terminal completeness 50/50 batches, 5,000/5,000 requests, C K receipts 240/240, failure/imputation 0.
3. `final-presentation-v5/tables/`에는 final-v5가 사용한 pre-edit, rewrite, rephrase, terminal-W10 소형 CSV/JSON 표를 둔다.

## Arm 경계

| Arm | Target / z | Writer |
|---|---|---|
| Official AlphaEdit | EasyEdit native `compute_z` | Official `apply_AlphaEdit_to_model`, native dynamic `cache_c` continuity ON |
| Native MEMIT | EasyEdit native latent-z | Official `apply_memit_to_model`, native static `COV_CACHE`; Alpha cache N/A |
| C0 | P1R52 K8 accepted-z | PIR-U control allocation + remaining-residual L4→L8 prefix-sequential writer |
| C1 | P1R52 K8 accepted-z | Joint P+C epigraph minimax allocation + remaining-residual L4→L8 prefix-sequential writer |
| C3 | P1R52 K8 accepted-z | direct Official AlphaEdit writer adapter, native `compute_z=0` |

Native AlphaEdit-z, MEMIT latent-z, P1R52 accepted-z는 provenance가 다르므로 같은 z 알고리즘으로 해석하지 않는다.
