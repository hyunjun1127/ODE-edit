# P1R52 C-writer Phase 1/2/3 canonical 결과 인덱스

세 Phase는 `SINGLE_CANONICAL_STREAM` 한 벌을 공유한다. B1–B10×B100의 1,000 requests와 순서는 byte/order identity로 봉인됐고 sample duplication은 0이다.

| Phase | Job | 실행 경계 | Canonical report |
|---|---:|---|---|
| Phase 1 | 22759 (array tasks 포함) | B1→B10 sequential W/Alpha-cache; 각 B100에서 z K1→K8 완료 후 writer 1회 | [`../2026-08-23-p1r52-c-writer-phase1/README.md`](../2026-08-23-p1r52-c-writer-phase1/README.md) |
| Phase 2 | 22839/22841/22842 | 10 independent W0 B100; 각 B100 내부 K1→K8 writer 8회; cache OFF | [`../2026-08-23-p1r52-c-writer-phase2/README.md`](../2026-08-23-p1r52-c-writer-phase2/README.md) |
| Phase 3 | 22840/23042/23043 | B1→B10 sequential W/Alpha-cache; 각 B100 내부 K1→K8 writer 8회 | [`../2026-08-23-p1r52-c-writer-phase3/README.md`](../2026-08-23-p1r52-c-writer-phase3/README.md) |

## 공통 stream seal

- stream root: `467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a`
- all-request order root: `018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3`
- ordered-record root: `af215235177d1ac07e82fadc62de7235244a82342a146c5867f673625486d0e1`
- sample payload identity: `fe7c8b0cb51abf591e0ec560c0009bdcc47f20e8c4efcf6efc3311710a373475`
- payload count 1, duplication count 0

`stream-seal/`에는 raw sample bytes가 아니라 canonical selection/order specification, identity index, mapping, manifest와 rooted receipt만 보존한다. `analysis/`에는 Phase 2/3 공통 집계 builder를 보존한다.

## 결과 해석 경계

- Phase 1의 대표 sequential endpoint는 B10 종료 후 W10으로 전체 1,000 requests를 재평가한 값이다.
- Phase 2는 final-v6 one-shot writer와 same-entry/slice/order paired 비교하며 same-z를 주장하지 않는다.
- Phase 3는 Phase 1의 같은 sequential stream 최종 W10과 paired 비교한다.
- 모든 실행은 FULL FP32이며 scientific promotion은 하지 않았다.

기계 판독용 terminal/status identity는 [`phase123-canonical-status.json`](phase123-canonical-status.json)에 있다.
