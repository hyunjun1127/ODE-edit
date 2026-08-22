# Server1 실험 코드·보고서 main 통합 인벤토리

상태: `PREPARED_WAITING_FOR_SH2_ORDER1_AND_SH4_ORDER2`

감사 기준 `origin/main`은 commit `f4fbfb5de132de9b986e4b291c5b41784c182743`, tree `8c3f444014907f69b1bb793674f7a5a9e2b9ee0e`이다. 이 문서는 read-only lineage 감사 뒤 clean branch `codex/server1-all-experiment-integration-v1`에서 작성했다. active Phase2/3 job source, result root, 사용자 dirty worktree는 변경하지 않았다.

## Lineage 판정

| 범위 | 최종 server1 ref / tree | 기준 main 조상 여부 | 통합 판정 |
|---|---|---:|---|
| PIR-U Alpha-cache continuity | `e4a5b8ff95717548af6d164f29fbc62ffb0c26c2` / `59ea4bc470656769e7590665d125b30106c86145` | Yes | 이미 main; 중복 통합하지 않음 |
| P1R52 target + Official AlphaEdit writer | `7ef609c7c1d3267ddbda624afd52bce8241acc4b` / `ccffb6ec4c53a10762885ebc6f7752cde7026370` | Yes | 이미 main |
| Residual-Reserve final repaired lineage | `6caafd1148498497602876a9693f6760e6b39e85` / `9e2c8aac6d21b3dedc7770d4de894e1b73d3c097` | Yes | 이미 main |
| Joint P+C FULL-FP32 independent 10×B100 final-v6 | `96494a05c37ffb32f101d53d14e0e61b2f25664d` / `58c07ad6ef33571650d92c28c58fb31871915a9d` | Yes | code/report 이미 main |
| C-writer Phase1/2/3 final repaired execution source | `034b72c59aace422b7048509b336bac26992d90e` / `76f3b8e0b3304dcdca25a8467439c4bf035bb41c` | No | SH2/SH4 뒤 최신 main에 통합 예정 |
| Phase1 common W0 pre-edit evaluator | `7a0aebd6f1a4bc728cb8d6ff50470b0d10c7cc39` / `f4b30677116f6b11a3b6fc59f9776bdea341a1f0` | No | 최종 단일 pre-edit commit만 통합 예정 |
| Phase1 exhaustive accepted-v3 + final-v5 | local canonical reports | No | 이 integration branch에 소형 보고서/표/receipt를 exact-byte 복사 |
| P4 sealed stream 준비 receipt | member root `28189ad33cdcda0c01d269216514508d62b15cd6c16ed053afd4683577169843` | No | source manifest/receipt만 통합; transfer tar 제외 |

## 아직 main에 없는 final source 범위

Phase2/3 repair ref `034b72c...`는 `96494a05...` 대비 22 files, 3,452 insertions, 7 deletions이다. 공통 K-step orchestration, cache policy, dependency gate, 세 phase launcher/sbatch/dry-plan 및 focused tests만 포함한다. 실행에 사용된 이 ref보다 앞선 `251e616c...`, `762c4788...` 등은 lineage 구성 commit이며 별도 canonical 복제 대상으로 취급하지 않는다.

Pre-edit는 최종 commit `7a0aebd6...`의 세 파일만 대상이다: evaluator launcher, sbatch, focused test. 중간 add/revert 시도는 제외한다.

## Canonical report identities

- Phase1 GH accepted-v3 report SHA256: `cc5c210ad3a80e2237bb55a4a4cb5f91c260d3f368e745319dd5f2fb92daec3b` (64자 actual SHA)
- Phase1 user-final-v5 report SHA256: `b93391e5c344bd5635fdd1c2f6532d41e683a88d85dabb3be21e949f05f3891b`
- Phase1 v5 lead endpoint: `FINAL_W10_REEVALUATION_ALL_B1_TO_B10`; `POOLED_IMMEDIATE_POST_W`는 lead 표에서 제거
- P4 source manifest SHA256: `3fd6e52fe838e1bff003af9d6bb83b23e661ba26ba5b831bd935bc1530ebf032`
- P4 rooted receipt SHA256: `ed47bc390722f9411537452e4e70f1e0a760a67db123bca589c612fbf1fd2ce4`

## 제외 클래스

- raw logs/results/checkpoints/datasets/model weights/cache/projector/statistics/tensors/generations
- P4 transfer tar 및 복제된 request/evaluator payload
- superseded technical roots, failed/partial endpoints, imputed data
- active job worktree/result의 미커밋 또는 실행 중 bytes
- credentials, host-local session boundary, scheduler state

## 남은 순서

1. SH2 order1 및 SH4 order2 완료를 확인한다.
2. 최신 `origin/main`을 fetch하고 이 branch를 그 위에 갱신한다.
3. `034b72c...` final Phase2/3 source와 `7a0aebd6...` pre-edit 최종 commit을 통합한다.
4. focused compile/report rehash/access gate 후 non-force push한다.
5. Phase2/3 terminal-valid 보고서는 완료 뒤 별도 non-force commit/push한다.
