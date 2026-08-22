# Server1 실험 코드·보고서 main 통합 인벤토리

상태: `ORDER3_RELEASED_INTEGRATION_VALIDATED_PENDING_PUSH`

초기 감사 기준 `origin/main`은 commit `f4fbfb5de132de9b986e4b291c5b41784c182743`, tree `8c3f444014907f69b1bb793674f7a5a9e2b9ee0e`였다. ORDER3 release 뒤 commit `4f9a7ec492b1bb937b7424ba32611a05ac5565a0`, tree `30b41f6d558b055a701889612e39c8a984a525db`를 독립 fetch/검증하고 clean branch `codex/server1-all-experiment-integration-v1`을 그 위로 rebase했다. active Phase2/3 job source, result root, 사용자 dirty worktree는 변경하지 않았다.

## Lineage 판정

| 범위 | 최종 server1 ref / tree | 기준 main 조상 여부 | 통합 판정 |
|---|---|---:|---|
| PIR-U Alpha-cache continuity | `e4a5b8ff95717548af6d164f29fbc62ffb0c26c2` / `59ea4bc470656769e7590665d125b30106c86145` | Yes | 이미 main; 중복 통합하지 않음 |
| P1R52 target + Official AlphaEdit writer | `7ef609c7c1d3267ddbda624afd52bce8241acc4b` / `ccffb6ec4c53a10762885ebc6f7752cde7026370` | Yes | 이미 main |
| Residual-Reserve final repaired lineage | `6caafd1148498497602876a9693f6760e6b39e85` / `9e2c8aac6d21b3dedc7770d4de894e1b73d3c097` | Yes | 이미 main |
| Joint P+C FULL-FP32 independent 10×B100 final-v6 | `96494a05c37ffb32f101d53d14e0e61b2f25664d` / `58c07ad6ef33571650d92c28c58fb31871915a9d` | Yes | code/report 이미 main |
| C-writer Phase1/2/3 final repaired execution source | `034b72c59aace422b7048509b336bac26992d90e` / `76f3b8e0b3304dcdca25a8467439c4bf035bb41c` | No | 최신 main과 merge 완료; P3 및 C-writer dispatch 동시 보존 |
| Phase1 common W0 pre-edit evaluator | `7a0aebd6f1a4bc728cb8d6ff50470b0d10c7cc39` / `f4b30677116f6b11a3b6fc59f9776bdea341a1f0` | No | 최종 단일 pre-edit commit 통합 완료 |
| Phase1 exhaustive accepted-v3 + final-v5 | local canonical reports | No | 소형 보고서/표/receipt exact-byte 통합 완료 |
| P4 sealed stream 준비 receipt | member root `28189ad33cdcda0c01d269216514508d62b15cd6c16ed053afd4683577169843` | No | source manifest/receipt 통합 완료; transfer tar 제외 |

## 아직 main에 없는 final source 범위

Phase2/3 repair ref `034b72c...`는 `96494a05...` 대비 22 files, 3,452 insertions, 7 deletions이다. 공통 K-step orchestration, cache policy, dependency gate, 세 phase launcher/sbatch/dry-plan 및 focused tests만 포함한다. 실행에 사용된 이 ref보다 앞선 `251e616c...`, `762c4788...` 등은 lineage 구성 commit이며 별도 canonical 복제 대상으로 취급하지 않는다.

Pre-edit는 최종 commit `7a0aebd6...`의 세 파일만 대상이다: evaluator launcher, sbatch, focused test. 중간 add/revert 시도는 제외한다.

## Canonical report identities

- Phase1 GH accepted-v3 report SHA256: `cc5c210ad3a80e2237bb55a4a4cb5f91c260d3f368e745319dd5f2fb92daec3b` (64자 actual SHA)
- Phase1 user-final-v5 report SHA256: `b93391e5c344bd5635fdd1c2f6532d41e683a88d85dabb3be21e949f05f3891b`
- Phase1 v5 lead endpoint: `FINAL_W10_REEVALUATION_ALL_B1_TO_B10`; `POOLED_IMMEDIATE_POST_W`는 lead 표에서 제거
- P4 source manifest SHA256: `3fd6e52fe838e1bff003af9d6bb83b23e661ba26ba5b831bd935bc1530ebf032`
- P4 rooted receipt SHA256: `ed47bc390722f9411537452e4e70f1e0a760a67db123bca589c612fbf1fd2ce4`

최신 main에 이미 있던 canonical identities도 재확인했다.

- PIR-U Stage-A report actual SHA256: `d6c545da36f966b3e376d5524c23fe5e9f084786f7bc3c8ae685b29d87ce1ccf`
- P1R52 target + Official writer final-v2 report/manifest/receipt SHA256: `026db11d1e7c627759fd831592f29e77d1dfb6c5d407cf996ad4b81dfe22333a` / `40704f7025ca89be18c5b169a80a0415bc6f0c5513a7140fd2d74b5037f075b5` / `116fd88b50f317dc1b77146213bdc74ac7e4f4de4e29c278fc563ffa47dc9a57`; receipt identity `629b09d47d1facb4261c0e5afd280bfd1d210fdd53961e92c3172d724bf0ec00`
- Joint P+C final-v6 report/manifest/receipt SHA256: `3ea9739fa9200fd2e9fbf36a7a6b9e86c0212279fc52f9c6b2c0aa6f5f42e217` / `b38d8b23d7d2876e72dcab35c8a64083ba6baa5951769998cf3c1726d9a77d09` / `e82fb1dd0f8c1596823979b63f66168682a15bf231621771dae95340e68c8356`; receipt identity `ea5261b1f4b08732fa7ae08895dbe655d53a630d3c4f5b7bb86a5a74b644ff60`

PIR-U manifest가 report를 `8475` bytes / `a99cc5c5...`로 기록하지만 tracked canonical report는 `8474` bytes / `d6c545da...`다. 차이는 끝의 빈 줄 한 개이며 수치·본문은 동일하다. 이는 commit `bb4a0997...`부터 존재한 inherited metadata discrepancy이므로 ORDER3에서 report나 manifest를 재작성하지 않고 actual Git blob identity를 기록했다.

## Merge와 focused gate

- 충돌 파일은 `p1_runtime.py`, `p1r52_sequential_runtime.py` 두 개였다.
- 최신 main의 P3 role/result-parent와 C-writer Phase1/2/3 role dispatch를 함께 유지했다.
- update-norm 수집은 main의 `pir_policy or target_depth_policy` 확장을 유지했다.
- C-writer/pre-edit strict `-W error` focused tests: `22/22 PASS`.
- 충돌 인접 P3 focused tests: `6/6 PASS` (기존 test-only tensor scalar-conversion UserWarning은 표준 경로에서 관찰됨).
- changed Python `py_compile`, sbatch `bash -n`, `git diff --check`: PASS.
- Phase1/2/3 dry-plan identities: `30a5cb3805c0e3fb9e3a9e5e18425975935c67b2840de44fa12b755b568f4df8`, `b377cafceef70403f82ef1ee7e37fbd9fbcf24ed2b3d28835bebbca4dfb80922`, `6c060faffc431b6d178c97ec33bc3edf94c7aa8efc754c9718c99b52facb7c5b`.
- session boundary: `PASS`, server1 / Sol Ultra.

## 제외 클래스

- raw logs/results/checkpoints/datasets/model weights/cache/projector/statistics/tensors/generations
- P4 transfer tar 및 복제된 request/evaluator payload
- superseded technical roots, failed/partial endpoints, imputed data
- active job worktree/result의 미커밋 또는 실행 중 bytes
- credentials, host-local session boundary, scheduler state

## 남은 순서

1. 최종 branch commit/tree와 최신 remote 상태를 재검증한다.
2. non-force로 `HEAD:main`을 push한다.
3. Phase2/3 terminal-valid 보고서는 완료 뒤 별도 non-force commit/push한다.
