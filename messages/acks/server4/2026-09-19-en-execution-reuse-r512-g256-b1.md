# SH4 FULL_READ / M0 — EN R512/generated256, B1 only

- Instruction `ODEEDIT-S06-EN-EXECUTION-REUSE-R512-G256-B1-SH4-V1`, nonce `ODEEDIT-GH-SH4-EN-REUSE-R512-G256-B1-20260919-R1`.
- Clean worktree/branch `codex/server4-en-execution-reuse-r512-g256-b1-v1`, base `af343b72f4ad7d63984a9c4ec986f6d0238f754f`; shared root and previous task artifacts unchanged.
- Design SHA `b671e1eb26d01068b561cf32144d66291f43c8f11505d550e6945229b43b16bd`, contract `84dc6d7b35afb21cb095606a490132ccfc3b72bdbd5ac6ef3de2d241ce1700a0`, design-checks `34a1e5a00a55e3bb6c5f61ef8315a8825e7231043e71d02b49d3f280c19c675d` read in full. Envelope, inherited EN numeric contract, source alltoken/binding/runtime/runner/sequential_runner/geometry/optimizer and linked completed report/compute read. Unchanged PROTOCOL exact prior FULL_READ binding reused (`af806a449be800251393bfcd81b2dfa5689ee34305f3bf1323e0fae82f16c87b`). Design checks are not actual Llama validation.
- Primary pair is LEGACY_SCHEDULE_R512_G256 versus REUSE_SCHEDULE_R512_G256, same R512 input SHA `507b202108acc90e10810455f57da60df14e9e9e31ba567c8c75f51cf76afaeb`; same immutable native/geometry, independent method gradients/trials. Old S64 and BPCW choice-QP are not primary arms.
- Single cold B100 only. `max_batches=1`, `sequential_authorized=false`, `auto_continue=false` and negative tests implemented. No B2 submission/dependency/callback. Continue through B1 comparison/report, not merely initial gate.
- New generated256 teacher/capsules: NOT_BUILT. Endpoint session/observation/strict teacher components under CPU implementation; actual model runner and numerical validation NOT_READY/NOT_RUN. Old16 teacher is not relabeled. Native BPCW same-host B1 capsule is a reuse candidate, not yet a completed reuse audit.
- Runtime requires existing task-local transformers4.44.2 dependency path, torch2.9.1+cu128. Default environment transformers4.57.1 is not used. Shared environment unchanged.
- Project cap2; task1 GPU/job, two schedules sequential on same GPU/runtime; 8CPU/60416MiB/exportNONE/Requeue0. New jobs0, model forwards0 at M0. Wall/source/resource lock awaits complete implementation.
- Initial shared volume free54.49GiB; preparation observation `2026-09-18T18:15:45.553994+00:00` free139.8708GiB. No files deleted/moved by SH4, reason for shared-space change not attributed. Maximum teacher/key payload95.15625GiB; preliminary CP/gradient/geometry/atomic total103.15356GiB, before Current caches/raw/serialization overhead. Current bound and submission reserve remain to finalize; this is not an exclusive reservation or inherited storage waiver.
- No old T skip/FD skip/noCP/storage waiver/deletion/cancellation authority inherited. Existing EN/BPCW artifacts preserved. Same-host NO_BROADCAST_NOT_REQUIRED.
- Local receipt `/data/janghj/ODE-edit/local/en-execution-reuse/20260919-v1/receipts/preparation-m0.json`; authoritative exact copies in sibling `authoritative/`.

## 2026-09-19 완료 리뷰 FULL_READ / M0 추가 기록

Instruction `ODEEDIT-S06-EN-REUSE-R512-G256-B1-COMPLETED-REVIEW-SH4-V1`, nonce `ODEEDIT-GH-SH4-EN-REUSE-G256-B1-DETAILED-REVIEW-20260919-R1`의 명시적 recall을 적용했다. 새 envelope 전체 SHA `b65ae8307f51b3b2020e57d2fe4d5442caedf12881764fe9cc433605154d11b6`, 원 envelope/실제 제출·pause·admission/effective lock을 읽었다. 위 design/contract/checks/PROTOCOL FULL_READ는 현재 exact SHA를 대조해 재사용했다. 새 분석대상 source는 직접 읽었다.

- clean 분석 branch `codex/server4-en-reuse-g256-b1-completed-review-v1`, base `2e9fd379bc441c7a33503a5af5fb384bc46c0054`. shared root/기존 WT 변경0. 기존 own implementation `a2b31eb`을 충돌 없이 보존 통합했으며 실행source와 구분한다.
- 정확50410/50449 accounting 한 번: 둘 다 COMPLETED/0:0, allocated GPU-sec 각각15476/7696. 과거 PENDING/Dependency 관측은 수정하지 않는다. 새 sibling/task 조회0.
- 실제 B1 execution `5574f2c63a355043ba28c8e557383be3075a5e48`, effective lock SHA `01db7d4a9afb5bc6da9632967b32d4015613e1b5e7e6f324705858af7bae35af`; 준비execution a297039와 구분한다.
- 첫 actual 독립표 N4/Legacy/Reuse 모두 RS100/100, PS194/200, NS865/1000. 최초표 SHA `ef53af2e4c7400782d359a8cf888ea1f7d1274c6edd91e3e18cc7e6f78065281`. 첫표 당시 전체 state/source 감사 완료를 주장하지 않았다. 이후 metadata 표시수정과 모든 검산은 새 completed-review-v1에만 기록한다.
- 리뷰 CPU-only, 신규 GPU/model/evaluator/Slurm mutation/전송/삭제0. `max_batches=1`, sequential 미승인. 별도 독립 agent red는 이번 리뷰에 사용하지 않았다.
