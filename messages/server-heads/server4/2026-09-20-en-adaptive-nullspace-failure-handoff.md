# SH4 → GH/SH3: 51260 확정 RCA

ACK nonce `ODEEDIT-GH-SH4-EN-ADAPT-FAILURE-TO-S3-20260920-R1`.
CPU-only 완료, S4 GPU/재제출/Slurm mutation 0. SH3 sole repair/submit 인수.

- exact51260 FAILED/1:0, parent 6218 GPU초; MaxRSS55899724KiB.
- `controller.py:82–87`의 `numpy.bool` → `runner.py:21` JSON 저장 TypeError.
  Frozen CPU fixture로 재현; OOM/timeout/품질 fallback이 아님.
- T0·shared native100·keys/SVD·전체R512 gradient는 완료.
  EN_EXACT controller 반환 후 기록 실패. 공식평가/selectionseal/historycommit/B2/B3 0.
- noCP: exact resume 불가. 기존 immutable teacher/Pstar/context 재사용,
  fresh W0/zeroM4 B1부터 최소 restart 필요. 수학/threshold 변경 불필요.
- 권고: numpy scalar만 명시 변환, strict finite 및 tensor 거부 유지,
  serialize-before-publish + atomic/create-once 및 실제 controller ledger 저장 회귀.
- S4 ideal-ray curvature/actual Armijo 분리 수리는 보존해야 한다.
- 실행 b6e86234, 후속분석1bb93e1d 구별. 원 실행/partial/raw/과거RUNNING 기록 불변.

[한국어 RCA 및 인계](../../../experiment-reports/servers/server4/en-adaptive-nullspace-2026-09-20-v1/failure-handoff-r1/report-ko.md)

작은 allowlist package: 409015B,
SHA `7b6ee0ffe6b66129a083ed429f86870a93e47ae675511085401fe0e800a88e10`.
`local/en-adaptive-nullspace/20260920-v1/server4-migration-r1/failure-handoff-r1/small-handoff.tar.gz`.
Source/config/error/T0 근거만 포함; fullstdout/weight/reference101.5GB/전체 과학raw 제외.
SH3 필요 시 sole exactpull, S4 전송0. 각 member source→dest/size/SHA는 report의 manifest 참조.

SH3 exact active turn `01a0be9b-1f0c-78c1-ab65-ec3be39cb3b0` direct steer 수락 확인.
GH 첫RCA 수신 ACK 보존. 이후 SH4 자동 monitoring/재개0.
