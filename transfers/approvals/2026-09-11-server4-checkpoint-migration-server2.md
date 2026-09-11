# GH 승인: checkpoint 보존 이관 및 검증 후 server4 exact source copy 삭제

approval_id: ODEEDIT-S4-TO-S2-CHECKPOINT-MOVE-20260911-V1
instruction_id: ODEEDIT-S06-SERVER4-CHECKPOINT-MIGRATION-SERVER2-V1

사용자는 server4 checkpoint를 보존하면서 server2로 옮기고 server4 copy 제거를 명시 승인했다.
상세 범위·예외·순서는 messages/head/2026-09-11-server4-checkpoint-migration-server2.md 전체가 구속한다.

- Registered rke-server4 → rke-server2, SH4 sole payload rsync writer.
- SH2 independent destination full SHA256/size/closure VERIFIED receipt 이후에만 SH4 exact manifest regular owned non-symlink single-link checkpoint unlink.
- 기존 initial6-v1 72 copy는 실제 독립 재검증 후 in-place reuse, retention catalog 필수.
- Reports/code/shared model/P/stats/dataset/비checkpoint raw/source directories 삭제 금지.
- Live/pending consumer 입력·작성 중·scope/identity 불명 파일은 HOLD.
- Rsync --delete/--remove-source-files, overwrite, recursive/glob delete 금지.
- Source 삭제는 server4 로컬 영구 삭제이며 server2 verified copy가 복구 원본으로 남는다.
- 명시된 source roots와 정확한 task checkpoint에 한정. Broad workspace/home/cache cleanup 불허.
- GPU/model/evaluator/Slurm/source scientific mutation0. 해당 storage 체크 외 모니터링 재개0.
- SH4/SH2 own raw-free receipt/report scope main non-force push 허용. GH 중복 raw audit/재승인 대기 없음.
