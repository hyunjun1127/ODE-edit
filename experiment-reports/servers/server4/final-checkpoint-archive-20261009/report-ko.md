# server4 최종 checkpoint 정책: 미래 job 한정 수락

nonce `GH-SH4-FINAL-CHECKPOINT-FUTURE-ONLY-20261009-R1` 수락.
첫 기록 cutover는 **2026-10-09 01:44:16 KST / 2026-10-08 16:44:16 UTC**.
ACK와 시각 기록 사이 신규 제출은 없다. 정본 `f9a6084a`의 policy SHA256
`6da0d1fcb576a51610daf3a733b896cf48a373dc60a3d8aa3c2ec1405ecfd9da`를 검산했다.

- 이후 새로 제출하는 job만 source/config에 결속한다. 이미 등록된 PENDING/RUNNING/
  완료 job과 기존 CP는 나중에 완료되어도 이번 정책으로 archive admission·전송·삭제하지 않는다.
- PRICE/replay/official 기존 job 및 frozen source는 보존한다. noCP 작업에 CP 저장을 추가하지 않는다.
- 미래 해당 CP도 최종 계산/consumer 종료, 정확 allowlist·identity·fullSHA/bytes,
  SH1 공간·inode·예약·64GiB reserve, 별도 사본 및 독립 VERIFIED_DESTINATION,
  원본 불변/consumer 재검산 이후에만 개별 payload unlink 대상이 된다. 오류 시 KEEP.
- SH1 shared receiver/API는 현재 main `03eefd27`에 없다. 복제 구현하거나 전송을
  시작하지 않았으며 future adoption 상태는 `ARCHIVE_PENDING_KEEP_SOURCE`다.

정책 수락과 실제 구현·전송·검증·삭제는 별도다. 이번 신규 job/기존 job 변경/
archive admission/전송/수신 검증/삭제 **모두 0**. CP inventory도 수행하지 않았다.

공유 W0 `100f49d7`에 대한 GH review를 읽었다. generation 점수/분모,
CF stream/token/work 결속 관련 BLOCK을 반영하여 **수리본의 GH 검토/main 게시와
실제 READY 전까지 production 차단을 유지**한다. own `abb902e7` 및 원 source는
보존하며 CPU fixture PASS를 actual READY로 승격하지 않는다. 별도 full W0 job0.
