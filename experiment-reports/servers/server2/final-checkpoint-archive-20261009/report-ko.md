# Server2 최종 checkpoint 보존 정책 수락

수신 nonce: `USER-GH-ALL-SH-FINAL-CHECKPOINT-ARCHIVE-SERVER1-20261009-R1-SERVER2`.
정본 main `12eff7e36e1128dba2daa9a463ea1cc09563de46`의 policy/전송·조건부삭제 승인 전체를 읽었다.
정책 SHA256 `aa7cc7ee9bb0bf1a08a183d7807ac2699d8cc4a09b28e1b90fbf50f955a202d6` 일치.

사용자가 이어서 **“이건 앞으로의 job에 대한 정책이다.”**라고 범위를 명시했다.
따라서 **앞으로 새로 제출하는 job에만 적용**하며 이미 등록·실행된 job과 그 checkpoint는
이번 지시로 소급 전송하거나 삭제하지 않는다. SH2는 global policy를 편집하지 않았다.

GH가 이 정정을 정본 main `f9a6084ab050c1283ff3910e75c2032a81bc5997`에 게시했다.
새 policy/approval/정정 envelope 전체를 읽었으며 유효 policy SHA256은
`6da0d1fcb576a51610daf3a733b896cf48a373dc60a3d8aa3c2ec1405ecfd9da`다.
SH2 cutover 기록은2026-10-08 16:40:19 UTC(2026-10-09 01:40:19 KST)다.
정정 이후 신규 제출0. 기존 PENDING/RUNNING/terminal job 모두 제외하며 나중에 완료돼도 KEEP이다.

수락과 구현·전송·검증·삭제는 별개다. 현재는 `ACCEPTED_PROSPECTIVE_ONLY`이며,
SH1의 직접 owner ACK는 공유 helper 구현/독립 CPU 검토 중이며 API/receiver `NOT_READY`를 확인했다. 향후 caller 결속은
`SOURCE_INPUT_PENDING`이다. 공통 helper는 SH1 단독 소유이며 SH2 중복 구현은 없다.
repo app-server direct로 SH1 API 입력 요청 및 SH1/GH에 최신 USER 범위 정정을 전달했다.
bounded direct acceptance는 받았으며 장기 응답/과학 완료/대형 전송을 기다리지 않는다.

향후 적용 순서는 최종 계산·consumer 종료 → exact source allowlist/bytes/fullSHA/stat/provenance
→ SH1 용량·unique archive admission(최소64GiB reserve/수신동시1) → staged one-way transfer
→ SH1 fullSHA/bytes/identity·별도 object 독립검증 및 atomic `VERIFIED_DESTINATION`
→ source 불변·consumer 종료 재확인 → 정확 regular payload만 개별 unlink다.
symlink/hardlink/alias/동일 object/사용 중/변경/검증 실패는 KEEP/HOLD다.
source 최신 metadata/provenance와 archive-location·삭제 receipt는 남기며 로컬 resume payload가
남아 있다고 허위 기록하지 않는다. noCP 작업에 새 저장 권한을 추가하지 않는다.

추가 USER 정정 이전 한정 metadata 확인은 current official qualification의 알려진51경로뿐이었다.
입증된 final W20 후보0; payload 크기·실행 상태·consumer 종료는 미확인이다.
Qualification B2→B3 증빙은 final W20이 아니며 과거 checkpoint 일괄 선별/정리는 하지 않았다.
61619/61624..29와 old61534..40/61428/source/raw를 그대로 보존했다.

**실제 전송0 files/0 bytes, 목적지 검증0, 삭제0 files/0 bytes, 회수 allocated bytes0.**
GPU/model/fit/CP 생성·Slurm 신규등록/조회·대형 SHA 전수계산0.
새 science source/봉인 archive/job 변경이나 recurring monitor/heartbeat/retry는 없다.
상세 근거: `audits/servers/server2/final-checkpoint-archive-20261009/acceptance-r1.json`.

`NO_BROADCAST_NOT_REQUIRED`: 현재 소형 정책 receipt만 Git 공유하고 payload/RNG/tensor/credential은
Git 또는 W&B에 올리지 않았다. 원 dirty root/다른 worker 변경/모델·C0/P/dataset/raw/log KEEP.
