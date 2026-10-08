# SH1 최종 checkpoint 보존 수신

`USER-GH-ALL-SH-FINAL-CHECKPOINT-ARCHIVE-SERVER1-20261009-R1-SERVER1` 직접 수락.
최신 policy main `f9a6084ab050c1283ff3910e75c2032a81bc5997`, file SHA
`6da0d1fcb576a51610daf3a733b896cf48a373dc60a3d8aa3c2ec1405ecfd9da`
전체와 갱신된 conditional-transfer/deletion 승인문을 읽었다.
최신 USER 정정 `이건 앞으로의 job에 대한 정책이다`를 우선 적용한다.
앞으로 신규 제출한 job의 source/registration에 사전 policy adoption을 결속한
최종 checkpoint만 대상이다. 이미 등록된 job과 기존 checkpoint는 이 지시로
archive/삭제하지 않고 KEEP한다. 과거의 일반 범위를 역사적 허가로 확대하지 않는다.

SH1 수신 cutover는 `2026-10-08T16:54:50Z`로 기록했다. 이는 직접 범위 ACK 뒤
clock 관측 시각을 사용한 보수적 경계이며 기존 job을 새 제출로 relabel하지 않는다.

SH1은 단일 수신·예약·독립 fullSHA/bytes 검증 owner다. 공유 storage API source
`0a536420bc47870980ff7c1449ab5d721f762534`, namespace tree
`f71d01ca0f486e27f28cbf554a6589f9dfe72a5a`를 봉인했다.
owner와 별도 reviewer가 임시 CPU 파일 기반 46개 회귀검사를 각각 PASS했다.
최신 실제 policy·server1 cutover exact member로 production constructor만 검산했다.
이 API 준비는 실제 payload admission·전송·수신 검증·source 삭제 완료가 아니다.
확인된 RAID 여유1294375956480 bytes/inode335039550는 단발 준비 snapshot이고
특정 checkpoint admission 승인이 아니다. exact manifest마다 active 예약/staging과
payload/companion·64GiB reserve를 다시 검사한다. 동시수신 기본1, GPU0이다.

실제 final 계산·writer/consumer 종료·batch20/finalW20·pointer/identity·regular nlink1
증거 없는 파일은 KEEP한다. source/receiver size/fullSHA·독립 object 검증 뒤 atomic
VERIFIED_DESTINATION이 있어야 source owner의 재검산과 개별 unlink를 허용한다.
afterany/rsync exit0/collector PASS/filename은 이 증거를 대신하지 않는다.
symlink/hardlink/shared alias/sameobject/변경/미완료/용량 부족이면 삭제0.
원 metadata/provenance/raw/log/latest pointer는 보존하며 archive-location receipt를 별도로 남긴다.
server1 자체 CP는 이미 중앙서버에 있으므로 등록/보호하고 unlink0이다.
별도 사본이 있어도 source-delete gate는 server1 payload 삭제를 거절한다.

API는 `Receiver.from_policy_file` → `seal_manifest` → `admit` → source-owner exact
one-way transport → `verify`/`recheck` → `source_delete_gate`다. source deletion은
구현하지 않았으며 마지막 함수도 ELIGIBLE 메타데이터만 반환한다.
`project/run_scripts/checkpoint_archive/README.md`에 정확 signature·7개 companion·
pre-submit adoption/lock·actual submission·최종 계산/모든 consumer 증거 계약을 기록했다.
원 scientific evidence schema를 이해하는 source-owned frozen adapter와 source/receiver의
trusted cutover가 없으면 `ARCHIVE_PENDING_KEEP_SOURCE`다. 일반 PASS Boolean이나
afterany 종료만으로 source의 원 계산/consumer 증거를 대체할 수 없다.
수신 측은 전송된 작은 원 증거도 독립 fullSHA/read하고 create-once receipt에
전체 sealed manifest/복원 출처를 보존한다. 이는 GPU/과학 독립 인증이 아니다.

현재 SH1 새 official pipeline-r2에는 아직 실제 job/CP/완료 READY가 없다.
SH2는 앞선 자기 bounded inventory에서 finalW20 후보0을 보고했으며, 이는
기존 checkpoint admission 요청이 아닌 future caller 준비였다. 기존 중간/qualification
checkpoint와 등록된61619/61624..29/61534..40/61428은 KEEP한다. 전송0/삭제0을
접수했으며 이를 다른 서버의 전체 현재 상태로 확대하지 않는다.
기존 official/source/raw/job 및 GPU 준비는 계속하며 archive 준비를 과학 성능 gate로 쓰지 않는다.
실제 source/API와 미래 caller-only 사용법을 게시 후 SH2/3/4에 직접 전달한다.
peer cutover와 실제 source adapter 연결은 미완료이며, 이 입력 대기를 기존
과학 성능 gate나 추가 사용자승인 대기로 바꾸지 않는다.

전송0 bytes, 삭제0 files/0 bytes. tensor/model/RNG/CP payload Git/W&B 업로드0.
무관 파일 이동·삭제, broad cleanup, 새 GPU/fit, 기존 frozen job 변경, recurring monitor,
heartbeat, 자동 retry는 하지 않았다.
