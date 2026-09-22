# 사용자 승인: 원본 AlphaEdit checkpoint server2→server4

Instruction/nonce ODEEDIT-GH-SH4-ALPHA-KEY-CAUSAL-20260923-R1.
사용자 원문과 실행 권한은 messages/head/2026-09-23-alpha-key-causal-sh4.md.
담당: SH4 sole pull/destination writer. SH2 source는 read-only, 원본 KEEP.
전송 허가는 주어졌으며 동일 승인 재요청 불필요. 실제 복사/검증은 아직 미실행.

Source host rke-server2:
/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/new177-v1/payload/local/fixed10k-native-baselines/attempt-v1/output/main-cell-1/

Destination host rke-server4:
/data/janghj/ODE-edit/local/alpha-key-concentration-causal/20260923-r1/inputs/checkpoints/BASE_ALPHAEDIT/

원본 checkpoint-transfer-manifest.json
SHA3fdc3b06c4689dad6756258a3a4d240531848a71f2088b7706379d7ff52aac1b.
B001/B005/B010/B020/B030/B040/B050/B060/B070/B080/B090/B100의
W-method-state.pt 12개 / 63,418,321,276 bytes만 승인한다.
각 file의 expected SHA/size는 원 manifest이며 변경하거나 이름만으로 대체하지 않는다.
GH host에 대형 checkpoint를 중간 저장하지 않는다.

SH4가 source exact path/owner/regularfile·destination collision·현재 disk를 확인하고
명시 file list로 rsync --partial-dir=.rsync-partial 또는 동등한 보존형 partial copy.
--delete/기존 완성파일 overwrite/원본 삭제 금지. 이미 destination fullSHA 일치면
REUSED_VERIFIED. 다른 bytes 충돌은 보존·별도 namespace 또는 block.
전송완료와 content검증완료는 별도 receipt; 완성 확정 전 destination SHA/size 필수.
contexts.json/commit.json/native-observation.json sidecar는 필요한 정확12시점/root
scope에서 존재·SHA inventory를 만든 뒤 추가 allowlist로만 수신한다.
전체 output root 재귀 복제와 unrelated/live task raw 접근은 금지한다.

소형 design/evidence23개는 동명 .json approval, GH→S4 source KEEP.
Dataset/model/P/target/native closure는 S4 exact asset 재사용 우선.
누락시 frozen original execution lock에 명시된 exact member만 source경로와
destination/hash/bytes를 receipt에 고정하여 선택 수신하도록 승인한다.
다른 dataset이나 current main source로 대체 금지, 공유환경 설치/변경은 포함하지 않는다.

별도 input복사와 명시 진단 K/R/Δ factor 저장 외 신규 edited state checkpoint는
기본 미저장이다. 기존 checkpoint나 타실험 자료 삭제권한은 없다.
검증: transfers/verifications/alpha-key-causal-20260923-r1/와 scoped raw receipts.
