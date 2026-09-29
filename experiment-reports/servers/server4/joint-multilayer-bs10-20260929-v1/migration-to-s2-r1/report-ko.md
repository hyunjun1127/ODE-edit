# S4 → S2 다층 BS1 이관 완료 기록

Nonce `ODEEDIT-GH-SH4-SH2-JOINT-BS1-MIGRATION-20260929-R1`. 최신 USER 요청은 “SERVER4는 GPU 리소스가 부족해서 SERVER2로 가자.”다.

S4 실행 소유권은 중지됐다. 정확 owner janghj(1025)/submission/source/tree/lock/launcher 전체argv를 결속한 뒤, CPU **55091 먼저**, array **55090_0..8** 순서로 취소했다. 직접 확인 시 모두 PENDING이었고, 취소 후 모두 CANCELLED 및 한정 queue active/pending0을 확인했다. 다른 job/원 source/raw/CP/model은 변경·삭제하지 않았다.

|Job|Name|취소확인 상태|elapsed초|AllocTRES|
|---|---|---|---:|---|
|55091|odeedit_joint_collect_s4|CANCELLED by 1025|0|없음|
|55090_0|odeedit_joint_bs1_s4|CANCELLED by 1025|0|없음|
|55090_1|odeedit_joint_bs1_s4|CANCELLED by 1025|0|없음|
|55090_2|odeedit_joint_bs1_s4|CANCELLED by 1025|0|없음|
|55090_3|odeedit_joint_bs1_s4|CANCELLED by 1025|0|없음|
|55090_4|odeedit_joint_bs1_s4|CANCELLED by 1025|0|없음|
|55090_5|odeedit_joint_bs1_s4|CANCELLED by 1025|0|없음|
|55090_6|odeedit_joint_bs1_s4|CANCELLED by 1025|0|없음|
|55090_7|odeedit_joint_bs1_s4|CANCELLED by 1025|0|없음|
|55090_8|odeedit_joint_bs1_s4|CANCELLED by 1025|0|없음|

취소확인 시각 `2026-09-29T09:45:01.011522+00:00`. Array parent 물리 ID55090의 여러 child를 allocation 중복합산하지 않으며 실제 parent GPU비용은0초다. scientific model/fit/partialstate/snapshot0, 취소 후 output directory 비어 있음. 복원 대상 실행state가 없어 rollback은 NOT_APPLICABLE_NO_ALLOCATION이다. 취소 후 과학 모니터링은 하지 않았다.

## 소형 handoff

압축 `/data/janghj/ODE-edit/local/joint-multilayer-bs10/20260929-v1/migration-to-s2-r1/handoff-small.tar.gz`, **464540B**, SHA `0c9422297d550288f0d9e32daa83b9ac7d177ca2c84ff2bc4714a98eda58a5ad`.
안전한 regular member **49개/2096027B**, member별 size/fullSHA 및 archive 재열기 검산을 수행했다. CP/model/P/teacher/weight payload0. 원 config/lock/source archive·launchers/CPU29 receipt/USER override/derived BS1 batches/snapshot표/CP binding/context·hparams/CPU native import 소형 source를 담았다.

Receiver exact allowlist `/data/janghj/ODE-edit/local/joint-multilayer-bs10/20260929-v1/migration-to-s2-r1/receiver-exact-allowlist-s2.json`, SHA `dbc3180d680a5a534b19de2084147f280f217adca949f68bc774bcc03005ccf2`. Receiver root는 `/mnt/raid5/janghj/ODE-edit/local/joint-multilayer-bs1/20260929-v1/inputs/server4-handoff`. GH 승인 아래 SH2 sole pull/sourceKEEP/nooverwrite/nodelete이며 S4는 원격 쓰기·대형 역전송0이다. 현재 기록은 HANDOFF_READY이며 SH2가 수신했음을 이 보고가 주장하지 않는다.

S4 CPU29는 fixtures/import/token 검산 수준이며 GPU/model actual PASS가 아니다. 원3CP/model은 S2 기존 실물을 우선 재사용한다. S2 경로/launcher/환경/자원 port와 actual integrated 검증은 SH2 소유다. S4의 source에는 절대경로와 torch2.9.1+cu128/TF4.44.2 guard, 넓은1801file inherited binding이 있으므로 SH2가 사용 closure/reuse bridge를 명시해야 한다. 이 문서가 S2 환경 적합성을 확인한 것은 아니다.

## Provenance 및 보존

- 실행 source `6b07104954920bb4c5478a88ebed478dcb9527df`, tree `9ea717d43e9b693285ed58b6885b607a7dc71f6d`.
- 원 publication `7a1b59cfe1814ab4498e703343735c0eeb6b8927`는 runtime commit과 구별한다.
- lock `4a64f85fffed70a901e22613f8b52a3e9730ccf362238d84246f188296032f95`, config `d903f36882086482ddceee410e00eb6f4ec98f935a37292b5ee3592ee115ec47`.
- cancellation receipt `2f934836622526bc5555ede5ea54e185323968a8f6f3cba507dd07f5c54f683d`.
- 원 main BS10 정본/USER BS1 override/이전 PENDING 인계/단층STOP/source·입력은 그대로 보존한다.

GH 첫 direct 메시지는 복수 active turn으로 COMMUNICATION_HOLD였지만, 이후 GH가 저장 cancellation receipt를 exact SHA로 읽었음을 main d15ddd1f의 delivery receipt에서 확인했다. 재승인 대기·강제 steer·타task 재개0.

[취소 receipt](../../../../../audits/servers/server4/joint-multilayer-bs10-20260929-v1/migration-to-s2-r1/cancellation-receipt.json), [handoff receipt](../../../../../audits/servers/server4/joint-multilayer-bs10-20260929-v1/migration-to-s2-r1/handoff-receipt.json). Full scheduler stdout/config/context는 local만 유지하며 Git에는 축약 사실·경로·hash만 게시한다.

STOP_MIGRATED_TO_SH2. monitoring_active=false / automatic_resume=false / S4 new submit·release·retry0. 이전 단층 task STOP 유지. NO_BROADCAST_NOT_REQUIRED: 필요한 소형 인계 외 대형원본 재방송0.
