# GH → SH4 / SH2: server4 checkpoint를 server2에 보존한 뒤 source copy 제거

- instruction_id: ODEEDIT-S06-SERVER4-CHECKPOINT-MIGRATION-SERVER2-V1
- nonce: ODEEDIT-GH-S4-S2-CHECKPOINT-MOVE-20260911-R1
- 사용자 승인: “현재 계산해놓은 checkpoint들은 필요하다 ... 2번 서버로 옮기고 4번 서버에 존재하는 checkpoint들 제거하자.”
- 기준 main: 6e98103842c9ebae39003264a328f30db1916a57
- GH session: 01a04939-8873-7673-8dca-4c7fc5e31af0
- SH4 source owner: 01a04939-b5c7-7a03-ba2d-ef3343d62cfd, /data/janghj/ODE-edit
- SH2 destination owner: 01a0493a-074c-7f91-9a13-769116326fef, /mnt/raid5/janghj/ODE-edit
- 저장소: hyunjun1127/ODE-edit. CPU/I/O storage task이며 GPU/model/evaluator/edit/Slurm mutation 권한 없음.

## 1. 실행 권한 및 소유

SH4는 source inventory·사용 중 여부 확인·유일한 payload rsync writer·server4 exact-file 삭제를 담당한다. SH2는 server2 용량/기존 복사본 확인·독립 full SHA/size 검증·보존 catalog를 담당한다. GH는 본 지시/전송 승인 및 완료 수신만 담당하며 중복 raw/hash/code 감사를 하지 않는다. SH끼리 registered peer-direct로 manifest→destination-ready→verified-receipt를 교환한다.

최종 목표는 단순 복사가 아니라 **server2에서 복원 가능한 checkpoint가 검증·보존된 exact source 파일만 server4에서 제거**하는 것이다. 본 범위에서 반복 사용자/GH 승인 대기를 추가하지 않는다. 부족 용량·불명확한 소유·활성 consumer·hash mismatch는 해당 bundle을 HOLD하고 안전한 독립 bundle은 진행한다.

각 SH는 별도 branch/worktree/control directory를 사용한다. 다른 agent/사용자 변경을 되돌리거나 덮어쓰지 않는다. 작업 완료까지 자율 실행하되 타 task의 모니터링을 재개하지 않는다.

## 2. source 범위와 제외

우선 완료 14 chains/168 CP publication을 사용하여 실제 checkpoint allowlist를 만든다. 과거 count는 현재 파일 존재·동일성 증거가 아니다. 다음은 탐색 가능한 task roots이며 **디렉터리 전체 삭제 허가가 아니다**.

- /data/janghj/ODE-edit/local/blue-lifelong-b100x100/attempt-cell0-checkpoint-r3/
- /data/janghj/ODE-edit/local/blue-lifelong-b100x100/attempt-checkpoint-r2/
- /data/janghj/ODE-edit/local/blue-lifelong-b100x100-l567/attempt-v1/
- /data/janghj/ODE-edit/local/fixed10k-native-baselines/attempt-v1/
- /data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/
- /data/janghj/ODE-edit/local/blue-alphaedit-l4-oneshot-sequential/attempt-v1/
- /data/janghj/ODE-edit/local/blue-alphaedit-l8-oneshot-sequential/attempt-v1/
- /data/janghj/ODE-edit/local/state/alpha-jv-migration-server4-20260907/tech-r1/

Run/publication manifest로 확인된 실제 experiment checkpoint selected-weight/history/optimizer/state shard만 삭제 후보에 넣는다. 취소·superseded prefix도 보존 가능하나 canonical 결과와 별도 상태를 유지한다. 위 root 밖의 추가 checkpoint는 정확한 경로·크기·소유만 inventory에 보고하고, 소유/용도가 불명확하면 삭제하지 않는다. 일반 *.pt/*.bin 검색 결과를 전부 checkpoint로 취급하지 않는다.

삭제 금지:
- code/worktree/.git, 모든 보고서·CSV·PNG·manifest/checksum·작은 receipt
- HF pretrained/tokenizer, P/nullspace, C0/covariance/stats, fixed dataset, BLUE/EasyEdit 원본
- raw evaluation NLL/log/비checkpoint 결과
- 다른 서버 checkpoint, 현재 작업의 입력 또는 실행 중 작성/사용되는 파일
- 중지 ORBODE audit 자료; 이미 삭제된 ORBODE raw는 복구/재실행하지 않음

직접 관련 exact job/process/consumer에 대한 bounded read-only 확인만 허용한다. RUNNING/PENDING job이 source path를 필요로 하거나 open fd/작성 중이면 HOLD_IN_USE. cancel/hold/restart/throttle 변경으로 소비자를 제거하지 않는다. 현재 SH1 two-memory source/input/process에는 접근·변경하지 않으며 필요한 경우 소유 SH의 path-use 확인만 요청한다.

## 3. 보존 범위 및 기존 copy 재사용

가중치와 결속된 method history/optimizer shards, shard index, context/RNG/config/base revision/source/input identity 등 재구성에 필요한 closure를 함께 보존한다. 필요한 작은 companion은 copy-only로 server4에도 남긴다. Native-target/cache/journal이 복원에 필요하면 copy-only 포함하고 checkpoint와 임의로 동일시하여 삭제하지 않는다. 공용 pretrained/P/stats는 server2의 정확한 자산 reference로 재사용하며 불필요한 전체 재전송은 하지 않는다.

Server2 새 archive 권장:
 /mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/
Create-once private staging에 source-run namespace/상대경로를 보존한다. 기존 다른 root이면 고유 suffix 사용. 기존 파일 overwrite·old sealed package rewrite 금지.

기존 72 CP 복사본 후보:
 /mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1/
과거 checkpoint-manifest SHA e4625ab025e6bf57c30a5c3a1e6eece01557cd2d204c3266a37777368368884a, payload 72 files / 62011141768 bytes.
SH2가 현재 실제 파일 전체 size/SHA 및 source manifest 결속을 독립 검증한 뒤 재사용한다. 기존 downstream import는 이동/rename/overwrite하지 않고 해당 원래 path를 지속 보존 대상으로 catalog에 등록한다. 새 별도 keep/retention receipt는 기존 sealed bytes 밖에 둔다. Server1에만 있는 복사본은 이번 삭제 조건이 아니다.

Source old path→server2 actual retained path/SHA mapping을 남긴다. 기존 report는 수정하지 않는다. 빈 placeholder .pt나 원격 symlink를 만들지 않는다. Legacy source raw 검증 명령은 migration map을 참조해야 함을 작은 README/receipt에 명시한다.

## 4. 자원·전송·검증 후 삭제 순서

1. SH4 exact-file manifest: realpath, owner, type, nlink, dev/inode, size/mtime, SHA256, run/arm/batch, W/M/source identity. Regular non-symlink owned single-link 파일만 기본 삭제 후보로 한다. Hardlink/예상 외 symlink/path escape는 HOLD한다.
2. SH2 free disk/inodes, 작업별 기존 reserve, 재사용 제외 신규 전송량을 계산한다. 다른 자료 삭제로 공간을 만들지 않는다. 부족하면 완료 bundle별 이관/삭제 또는 PARTIAL_CAPACITY_HOLD를 보고한다.
3. SH2는 기존 copy를 full SHA/size 검증하거나 new staging을 마련하고 READY를 보낸다. SH4가 유일하게 exact allowlist/--files-from/--relative 방식으로 rsync한다. --delete, --remove-source-files, blanket mirror 사용 금지. Partial copy는 READY가 아니다.
4. SH2는 모든 destination 파일 full SHA256/size를 source manifest와 독립 비교한다. 기존 CPU checkpoint schema 감사는 exact SHA 결속 후 재사용할 수 있다. 새 schema 불명 파일만 필요한 CPU weights_only 확인; 새 GPU continuation replay는 하지 않는다.
5. 재구성 closure와 참조 가용성을 확인한 SH2가 staging을 atomic seal하고 VERIFIED_DESTINATION receipt를 보낸다. Receipt는 양쪽 full paths, count/bytes, source+destination manifest SHA, retained location과 closure/미검증 경계를 포함한다. 재사용 copy도 동일 수준 receipt 필수.
6. SH4는 receiver receipt를 결속하고 삭제 직전 source identity(dev/inode/size/mtime/안정된 SHA)와 in-use 여부를 다시 확인한다. 변경/불확실한 파일은 삭제하지 않는다. Bundle 전체가 검증되지 않았으면 부분 state를 삭제하지 않는다.
7. SH4만 exact validated absolute filenames를 개별 unlink한다. 재귀 directory/root/glob 삭제 금지. Source checkpoint 제거는 사용자 승인에 따른 로컬 영구 삭제이며, 검증된 server2 복사본으로 복구 가능해야 한다.
8. 파일별 removed/held/failed 및 이유, logical bytes 합과 실제 filesystem available 증가를 분리해 기록한다. Source·destination mapping과 삭제 receipt를 양쪽에 보존한다. 공간 확보를 과장하거나 열린 inode의 logical bytes를 즉시 free라고 주장하지 않는다.

안전한 묶음별 수신 검증→삭제는 허용하며 이미 정확한 72 copy를 재사용할 수 있으면 중복 전송 없이 먼저 처리할 수 있다. 한정 storage 체크는 허용하되 다른 실험 scheduler/result monitoring이나 새 자동화는 만들지 않는다.

## 5. 산출물·Git 권한·완료

SH4 local control: /data/janghj/ODE-edit/local/checkpoint-migration-server2/20260911-v1/
SH2 local control: /mnt/raid5/janghj/ODE-edit/local/checkpoint-migration-server4/20260911-v1/
각각 source/deletion-plan, capacity/transfer/destination-verification, closure, migration-map, deletion/completion receipt를 보존한다. Credentials/raw tensor/prompt/log payload는 Git에 올리지 않는다.

SH4 raw-free write:
- audits/servers/server4/2026-09-11-checkpoint-migration-server2/
- messages/server-heads/server4/2026-09-11-checkpoint-migration-server2.md
- transfers/verifications/2026-09-11-checkpoint-migration-server4-source/
- experiment-reports/servers/server4/checkpoint-migration-server2-2026-09-11-v1/factual-migration-ko.md

SH2 raw-free write:
- audits/servers/server2/2026-09-11-checkpoint-migration-server4/
- messages/server-heads/server2/2026-09-11-checkpoint-migration-server4.md
- transfers/verifications/2026-09-11-checkpoint-migration-server2-destination/

각 tasks/status/server4-checkpoint-migration-server2-v1/<server>.json도 허용한다. 본인 완료한 scope만 clean integration/non-force main push한다. 충돌 시 사용자 변경을 덮어쓰지 않는다. Raw 방송은 이번 exact S4→S2 이관만 허용하며 제3서버 중복 전송은 없다.

SH4 최종 통합 보고: 재사용/신규 이관/삭제/보류 count·bytes, 실제 확보 용량, source→S2 복원 경로, CPU 검증 범위/미실행 GPU replay, 다른 자산 무변경, source 삭제와 복구 방법. SH2는 destination 보존 확인을 별도 보고한다. 완료 또는 명확한 부분 HOLD 보고 후 STOP. Scientific outcome 수정/승격은 없다.
