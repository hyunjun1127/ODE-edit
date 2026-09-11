# Server4 checkpoint의 Server2 보존 이관 및 source copy 제거

## 결과

사용자 승인에 따른 CPU/I/O 저장소 작업이다. 과학 결과·점수·실행 설정을 변경하지 않았다. 원본 보고서와 sealed manifest는 그대로이며, 과거 source checkpoint 경로는 아래 migration map으로 해석해야 한다. 빈 placeholder나 원격 symlink는 만들지 않았다.

|항목|파일 수|논리 bytes|
|---|---:|---:|
|기존 Server2 보존본 재사용 후 Server4 제거|72|62011141768|
|신규 이관·독립 검증 후 Server4 제거|111|178165006043|
|전체 제거|183|240176147811|
|우선 대상 중 보류|0|원본 보존|
|추가 smoke inventory: 이번 이관 외 보존|4|6341797900|

완료 lifelong 14 chains의 168 CP와 BLUE 1k 3 chains의 9 CP, JVP-L8 1k 2 chains의 6 CP를 서로 구분했다. Full pretrained model이 아니라 선택 weight와 method history checkpoint다. Native baseline은 L4–L8의 5개 선택 weight, BLUE는 L4+L8의 2개, single-layer는 해당 1개를 보존한다. JVP-L8는 지원 write가 L8이어도 저장 inventory는 원래 5개 weight/history이므로 임의로 한 layer만 추출하지 않았다.

## 보존 위치와 복원

- 기존72: `/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1/` 원래 위치 유지.
- 신규105: `/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/new177-v1/`.
- JVP1k6: `/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/jvp1k-v1/`.
- 파일별 source→destination/SHA/size: `transfers/verifications/2026-09-11-checkpoint-migration-server4-source/migration-map.csv`.

Server4 삭제는 로컬 영구 unlink이다. 검증된 Server2 retained_path가 복구 원본이다. 복구 시 이 CSV의 SHA/size를 검증한 뒤 필요한 파일만 별도 사용자 복원 지시에 따라 복사한다. Server2 보존본을 별도 승인 없이 삭제·이동·덮어쓰면 안 된다. 원본 보고서의 과거 raw 전체 검산 명령은 삭제된 경로에서 그대로 실행되지 않으며 이 migration map이 필요하다.

## 검증과 경계

1. 최신 승인 main58034b67d1ddb6962796b2448240f4c51a1948a1/treecb905c7169047be3d4c53de9a3eaf2292e8fed96를 별도 clean worktree에서 FULL_READ하고 원문 create-once 보존.
2. exact terminal/report binding과 현재 source full SHA/size, regular/non-symlink/owner/single-link/dev/inode/mtime 확인. 대상19 jobs만 단발 COMPLETED0/queueempty 확인. 다른 task의 scheduler/source/process monitoring 없음.
3. SH4 유일 rsync writer. exact files-from, private staging, ignore-existing. --delete/--remove-source-files/blanket mirror0. 승인된 S4→S2 전용 archive이므로 일반 다중서버 broadcast helper 대신 범위 제한 rsync 예외를 receipt에 기록.
4. SH2 독립 full destination SHA/size 및 closure 확인 후 VERIFIED_DESTINATION receipt. 기존72도 과거 receipt만 재사용하지 않고 현재 재검증. 새로운 staging final rename은 SH2 소유.
5. 삭제 직전 source 재해시·stat 및 own-user open fd/mmap 확인. SH1은 Server4 원본 현재/예정 의존성이 없다고 직접 확인. OS 로그인 daemon(systemd/sd-pam/sshd)의 privileged fd는 관측 불가로 구분했고 experiment children은 별도 검사. 알 수 없는 접근 불가 프로세스나 exact consumer는 HOLD한다.
6. Bundle 검증 완료 후 명시된 절대 파일명 개별 unlink. 디렉터리/root/glob 삭제0. 논리 bytes와 filesystem 가용 증가를 별도 기록.

관측한 파일시스템 가용량 증가 합은 240176029696 bytes이다. 공유 filesystem의 동시 활동이 있으므로 이 차이 전부를 본 삭제만의 배타적 회수량이라고 주장하지 않는다.

## 복원 수준

Lifelong checkpoint는 선택 W+Alpha selected M 또는 MEMIT empty/static-state sentinel+context/RNG/identity를 담는다. 기존 exact-SHA CPU weights_only/tensor audit를 결속했고 필요한 source/config/base/P/stats references를 함께 보존했다. Alpha M은 edit history이며 forward weight로 적용하지 않는다. MEMIT에 새 history를 만들지 않았다.

BLUE1k checkpoint는 W/M/context를 복원할 수 있으나 당시 RNG state가 저장되지 않았다. Seed 설정만으로 bitwise sequential continuation을 검증했다고 주장하지 않는다. JVP 형식은 별도 원본 receipt/schema에 결속한다. GPU model load/forward/continuation replay는 전부0이며 CPU reload와 실제 continuation parity를 혼동하지 않는다.

## 보존 자료와 추가 범위

Code/worktree/.git, 모든 기존 보고서·CSV·PNG·manifest·receipt, 평가 NLL/log, native targets/chronological companion, HF 모델/tokenizer/P/covariance/stats/dataset, BLUE/EasyEdit 원본, 타 서버 복사본은 제거하지 않았다. Companion은 copy-only이고 Server4에 남는다. 중지 ORBODE audit도 재개·변경하지 않았다. 추가 smoke4 CP는 destination 검증 묶음에 넣지 않아 원본을 보존했다.

## 재현·증거

Local control: `/data/janghj/ODE-edit/local/checkpoint-migration-server2/20260911-v1/`. Source/deletion plan, 독립 destination receipts, exact unlink journal, transfer commands/stdout 통계와 source scripts가 create-once 보존된다. Git에는 아래 raw-free mapping와 checksum만 포함한다. Tensor/prompt/log payload는 Git0.

검증 실행은 해당 control의 `inventory.py`, `jvp_inventory.py`, `delete_verified.py`, `transfer.py` SHA로 결속된다. 삭제 script는 재실행 지시가 아니며 이미 삭제된 source를 대상으로 재실행하면 identity guard에서 중단해야 한다. Mapping 검사만 read-only 수행한다. Scientific promotion=false. 완료 또는 명시적 bundle HOLD 후 STOP; 새 실험/모니터링0.
