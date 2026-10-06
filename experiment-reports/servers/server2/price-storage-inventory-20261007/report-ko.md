# Server2 PRICE 기준 읽기전용 저장공간 점검

삭제 권한 **없음** (`delete_authorized=false`). 파일 삭제/이동/압축/전송/덮어쓰기/권한변경0, GPU/Slurm/W&B 신규 실행0. 원 dirty1391항목 보존.

## 기준 및 경계

실제 server2 / SH2 session01a0493a-074c-7f91-9a13-769116326fef / `/mnt/raid5/janghj/ODE-edit`. 현재 registry/ignored boundary와 일치한다. 별도 non-main worktree 사용.

PRICE MEMIT source87a5a736 및 Alpha source019922b1의 현재 게시된 실행 lock은 Server4 경로다. S4 실물/결과/queue를 읽지 않았다. Server2에 같은 모델/통계/프로젝터가 있어도 S4 실행의 직접 입력이라고 추정하지 않는다.

FE frozen config의 정확 asset 경로와 기존 migration-map을 우선 사용했다. FE 및 W&B spool은 보호하고 실험 진행률/완료 여부는 조회하지 않았다.

## 분류별 중복 제외 용량

|분류|파일 수|논리 GiB|할당 GiB|tensor 논리 GiB|
|---|---:|---:|---:|---:|
|PRICE_REQUIRED|0|0.000|0.000|0.000|
|OTHER_TASK_REQUIRED|542|18.930|18.932|3.828|
|REPRODUCTION_KEEP|35774|238.660|238.734|220.977|
|UNREFERENCED_CANDIDATE|0|0.000|0.000|0.000|
|UNKNOWN|16295|137.554|137.588|137.210|

dev/inode 중복 경로 0개 제외. 파일명/크기 일치는 동일 bytes 증거가 아니다. 확정 회수가능 용량은 **미확정**이며 실제 삭제0B.

기존 이관183CP: 현재 metadata present 111/183, size 일치 111, 역사 논리합 240176147811B. 기존 SHA는 재사용 표시만, 전수 재해시0. 원 S4 제거 후 보존 계약이므로 REPRODUCTION_KEEP.

## 큰 파일 상위20 (논리 크기 순)

|경로|논리 GiB|할당 GiB|분류|
|---|---:|---:|---|
|`/mnt/raid5/janghj/ODE-edit/local/results/official-layer-realization-debt-lifelong-b100-v1/campaign-20260901-v1/qwen2.5-7b-inst-alphaedit-lifelong-b100x100/checkpoints/state-05000.pt`|7.949|7.949|UNKNOWN|
|`/mnt/raid5/janghj/ODE-edit/local/results/official-layer-realization-debt-lifelong-b100-v1/campaign-20260901-v1/qwen2.5-7b-inst-alphaedit-lifelong-b100x100/checkpoints/state-03000.pt`|7.949|7.949|UNKNOWN|
|`/mnt/raid5/janghj/ODE-edit/local/results/official-layer-realization-debt-lifelong-b100-v1/campaign-20260901-v1/qwen2.5-7b-inst-alphaedit-lifelong-b100x100/checkpoints/state-10000.pt`|7.949|7.949|UNKNOWN|
|`/mnt/raid5/janghj/ODE-edit/local/results/official-layer-realization-debt-lifelong-b100-v1/campaign-20260901-v1/qwen2.5-7b-inst-alphaedit-lifelong-b100x100/checkpoints/state-01500.pt`|7.949|7.949|UNKNOWN|
|`/mnt/raid5/janghj/ODE-edit/local/results/official-layer-realization-debt-lifelong-b100-v1/campaign-20260901-v1/qwen2.5-7b-inst-alphaedit-lifelong-b100x100/checkpoints/state-07500.pt`|7.949|7.949|UNKNOWN|
|`/mnt/raid5/janghj/ODE-edit/local/results/official-layer-realization-debt-lifelong-b100-v1/campaign-20260901-v1/qwen2.5-7b-inst-alphaedit-lifelong-b100x100/checkpoints/state-02000.pt`|7.949|7.949|UNKNOWN|
|`/mnt/raid5/janghj/ODE-edit/local/results/official-layer-realization-debt-lifelong-b100-v1/campaign-20260901-v1/qwen2.5-7b-inst-alphaedit-lifelong-b100x100/checkpoints/state-01000.pt`|7.949|7.949|UNKNOWN|
|`/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/jvp1k-v1/payload/local/state/alpha-jv-migration-server4-20260907/tech-r1/chain-5-qwen2.5-7b-inst-L8_ONLY_NATIVE/checkpoints/W1-M1.pt`|7.949|7.949|REPRODUCTION_KEEP|
|`/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/jvp1k-v1/payload/local/state/alpha-jv-migration-server4-20260907/tech-r1/chain-5-qwen2.5-7b-inst-L8_ONLY_NATIVE/checkpoints/W5-M5.pt`|7.949|7.949|REPRODUCTION_KEEP|
|`/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/jvp1k-v1/payload/local/state/alpha-jv-migration-server4-20260907/tech-r1/chain-5-qwen2.5-7b-inst-L8_ONLY_NATIVE/checkpoints/W10-M10.pt`|7.949|7.949|REPRODUCTION_KEEP|
|`/mnt/raid5/janghj/EasyEdit/examples/null_space_project_Qwen2.5-7B-Instruct.pt`|6.685|6.685|REPRODUCTION_KEEP|
|`/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/new177-v1/payload/local/fixed10k-native-baselines/attempt-v1/output/main-cell-1/B100/W-method-state.pt`|4.922|4.922|REPRODUCTION_KEEP|
|`/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/new177-v1/payload/local/fixed10k-native-baselines/attempt-v1/output/main-cell-1/B090/W-method-state.pt`|4.922|4.922|REPRODUCTION_KEEP|
|`/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/new177-v1/payload/local/fixed10k-native-baselines/attempt-v1/output/main-cell-1/B080/W-method-state.pt`|4.922|4.922|REPRODUCTION_KEEP|
|`/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/new177-v1/payload/local/fixed10k-native-baselines/attempt-v1/output/main-cell-1/B070/W-method-state.pt`|4.922|4.922|REPRODUCTION_KEEP|
|`/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/new177-v1/payload/local/fixed10k-native-baselines/attempt-v1/output/main-cell-1/B060/W-method-state.pt`|4.922|4.922|REPRODUCTION_KEEP|
|`/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/new177-v1/payload/local/fixed10k-native-baselines/attempt-v1/output/main-cell-1/B050/W-method-state.pt`|4.922|4.922|REPRODUCTION_KEEP|
|`/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/new177-v1/payload/local/fixed10k-native-baselines/attempt-v1/output/main-cell-1/B040/W-method-state.pt`|4.922|4.922|REPRODUCTION_KEEP|
|`/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/new177-v1/payload/local/fixed10k-native-baselines/attempt-v1/output/main-cell-1/B030/W-method-state.pt`|4.922|4.922|REPRODUCTION_KEEP|
|`/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/new177-v1/payload/local/fixed10k-native-baselines/attempt-v1/output/main-cell-1/B020/W-method-state.pt`|4.922|4.922|REPRODUCTION_KEEP|

## Coverage와 불확실성

등록 WT 67개, regular metadata 52578개, 관측 제한/누락 0건, 오류 0건. 상한180초/400000files/metadata24MiB. 상세는 coverage.json. symlink 미추적, 외부 HF는 등록 모델 cache root만, EasyEdit는 등록 통계와 정확 P 파일만 metadata 식별.

UNKNOWN은 PRICE 비관련/안전삭제 판정이 아니다. 기존 task/재현 raw/유일 사본 가능성을 배제하지 못했으며 전부 KEEP. 실제 live open-FD/전체 참조 graph/다른 host 의존성은 미점검. 모델/C0/P/고정data/context/코드/Git/W&B spool은 보호한다. NoCP 정책은 과거 checkpoint 삭제 권한이 아니다.

파일별 전체 metadata: `/mnt/raid5/janghj/ODE-edit/local/price-storage-inventory-20261007/full-inventory-final.csv` (local-only). Git inventory.csv는 top20+분류별 큰 tensor20개씩의 소형 발췌이고 summary의 집계는 전체 scan 결과다. root별 aggregate는 summary.json. 현재 FS available 512800018432B는 공유 FS 순간값이며 본 작업이 확보한 공간이 아니다.

재현: `nice -n 5 ionice -c 3 python3 audits/servers/server2/price-storage-inventory-20261007/inventory_scan.py` (create-once, 자동 재실행 없음). metadata 외 tensor 내용읽기 없음. 후속 삭제는 별도 사용자 승인과 정확 대상 검증이 필요하다.

NO_BROADCAST_NOT_REQUIRED. 소형 보고만 게시 후 STOP; 다른 실험의 monitoring pause 유지.

## Exact 경로 후속 metadata 확인

이관 목록 중 미관측 72개를 exact lstat한 결과 ENOENT 72개. 역사 합 46508999784B. arm별 {'MEMIT_L5_ONLY': 12, 'AlphaEdit_L5_ONLY': 12, 'MEMIT_L6_ONLY': 12, 'AlphaEdit_L6_ONLY': 12, 'MEMIT_L7_ONLY': 12, 'AlphaEdit_L7_ONLY': 12}. 이는 현재 목록 경로의 부재 사실이며 삭제 주체/시점/원인이나 다른 경로의 복사본 존재 여부는 확인하지 않았다. 본 점검 삭제0. 초기72 및 나머지관측111의 크기일치와 혼동하지 않는다. 정확 경로는 missing-migration-paths.csv.

기존 jvp reference closure에 이미 등록된 Qwen/공용자산 exact metadata 11파일을 보완했다(20.876GiB). symlink를 따라 탐색하지 않고 receipt의 명시 regular realpath만 사용했다. 본표는 이 보완을 포함하며 원 scan/새 fullinventory를 모두 local 보존했다.
