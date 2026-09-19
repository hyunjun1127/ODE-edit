# 승인: archived L4-only companion의 Server4 → Server2 선택 수신

Instruction: ODEEDIT-S06-S2-CHECKPOINT-MECHANISM-AUDIT-20260920-V1.
사용자가 첨부 지시문에서 원 target·평가·source/dependencies staging을 명시 승인했다.
SH2가 sole destination writer/puller이며 S4 원본은 read/hash/stat만 한다. S4 모델·행렬 분석/GPU/실험 변경0.

## Exact 허용 source

- /data/janghj/ODE-edit/local/blue-lifelong-b100x100/attempt-checkpoint-r2/output/main-cell-3/
  - B001..B100: current.json, commit.json, entry.json, native-observation.json, contexts.json
  - B001,B005,B010,B020,B030,B040,B050,B060,B070,B080,B090,B100: seen-full.json
  - B001,B002,B006,B011,B021,B051,B091: native-targets.pt
  - runtime.json, terminal.json 및 위 파일의 봉인된 manifest/receipt
- /data/janghj/ODE-edit/local/blue-lifelong-b100x100/attempt-checkpoint-r2/
  execution.lock.json, local-source.tar, lifelong/의 원 실행 코드
- /data/janghj/ODE-edit/local/blue-lifelong-b100x100/attempt-v1/
  sample.lock.json, configs/AlphaEdit-L4_ONLY.json
- /data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/
  blue-source/, source-tech-r2/, deps-transformers-4.44.2/
  에서 실제 원 lock/import closure에 필요한 code/config/dependency 파일만.
- /data/janghj/EasyEdit/data/counterfact/counterfact.json:
  S2 원 dataset identity가 맞으면 재사용; 누락/불일치일 때만 새 task inputs로 수신.

S2 기존 initial6-v1의 source/evaluator/lock/12W-M와 local pinned model/P는 먼저 재사용한다.
S2 W/M 재전송0, 무관 source tree/cache/venv 전체복사0.
HFmodel/P/기존 C4 R512-G256/다른 run raw는 이 전송 승인 대상이 아니다.

## Destination와 실행

/mnt/raid5/janghj/ODE-edit/local/checkpoint-mechanism-audit/20260920-v1/<new-attempt>/inputs/
와 같은 attempt의 source/deps 하위만. 기존 원 run 디렉터리를 destination으로 쓰지 않는다.
Exact source/relative/destination/bytes/SHA allowlist를 전송 전에 잠근다.
예상 eval+7targets448,927,340B, sample 포함457,710,625B; source/deps/metadata 별도 계측.
공유FS free/inode/temp/storage 여유 점검, missing만 수신, create-once/atomic completion,
destination SHA/size와 state/request/source 결속 확인. Source KEEP/no delete/no overwrite/no credential.
진행 중인 SH4 task/outputs 접근0. 기존 승인된 SSH/helper 경로로 전송하며 정확 local mapping만 사용.
Generic helper의 명시된 source 범위 지원 한계는 기록하고 전역 정책을 완화하지 않는다.

수신 검증 receipt:
transfers/verifications/2026-09-20-checkpoint-mechanism-server4-to-server2/
및 task local inputs manifest. 중복 수신/반복 전체해시를 피하고 검증 receipt를 재사용한다.
