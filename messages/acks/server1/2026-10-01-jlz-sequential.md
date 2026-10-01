# SH1 JLZ 수신 / M0

Nonce: `ODEEDIT-GH-SH1-JLZ-BS100X10-20261001-R1`.
SH1 session `01a04939-f93a-7b50-bca0-65438eab2062`; root `/mnt/raid5/janghj/ODE-edit`.
실제 작업은 clean child `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-jlz-sequential-bs100x10-20261001-v1`.
전용 branch `codex/server1-jlz-sequential-bs100x10-20261001-v1`, 기준 main `9d111dcba18aa49c5693fd70324e5046cbd9d32e`.
ignored child boundary 검사 PASS; 공유 root 설정/dirty 변경 없음.

## FULL_READ / identity

새 envelope/design/contract/cells, pilot v2 design/contract와 원 첨부 v1 전체를 읽었다.
정본 31 member SHA/bytes 일치. Envelope `1d12ed1deadd3de74282ea4821513f544b819a5850e2cae3a539b9292c48cafa`,
contract `47068bb3375d175f42f3c19194c402f432c939c7a08446eb0a9040fefdfe1f2e`,
design `6228ec0c8dbd9d8f36b0c1d92d6518aa69d61e27a340e5b7ea08d18935ac83c7`를 독립 확인했다.
pilot runtime/prompt/solver, native compute_z, canonical evaluator/token contracts를 직접 읽었다.

## 현재와 실행 계획

이 기록 시점 `IMPLEMENTING_NOT_SUBMITTED`, job ID 없음, actual BS100 `NOT_RUN`.
W0/H0부터 첫1000 하나의 B100×10 chain, 120 whole-batch oracle/batch(최종 재평가 포함).
비수렴 상태를 보존한 fixed-budget 반환 후보 admission과 원자 W/H/RNG/observer transaction을 새 namespace에 구현했다.
최초 actual 비교는 3회 oracle(full/suffix/shared-selected), 최대 허용6 이내. science1200과 별도계수.
MB2 고정 출발, key/C0/FP32·FP64 순서와 native KL 방향 불변.
W0 1000 RPN; 각 current100 RPN, 매 batch seen RP, W5/W10 seen RPN을 기록한다.
checkpoint_saved=false; exact_resume=NOT_AVAILABLE. 기존 task/Server3/원raw 변경0.

자원은 task1GPU/projectcap2, CPU8/131072MiB/devbox/exportNONE/Requeue0, wall48h(ETA 아님), hour budget null.
partition gpu MaxTime30일, 48h 이내를 확인했다. 제출 직전 active+admitted pending을 재계수한다.
디스크 최초관측 여유 약779GiB; 계획reserve16GiB. 실제 자산/packing/source lock은 실행 전에 별도 생성한다.
실측 BS100 비용이 아직 없으므로 수치 ETA를 만들지 않았다.

독립 red `/root/jlz_red` 검토를 수행했다. 첫 검토의 게시후반 I/O ambiguity와 collector 분모/label 누락을 수정했고 재검토 중이다.
CPU39 실행 PASS는 pilot 중복/상속 테스트를 포함하며 실제 GPU PASS가 아니다.
B1 commit/5 append/observer/B2 exact entry까지만 관찰하고 그 뒤 pause한다. afterany CPU collector는 자체 검산을 계속한다.
