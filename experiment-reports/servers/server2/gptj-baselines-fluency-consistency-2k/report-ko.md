# GPT-J native six-baseline fluency/consistency 준비 보고

상태: `CPU_PREPARATION_INPUT_PENDING_NOT_SUBMITTED`. 실험 완료·실제 generation/GPU PASS·Slurm PENDING이 아니다.

Instruction/nonce: `USER-GH-ALL-SH-BASELINE-FLUENCY-CONSISTENCY-RERUN-20261007-R1`.
권한 main `b0cee1a302c10d48e206a31c7c5cf318ac36d8c3`; SH2/server2/session `01a0493a-074c-7f91-9a13-769116326fef`.
전용 branch `codex/server2-gptj-baselines-fluency-consistency-2k`, 원 dirty root 보존.

## GH 입력 요청 전달

사용자 “GH에게 이걸 전달해”에 따라 SH1 evaluator publication/source commit·API와 reference bundle READY·exact path/SHA/size 요청을 공식 server-head에 게시했다. 요청 게시 main `43b5ff84b26b279a48c9d72f52d65b8e86a8c6b1`의 원격 exact를 확인했다. 앱 직접 전송 도구는 unavailable을 반환했고 codex_app MCP도 사용 불가였다. **직접 전달·GH 열람/ACK는 확인되지 않았다.** Git 게시를 live 전달 성공으로 표시하지 않는다.

요청은 추가 실행 승인 요청이 아니다. SH1 소유 `experiment_generation_eval` actual API/source와 `attribute_snippets.json`, `idf.npy`, `tfidf_vocab.json`, tokenizer provenance의 정확한 READY가 필요하다. 확인한 origin/main에는 공통 source namespace가 없었고 승인된 로컬 입력 경로에서 reference 3종도 확보되지 않았다. SH2가 공통 구현/TF-IDF를 별도 생성하거나 대체하지 않았다.

## 기존 baseline 취소 대조

현재 task의 exact 후보 8개를 단발 squeue/sacct 및 기존 source/manifest와 대조했다. 전부 terminal이므로 **이번 scancel 0건**이다.

| ID | 역할 | 실제 상태 |
| --- | --- | --- |
| 60656 / 60657 / 60658 | stock MEMIT / AlphaEdit / collector | COMPLETED |
| 60769 / 60770 | CAKE / AlphaEdit-BLUE | FAILED |
| 60771 / 60772 / 60773 | PRUNE / RECT / collector | CANCELLED |

과거 실패·취소 receipt 및 raw는 불변 보존했다. PRICE/ours, W0, FE와 타 서버/사용자 job을 취소·변경하지 않았다. 위 사실은 서버 전체 queue가 0이라는 주장이 아니다.

## 구현·CPU 관측

기존 stock source `56d3a445553b60bf1a5e33f0e820e0699364ba0b`와 four-arm source `3a4a107b7ae4c66f2f7a7f0cb441d26c5a639f68`의 model/revision/runtime/native closure/hparams/C0/P/input/scorer 결속을 재사용했다. 작은 source SHA와 이전 heavy-asset fullSHA + 현재 stat 수준이며 모델/tensor load·자산 복제는 0이다. 실제 first2000/20×100, generation prompt 20000개, R/P/N current100/200/1000를 metadata로 확인했다.

CAKE/BLUE fc_out는 bias가 존재하는 stock `nn.Linear`를 허용하고 bias object/storage/version을 보존한다. 첫 B1 기존 forward에서 출력 shape/dtype/finite를 관측하도록 수정했으며 추가 GEMM/forward/fit는 없다. native objective/solve/hparams 원본은 수정하지 않았다. CPU bias fixture는 통과했지만 actual affine numerical parity는 `NOT_MEASURED`다.

새 runner는 여섯 원 native factory를 재사용하고 자기 W/H를 이어간다. native25 evaluation/24 update 및 방법별 layers/hparams/history를 유지한다. PRUNE은 W20 post 평가 전에 `PRUNE_TERMINAL_BASE_FIX`를 한 번 적용하고 실제 `explicit_repair` receipt key를 사용한다. 새 telemetry private client/worker는 기존 immutable transport의 import/sidecar/metric-axis 연결만 변경한 task-local scalar 확장이다. 공유 helper는 수정하지 않았다.

좁은 CPU 44 tests PASS(0 failures/0 errors), wrapper elapsed 2.574초, peak RSS 824004608B, CPU threads1, CUDA initialized=false. 구성: bias6, six-arm mocked dispatch7, cap/count plan5, scalar/fake SDK26. 모델/native fit/generation/network/Slurm는 모두 0이다. 별도 focused source reviewer와 fake-SDK reviewer를 사용했으나 pretrained/actual generation red PASS는 아니다. bool 평균값 허용 및 readiness가 model load 뒤에 확인되는 두 CPU/source 결함을 수정했다.

## 실행 계획과 실제 미제출 경계

6 cold arms × first2000 BS100×20, 합계12000 edit applications/unique2000. BASE_MEMIT이 cold W0 generation의 sole publisher다. 그 job 종료 뒤 cap2 두 lane은 `BASE_ALPHAEDIT → ALPHAEDIT_BLUE → RECT` 및 `CAKE → PRUNE`; 더 엄격한 cap1이면 전량 직렬화한다. 모든 기존 surviving GPU frontier afterany와 새7개 held owner/argv/source/config/resources/dependency 검사가 실제 제출 때 필요하다. 성능 gate/afterok는 없다.

현재 job IDs는 `[]`, source freeze/held registration/release는 **미실행**이다. 계획 요청은 각 GPU1/CPU6/59392MiB/48h, collector GPU0/CPU6/24576MiB/4h, project/task cap2다. 48h는 ETA가 아니다. 기존 hardware metadata는 A6000 49140MiB 및 disk 여유 약446GiB였지만 최종 admission/동시 peak 인증을 대신하지 않는다.

계획상 arm당20pre+20post generation endpoint, milestone4회 current는 동일 prefix cases subset으로만 집계한다. 총53600 case observation 참조(고정10 prompts이면536000 prompt observation 참조)에는 cache reuse가 포함되며 신규 forward 수가 아니다. B1 pre600 cases는 exact cold W0 subset 재사용 대상이다. 같은 observation의 두 metric을 위해 다시 생성하지 않는다. 비용·peak·실현 유효 분모는 아직 미측정이다.

SH1 API/reference 이후 남은 부분은 exact allowlist 수신·검산, SH2 thin bridge, 실제 case/metric sum-count/status/profile/state/RNG-finally-restore 결속, final config/source/resource lock, 6GPU+CPUcollector 제출/release 및 actual raw generation reducer다. 공통 source readiness 없이 bridge API를 추정하거나 fixture를 actual PASS로 표시하지 않는다. 신규 source는 현재 provisional preparation이며 production executable freeze가 아니다.

NoCP/exact_resume=NOT_AVAILABLE. 원 source/raw KEEP, raw/text/token/model/H/secret Git/W&B0. 신규 online W&B run도 아직 없으며 과거 run backfill/hotpatch는 하지 않았다. agent recurring monitor/heartbeat/auto retry는 없다.

## 기록·재현

- 공식 입력 요청: `messages/server-heads/server2/gptj-baselines-fluency-consistency-2k.json`.
- compact CPU/source receipt: `audits/servers/server2/gptj-baselines-fluency-consistency-2k/cpu-preparation.json`.
- ignored CPU receipt: `/mnt/raid5/janghj/ODE-edit/local/gptj-baselines-fluency-consistency-2k/cpu-checks-preparation-r2.json`, SHA256 `340f6b22ece83c3458ccd166851bb5c60b0ec5bd0f449dbced8989adae547016`.
- provisional config SHA256 `dfd6f8059519ac0bfec7e8bf92ce21d2e21341681f75da4a5494084357e9d294`; metadata receipt SHA256 `890e735897001f9c3860d6652647b5d575a1704f49a8d03696e82fc69a0fdecf`.
- exact narrow CPU command와 immutable 신규 rN receipt 사용은 `project/run_scripts/gptj_native_baselines/GENERATION_RERUN_README.md`에 기록했다.

`NO_BROADCAST_NOT_REQUIRED`: 이번 단계는 소형 Git 요청/코드/receipt만 공유했다. reference bundle·raw·credential 대형 전송은 0이다.
