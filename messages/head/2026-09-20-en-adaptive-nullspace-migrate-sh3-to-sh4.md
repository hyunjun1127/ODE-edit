# GH → SH3/SH4: EN adaptive-nullspace B300 실행 소유권 이관

Instruction ID: ODEEDIT-S06-EN-ADAPTIVE-NULLSPACE-B300-SH4-MIGRATION-V1
Nonce SH4: ODEEDIT-GH-SH4-EN-ADAPT-MIGRATION-20260920-R1
Nonce SH3: ODEEDIT-GH-SH3-EN-ADAPT-MIGRATE-TO-S4-20260920-R1

## 1. 최신 사용자 지시 / 소유권

사용자: “server3실험 위한 gpu cap이 없으니 server4에서 돌리라고해. 인계 시켜”.
이 task의 **구현 통합·제출·실행·상세보고/main 소유자를 SH3에서 SH4로 변경**한다.
SH3는 이 task 신규GPU submit/release/retry를 중지하고 구현·입력검산 handoff만 완료한다.
SH3 direct ACK: 본task GPU job 미제출, 제출·release·재시도 중지, 인계본 준비.
따라서 현재 알려진 중복 과학 실행은0이다. SH3 최종 handoff에서 다시 사실을 결속한다.
기존 S3 전역 cap1은 이번 task 실행권한 중지와 별개로 역사/정책 그대로 둔다.

SH4=session01a04939-b5c7-7a03-ba2d-ef3343d62cfd / hostname server4 /
CWD /data/janghj/ODE-edit / origin hyunjun1127/ODE-edit.
GH read-only direct inspect에서 SH4 idle 확인 후 이 과학 task를 정식 전달한다.
SH3=session01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3, ubuntu, same logical repo/CWD.

## 2. 상속할 정본 / 바뀌지 않는 실험

messages/head/2026-09-20-sh3-en-adaptive-nullspace.md와
audits/global/2026-09-20-sh3-en-adaptive-nullspace/authority-manifest.json의 정본9개,
plans/global/2026-09-20-en-adaptive-nullspace-{experiment,contract,cells}-v1,
reference-transfer-confirmation을 FULL_READ 또는 exact prior FULL_READ로 재결속한다.
본 문서는 server/owner/runtime 경로/resource 변경만 우선하며 과학식/예산은 바꾸지 않는다.

- W0/zeroM4, Llama FP32/eager/TF32 off, L4 하나, L2=1/native hparams 유지.
- 기존 fixed10k datasetSHA3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1,
  orderedroot5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729.
  정확 B1[0:100],B2[100:200],B3[200:300]. 순서변경0.
- bounded T0 → B1 N4/EN_EXACT/EN_NUM/EN_ADAPT → N4/EN_EXACT/EN_ADAPT own B2/B3
  → 상세 한국어 사실보고/main. B1의 성능 부호로 sequential arm을 고르지 않는다.
- R512/Dev128 generated256 teacher 정확재사용, reference/history 역할/가중·overwrite 유지.
  B2이후 target/update/history teacher는 자기 trajectory, 타arm 것을 공유하지 않는다.
- z-hook batching/prefix cache/필요위치 head 재사용, no per-document dense gradient D2H,
  동일controller2candidate/epsilon.05/weighted geometry/고정budget 유지.
- RSPSNS와 별개 TF rewrite/rephrase/neighborhood token-micro/prompt-macro/strict,
  true/new/desired NLL, margin, case-paired gained/lost/atwrite 유지변화/clusterbootstrap을 보고.
  Official P/N·새metric은 postseal observer만. 자유생성accuracy로 TF정확도를 부르지 않는다.
- save_checkpoints=false; edited W/M/delta/resume disk0, RAM branch state로 단일persistentlane.
  Exact crash-resume NOT_AVAILABLE. 기존자료 삭제0.
- B1000/10k/추가arm/sweep0. 새 full strict gate/과거T waiver 자동상속0.
  실제 precision 미확립은 원설계 exploratory/NOT_ESTABLISHED로 구분, 가짜PASS0.

## 3. SH3 handoff / SH4 재사용

SH3 구현 branch codex/server3-en-adaptive-nullspace-b300-v1,
최근 사용자보고 commit2ff2393f19b74195ed52d016d4d73ae0878d9946.
project/run_scripts/en_adaptive_nullspace/에 geometry/current/native/controller/selector/
objective/runner/runtime/technical/metrics/tests가 구현되어 있다.
SH3 최종 인계 commit과 완료/미완료를 받아 exact Git object로 가져온다.
CPU43 tests/2486sourcefiles26.68MB/Pstar14326dimension은 SH3보고이며 실제GPU PASS가 아니다.
현재S3 source의 run.sbatch는 ubuntu/H200/121856M/S3venv 경로다.
**이 launcher를 S4에서 그대로 제출하지 않는다.** 새 S4 immutable attempt/lock/launcher를 만든다.
기존 source는 보존하고 필요한 S4 paths/resource/config binding만 격리 변경한다.
수학·metric 변경이 필요해 보이면 최소 technical bug와 새방법 변경을 구분해 선보고한다.

SH3는 source commit/tree·dirty diff, CPU logs/tests·실제 importer/runtime versions,
source/input/teacher/Pstar/metrics manifests, partial transfer/verification coverage,
known issues/미구현·memory/disk/실행계획을 compact handoff로 남긴다.
원raw/partial와 비용 보존. 인계완료 후 STOP, 별도 S3 실행 재개0.
SH4는 이 공개branch/인계자료를 재구현하지 않고 통합한다.
최종 인계가 늦어도 Git source 읽기·S4 binding·CPU 준비는 먼저 진행 가능하다.
SH3 정지 ACK/실제 input binding을 확인한 뒤만 S4 과학 submit한다. GH 추가승인은 불필요하다.

## 4. Reference는 server4 원본 재사용 / 제한된 선택 전송 승인

S4 reference source:
 /data/janghj/ODE-edit/local/en-execution-reuse/20260919-v1/PREP/attempt-v1/
및 기존 transfer approval이 결속한 완료 R512/Dev128 input/teacher/key/residual manifest.
SH3 보고: 2560members/101519959223B, R512130235positions/Dev12832473positions.
이것은 전송·검산 진행 보고이며 전체 완료 검산을 뜻하지 않는다.
같은 원본이 S4에 있으므로 **101.5GB teacher를 S3에서 되돌려 복사하지 않는다**.
원본 manifest/실제 file identity를 결속해 read-only reference 또는 task-local path mapping 사용.
기존 S4 model/P/C0/context/fixed10k/TF4.44.2 runtime도 동일 identity면 재사용.
원본 공유파일 덮어쓰기0; 다른 EN/BPCW/SLMF paused task를 재개하지 않는다.

SH4가 sole destination writer로 아래 S3-only 누락물만 exactpull 가능:
 /data/janghj/ODE-edit/local/en-adaptive-nullspace/20260920-v1/ 안의
 source archive/config/lock/metrics contract/CPU receipt/manifests,
 inputs/pstar-derived-v1/basis.npy 및 그 provenance (S4 exact기존basis가 있으면 먼저REUSE),
 source-imports의 지정hooking/source binding, input JSON/소형 provenance.
S3-only 실제 필요성·source→dest·size/SHA·owner allowlist를 먼저 봉인한다.
기존 teacher/모델/평가 raw/무관 checkpoint 재전송0, --delete/이동/덮어쓰기0.
S3 shareddirty/원본은 KEEP. 전송 전 actual available disk/inode/temp/후속실행 여유 확인.
작은 reference path mapping은 S4 tasklocal에서 만들되 원본 teacher를 수정하지 않는다.
원 source3의 large transfer는 safe boundary에서 중지·partial보존 가능; 정리권한은 아니다.
Reference teacher는 입력이고 no edited-checkpoint 정책의 예외가 아니다.

## 5. Server4 admission / 실행 운영

Server4 project cap2 유지, **이번 task는 단일1GPU lane**.
서버변경만으로 두 arm 병렬/추가 GPU 소모를 늘리지 않는다.
기존 다른 active/admitted pending을 합해 project≤2, task≤1.
각1GPU/8CPU/**host memory≤60416MiB(59GiB)**, node server4/partitiongpu,
exportNONE/Requeue0. S3 119GiB 요청은 S4 상한이 아니며 상속금지.
RAM branch state/teacher streaming/geometry scratch의 실제 peak를 산정한다.
S4에서 상한을 맞출 수 없다면 수학/분모를 줄이거나 체크포인트를 저장해 우회하지 말고
필요 memory와 최소 실행구조 변경안을 보고한다.
S3 dryrun6h는 실측이 아니므로 S4 실제 준비/비용예상 기준으로 wall/resource lock을 새로 봉인.
타job 취소/hold/throttle, 공유환경 변경/기존파일 삭제권한0.

initial/PENDING만으로 종료하지 않고 원 승인 범위 report/main까지 진행.
기술오류는 선보고·원source/raw/cost 보존 후 범위내 최소수리/재제출 가능.
과학 fallback/성능하락을 오류로 재실행하지 않는다.

## 6. Write ownership / 산출물

SH4 전용 codex/server4-en-adaptive-nullspace-b300-v1 worktree 및 새 execution namespace:
 /data/janghj/ODE-edit/local/en-adaptive-nullspace/20260920-v1/server4-migration-r1/
기존 S3 source/data identity는 provenance로 유지, 실행source/분석source/S4 runtime를 새로 결속.
허용 source project/run_scripts/en_adaptive_nullspace/** 및 scoped tests/config/launcher.
보고 experiment-reports/servers/server4/en-adaptive-nullspace-2026-09-20-v1/**,
audits/servers/server4/2026-09-20-en-adaptive-nullspace/**,
transfers/verifications/2026-09-20-en-adaptive-nullspace-server4/**,
messages/acks/server4/2026-09-20-en-adaptive-nullspace.md,
messages/server-heads/server4/2026-09-20-en-adaptive-nullspace*.md,
tasks/status/server4-en-adaptive-nullspace-20260920-v1/**,
runs/odeedit_en_adaptive_nullspace_s4_20260920/**,
plans/updates/server4/2026-09-20-en-adaptive-nullspace.md.
SH3 마지막handoff는 원 server3 task/status/ack/audit 경로에 게시할 수 있다.

첫 M0는 인계source확보/누락/입력재사용·S4메모리/시간/저장계획.
이어 actual T0/B1 firsttable/B300 terminal·상세보고를 중간 인계한다.
Gitsource+compact raw-free report/receipt의 ownscope nonforce main 통합까지 승인,
전서버 결과복제0/NO_BROADCAST_NOT_REQUIRED. SHfactual, GH별도 해석.
완료 후 STOP, 자동후속task/실험0.
