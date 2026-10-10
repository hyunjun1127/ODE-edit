# SH1 Qwen CF MEMIT_FE_HISTORY mask cold rerun

USER nonce `USER-GH-QWEN-BASELINE-MASK-COLD-RERUN-20261010-R1` ACK 후 정본 전체를 읽고 실행했다.
root dirty와 기존 worktree/frozen source를 보존한 전용 non-main branch에서 진행했다.
원 job61975는 scheduler active cache에서 이미 없어 accounting으로 exact name/user/workdir와
원 submission/script/config/lock을 대조했다. COMPLETED exit0, 2026-10-09 18:33:16–23:33:52 KST.
local COMPLETE는 실제20 native apply/2000 requests이다. 따라서 취소0, 기존 source/raw/CP KEEP.
61929는 과거 실패이며 추가 복제/취소0. 다른 모델·평가·OURS·보호 held job 변경0.

## 수리본 결속

GH source ac19db2f의 공통 EasyEdit generator를 읽기 전용 채택했다.
SHA `35506690c41ecb7d59f11660da41dde50338f5a2ba9613735e045a8c7ab98db4`.
MEMIT_FE_HISTORY→native FE→easyedit.util.generate_fast 실제 import 경로를 확인했다.
기존 FP64 system 메모리 수리/CPU rollback/H final mean-key append/모델별 native FE hparams는 그대로다.
공통 generator 및 SOURCES 중복 수정0. own prepare/submit/runner에 좁은 mask-rerun 분기만 추가했다.
CPU history13 + 공통 실제 tiny generator3 =16 PASS; source166 SHA/Python325/import0 PASS.
실제 pretrained GPU 성능/수치 parity PASS가 아니며 별도 GPU qualification/추가fit0.

새 config는 기존 model/tokenizer/C0/runtime/stream/seed/BS100x20/DEFERRED 계약을 유지하고
이 mask nonce·generator SHA·oldjob61975·새 출력 경로를 명시한다. old editedW/H/context/CP resume0.
native context cache는 cold None에서 첫 실제 apply가 생성한다. 첫B1 후 새
`native-context-identity.json`에 contextSHA/generatorSHA/source/config/requestSHA를 기록한다.
지금 actual contextSHA는 미관측이며 기존 값을 채워 넣지 않았다.

## 실제 제출

새 job **62061**, `official-s1-cf-qwen25-memit-fe-history`.
source **8da621a9e12fd341a4a70fba95531af6fe4fdb80**, official tree
`e475f8c01919b675a80122ecf70a0a1293fdfd75`.
config file SHA `a172c29d3b61b417f115da3cb581bbda606768f72a43871fdb3db6c9a534ef34`.
execution lock SHA `1759f6d5a0b18b35f885194e40f04db8e73f8d38f4c1fe548dfbea0f8c51141b`.
GPU1/CPU8/98304MiB/48h ceiling, devbox/gpu/lab_gpu_s1, exportNONE/Requeue0.
fresh owned allocation0, new DAG width1/cap4. dependencyなし; initial snapshot PENDING.
held owner/source/fullargv/config/script/input/resource/dependency 검산 후 실제 release했다.
기존 standalone history task와 동일하게 own GPU1 job이며 새 collector나 과학 arm 추가0.

attempt `/mnt/raid5/janghj/ODE-edit/local/qwen-baseline-mask-cold-rerun-20261010/server1/registration-r1/`.
새 W0/최신CP/W20 CP 경로는 같은 task `preparation-r1/runs/qwen25/checkpoint/`이다.
CF FLU/CON DEFERRED, future consumer 종료 전 CP KEEP. 현재 새W20/online readback 완료 미관측.
native FE 본표와 history 별도 variant를 유지한다. GH가 README의 old61975→new62061을 통합한다.
소형 receipt는 `audits/servers/server1/qwen-baseline-mask-cold-rerun-20261010/submission.json`.
`NO_BROADCAST_NOT_REQUIRED`: same-host 기존 자산 read-only; raw/CP/모델 대형 전송·삭제0.
등록 후 bounded handoff로 종료하고 장기GPU대기/주기monitor/자동retry를 만들지 않는다.
