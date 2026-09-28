# GH → SH3: MEMIT + history baseline, fixed10k B100×100

Instruction / ACK nonce: `ODEEDIT-GH-SH3-MEMIT-HISTORY-FIXED10K-20260928-R1`.
Target: server3 / ubuntu / `01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3`.
Expected CWD `/data/janghj/ODE-edit`; repository `hyunjun1127/ODE-edit`.
GH session `01a04939-8873-7673-8dca-4c7fc5e31af0`.

## 1. 사용자 권한 및 이번 실행 범위

사용자: “2026-09-28-memit-history-implementation-review-ko.md를 보고 자세히
파악한 다음 memit에 history를 추가한 baseline run. 기존 10000 samples,
batch size100, SH3 담당, server3 GPU cap1.”

본 envelope로 구현·최소 CPU 검산·source freeze·Slurm 제출·own-scope
branch/main 게시를 승인한다. 계획만 작성하거나 GH 재승인을 기다리지 않는다.
새 과학 arm은 **MEMIT_HISTORY_NATIVE (MEMIT_seq, blue=false)** 하나다.
Fresh pretrained W0 / layer별 H0=0에서 동일 fixed-order B100×100 한 chain.
추가 BLUE, AlphaEdit, history-off, single-layer, coefficient sweep는 제출하지 않는다.
검토문 마지막 4-arm 제안은 후속 연구 제안이며 이번 1-arm 승인을 확대하지 않는다.
기존 BASE_MEMIT42658은 history 없는 정상 경로였고, 이번은 그 결과의 수정/교체가
아닌 history baseline 추가다. 기존 결과·source·실패·다른 paused task를 보존한다.

## 2. 필독과 source identity

- `audits/global/2026-09-28-memit-history-implementation-review-ko.md`
  SHA256 `6b8a692ed344dc8f3679b8fe0886b018f7cb3eaf386c1bcc532a5c9b0f684508`.
- `plans/global/fixed-counterfact-10k-policy.md`, `PROTOCOL.md`, no-checkpoint 정책.
- `agents/server3/experiment-ready-paths-20260919-v1.json`은 기존 자산 위치의
  출발점이지 현재 존재/identity 또는 MEMIT 실행 준비 완료 증명이 아니다.
- `experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1/diagnostic-report-ko.md`
  및 그 compatibility/config 표에서 **BASE_MEMIT42658**만 직접 비교 기준으로 결속.
- 이번 `audits/global/2026-09-28-memit-history-sh3-dispatch/`의 local-audit
  source/config evidence는 GH local audit의 읽기 전용 참조 사본이다.
  runtime 사본 SHA와 과거 실행 CSV SHA는 달라 original execution bytes로
  인증하지 않는다. preflight-ko.md를 읽고 실제 frozen source를 별도로 결속한다.
- BLUE upstream `311b076a92e4ed0f14f5c8b4909732da781bc5f7`의
  `memit/memit_seq_main.py`, `memit_main.py`, `compute_ks.py`, `compute_z.py`,
  hparams 및 실제 호출 import closure를 확인한다. 필요한 pinned public source의
  task-local fetch는 허용한다. 최신 upstream으로 임의 교체하지 않는다.

AlphaEdit/SUIT의 wrapper는 cache_c 전달 누락 경로가 있으므로 그대로 사용하지
않는다. 새 namespace에서 실제 실행될 source/config/import SHA를 별도로 봉인.
기존 EasyEdit dirty·공유 환경·원 native source는 수정하지 않는다.

## 3. 방법의 고정 의미

물리 layer는 zero-based L4/L5/L6/L7/L8의 down_proj 5개, BLUE=false.
Layer i의 writer는 `solve(15000*C0_i + H_i + K_i K_i^T, K_i)`와
native residual 분배 `5,4,3,2,1`을 사용한다. H 계수는1, 감쇠·정규화·rank
truncation·projector P·AlphaEdit L2·EN correction을 추가하지 않는다.
H0는 5×14336×14336 FP32 zero history이며 C0 static covariance와 별도 관리한다.
Native FP64 solve/FP32 weight materialization과 고정 covariance 로더를 유지한다.

Native semantics를 정확히 유지할 것:

1. Batch entry의 현재 W에서 최종 편집 layer L8 native z를 요청당 한 번 계산.
2. 각 layer의 current key/residual을 해당 native 순서로 계산하며 solve에는
   **이번 batch 전 H**만 사용한다. 현재 K Gram은 solve에서 한 번만 더한다.
3. 모든 5층의 temporary native write가 적용된 endpoint에서 각 layer key를
   다시 계산하여 `H_i += K_post_i K_post_i^T`를 batch당 층별 정확히 한 번 수행.
   pre-write key를 대신 저장하거나 H를 solve 전에 append하지 않는다.
4. native 내부 temporary restore와 wrapper의 returned delta materialization은
   원래 동작이다. Wrapper가 반환하는 model과 cache_c를 받아 동일 live history를
   다음 batch로 이어간다. Wrapper/finalizer에서 두 번째 append하지 않는다.
5. B2 이후 own W/H/context/RNG를 연속 사용한다. batch별 W0 reset, baseline의
   later-batch z/updates 재사용, case_id-only stale z-cache 사용을 금지한다.

기존 BASE_MEMIT과 맞출 설정: Llama3-8B-Instruct revision
`8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, seed20260907,
torch2.9.1+cu128/transformers4.44.2, FP32/eager/autocast off,
matmul TF32=false / cuDNN TF32=true, native v_steps25/v_lr.1/
v_weight_decay.5/clamp.75/KL.0625/loss_layer31/subject_last,
mom2 wikipedia100000 float32 / update_weight15000. 나머지 원 config도 결속한다.
Server3 readiness의 cuDNN=false를 조용히 상속하지 말고 비교 기준을 명시적으로
설정한다. Tokenizer 속성 대입과 실제 BOS/token IDs를 혼동하지 않는다.
Native context 및 canonical evaluator tokenization/padding/MB16은 과거와 맞춘다.
이번에는 새로운 z batching/hook/BF16/clipping 최적화를 끼워 넣지 않는다.
원 native compute_z를 사용하여 history 변경 이외의 과학적 차이를 최소화한다.

## 4. 동일 데이터 / 관측 / 비교

Dataset `counterfact-fixed-10k-v1`, server3 canonical local dataset root를 검증.
JSON SHA `3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1`,
orderedroot `5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729`,
sample.lock SHA `a8d22c230611b6c14740d1d00658a53856bbde26d060189d5ddf30f2ffdbac92`.
B1=[0:100], … B100=[9900:10000], shuffle/재추출/성공기준선택0.

원 native baseline observer schedule 유지: 매 batch Current R/P/N 및 all-seen
rewrite, B1/5/10/20/30/40/50/60/70/80/90/100에서 full all-seen R/P/N.
이는 평가 시점이며 checkpoint 저장 시점이 아니다. W100 분모10000/20000/100000.
RS/PS는 new NLL<true NLL, NS는 true NLL<new NLL, ties failure.
같은 forward에서 teacher-forced rewrite/rephrase/neighborhood accuracy도 기록:
token-micro, prompt-macro, exact-sequence strict, true/new NLL 및 token 분모.
Preference와 TF accuracy를 같은 지표로 부르지 않는다. Current/at-write/
all-seen/active-vs-superseded를 구분하고 W5 first500→W100 유지 변화도 보존한다.
공식 P/N은 observer-only이며 fitting/history 선택·하이퍼 튜닝에 사용하지 않는다.

완료 비교는 기존 BASE_MEMIT 및 필요시 BASE_ALPHAEDIT raw-free 표/호환 raw를
재사용한다. 같은 endpoint·분모끼리 paired lost/gained를 계산하고 host/GPU/kernel
차이 및 history 처리비용을 명시한다. 기존 history-off 새10k chain은 승인하지 않는다.
과거에 TF 자료가 없으면 NOT_RECORDED로 두며 새 GPU 재평가를 자동 추가하지 않는다.
R512/C4 teacher는 이 baseline에 필요 없고, SH3가 삭제한 101.5GB reference를
복구하거나 EN/GSS 작업을 재개하지 않는다.

## 5. 구현 검산 / 오류와 수치 경고

CPU 회귀: cache 전달/반환, H0=0의 writer 동등식, nonzero H가 solve에 실제 영향,
post-final-key 누적시점, 5층당1회 append, B1→B2 W/H linkage, 중복commit 방지,
rollback, observer 비변이, noCP, full100batch routing과 source/token/order 확인.
첫 본 batch 안에서 실제 history 사용·유한성·B1 commit/B2 entry receipt를 저장한다.
기대10000 z/500 solve/500 layer history append는 계획이며 실측 counters로 검산한다.

별도 긴 GPU parity/FD/repeat 캠페인이나 성능 gate를 선행조건으로 만들지 않는다.
근접 NLL/MB/repeat/교차host 오차는 실제 수치·기준·flip·대상 행을 기록하는
비차단 diagnostic으로 구분한다. 재현이 확립되지 않았는데 PASS로 바꾸지 않는다.
정확한 input/source/order/layer identity, history 전달·누적·state continuity,
finite, 실제 weight 적용/복원, IO/분모/권한/자원 오류는 무시하지 않는다.
과학적 낮은 성능/보존 손실을 기술 실패로 삼지 않는다. 확인된 구현 오류는
선보고 후 이 task 범위 내 최소수리·새 immutable attempt/재제출 가능;
원 source/raw/partial/cost 보존, 수식·history 의미·hparams 변경은 별도 결정이다.

## 6. 자원 / 저장 / 제출 후 경계

Server3 project cap **1**, task cap1. 이전 cap2를 prospective cap1로 대체한다.
단일1GPU/8CPU/host≤121856MiB, ubuntu/gpu, exportNONE/Requeue0.
제출 직전 resource-only active/admitted 조회로 충돌 확인, 기존 타job 취소/변경0.
슬롯이 없으면 cap-safe dependency/pending으로 등록하거나 정확 resource block을
보고한다. 무자원 PENDING을 actual실행 또는 initial PASS로 보고하지 않는다.
실측되지 않은 시간/host/GPU/disk 추정과 wall request를 별도 봉인하며 cap은
GPU시간 budget이 아니다. 원 C0/H 5층의 CPU RAM, FP64 temporary solve peak 포함.

`save_checkpoints=false`: edited W/H, final/periodic/best state, optimizer/RNG
resume bundle 및 동등 복원 delta 모두 disk 미저장. H는 프로세스 RAM에서 유지.
작은 hash/norm/counters/commit metadata와 raw NLL/TF/시간/로그는 보존한다.
`exact_resume=NOT_AVAILABLE`; 기존 자료 삭제0, 별도 full-model copy 저장0.

source/config/input/resource lock 뒤 held owner/fullargv/source/GPU/CPU/mem/
dependency 검사→release. 100batch와 최종 compact reducer/manifest를 프로그램에
연결하며 B1 품질/GH 추가승인으로 중간중단하지 않는다. 앞선 사용자 server3
post-release no-monitor 경계를 유지: release 뒤 scheduler/log/result/초기/terminal
polling, sleep-wait/heartbeat/callback/자동 recall0. 제출/source/lock/job mapping
게시 후 `MONITORING_PAUSED_AWAITING_USER`. 등록 프로그램은 자연 진행한다.
완료 상세 CPU 리뷰는 사용자 recall 때 수행; 제출을 완료 결과로 표시하지 않는다.

## 7. 쓰기 / 전송 / 게시 envelope

Dedicated branch `codex/server3-memit-history-fixed10k-20260928-v1` 권장.
구현 `project/run_scripts/memit_history_lifelong/` 아래만 신규/수리.
읽기 전용 원 source를 이 namespace 또는 task-local vendor로 명시적으로 가져와
adapter로 연결할 수 있다. 공유 source/helper/env/native 전체 수정0.
Local `/data/janghj/ODE-edit/local/memit-history-fixed10k/20260928-v1/`.
다음 own scope 쓰기 및 검산 후 nonforce branch/main 통합을 사전 승인한다:

- `experiment-reports/servers/server3/memit-history-fixed10k-20260928-v1/`
- `audits/servers/server3/memit-history-fixed10k-20260928-v1/`
- `plans/updates/server3/memit-history-fixed10k-20260928-v1/`
- `tasks/status/memit-history-fixed10k-20260928-v1/server3.json`
- `messages/acks/server3/2026-09-28-memit-history-fixed10k.md`
- `messages/server-heads/server3/2026-09-28-memit-history-fixed10k.md`
- `runs/odeedit_memit_history_10k_s3_20260928/` (명시적 namespace 허용)
- `transfers/verifications/2026-09-28-memit-history-sh3/`
- ignored local cap/boundary config의 실제 server3 task 값.

필요한 과거 완료 source/config/context/evaluator만 exact inventory/SHA를 먼저
만들어 repo-local approved local source에서 task-local inputs로 선택 pull 가능.
SH3 sole destination writer, SOURCE_KEEP/no overwrite/no delete. 원격 live task 조회,
대형CP/model 재전송·통계 재계산·C4 teacher 전송은 이번에 필요없으며 승인0.
원본24CP 등은 가져오지 않는다. input/stat 모델은 기존 SH3 verified assets 우선.
NO_BROADCAST_NOT_REQUIRED: raw/GPU/teacher/CP를 타서버로 자동 전파하지 않는다.
Git에는 source·small config·표/보고/receipt만, raw/prompt/tensor/fullstdout0.
Preflight/source/data/resources/metadata 검사와 향후 postrun reducer를 구현하고
실제 감사 수준(owner/분리 reducer/독립 red 유무)을 솔직히 기록한다.

## 8. 첫 ACK/M0 필수

위 nonce로 실제 수신 ACK, review SHA/FULL_READ, history 키 수집시점/계수,
기존 baseline과의 차이표, asset/source 확보, 남은 구현, cap1 및 자원·저장 계획,
기존 결과 재사용 범위를 보고한 뒤 작업을 진행한다. M0/CPU PASS/제출/actual
initial/완료를 구분한다. GH ACK나 추가 raw/GPU 감사를 선행조건으로 기다리지 않는다.
