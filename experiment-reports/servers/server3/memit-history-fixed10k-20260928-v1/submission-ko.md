# MEMIT history fixed10k 제출 인계

Nonce `ODEEDIT-GH-SH3-MEMIT-HISTORY-FIXED10K-20260928-R1` 및 추가 `ODEEDIT-GH-SH3-MEMIT-HISTORY-USE-BLUE-20260928-R1`.
상태 **MONITORING_PAUSED_AWAITING_USER**. 단일 job **53996**의 held 검사 후 release 명령 성공.
Actual initial/terminal은 **NOT_OBSERVED**. CPU/source 검산과 실제 GPU 실행 성공을 구분한다.
Release 이후 scheduler/log/result 조회, callback, 대기, 자동 recall 0.

## 실제 호출과 immutable source

- 실행 source commit `f2c1d6f17774d6c591d2a0d71e8165a615b26cba`, tree `419619e2645208bf9cc7709e91fa00dfc30b8ca7`.
- source archive SHA256 `a7698edb18c03cdc8ee8fe4b0cfec44710faf5f6040f28217f1af898802bd5d3`.
- 실행 lock `/data/janghj/ODE-edit/local/memit-history-fixed10k/20260928-v1/attempt-v1/execution.lock.json`, SHA256 `e9d2a322dee39ae67f5d0258279d2df3a8bc7576b726de4a098c39f12b68dfa5`.
- pinned BLUE commit `311b076a92e4ed0f14f5c8b4909732da781bc5f7`.
- actual import 예정 경로 `/data/janghj/ODE-edit/local/memit-history-fixed10k/20260928-v1/blue-upstream/memit/memit_seq_main.py`, SHA256 `f84fcf4b388ff1e5c5c9d520202d926e5b16314b8269063b57d8d25243bc716a`.
- `memit.memit_seq_main.apply_memit_seq_to_model` 직접 호출. 반환 model/cache_c object identity를 검산한다.
- 원 native compute_z/FP64 solve/history append 코드 변경0. 경로·관측·transaction adapter만 신규.
- 전체5층 temporary write 뒤 post-key로 history층별1회 append. Prior-H 계수1, C0계수15000, residual분배5/4/3/2/1.
- `blue=false`, zero-based L4–8, cold W0/H0. 별도 history writer/AlphaEdit/P/EN/z-hook0.

## 입력·비교 설정

S3 기존 model revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2와 5개 static C0의 SHA/size 및 C0 dtype/shape 검산.
Fixed10k JSON SHA3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1,
orderedroot5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729.
10000 native target/prompt token order 봉인, 최대 native fitting input32tokens/7sequences.
Seed20260907, torch2.9.1+cu128/transformers4.44.2, FP32/eager/autocast off,
matmulTF32=false/cuDNN TF32=true. 기존 BASE_MEMIT hparams JSON 그대로 SHA1d701acd….
Native context는 BASE_MEMIT B001 contexts SHA33cec0ee… 선택 재사용.
Tokenizer property와 실제 BOS/token IDs는 CPU token manifest 및 runtime.json에 별도 기록한다.
Canonical evaluator/contracts 파일은 과거 execution lock SHA와 동일, explicit-left padding/MB16 유지.

원 BASE_MEMIT archive582a3789… 및 lockf6d1d401…은 정확 수신했다.
그러나 archive/current runtime065dccfe…는 과거 실행 표cef9e07b…와 다르므로
original execution runtime bytes라고 인증하지 않는다. 이번 실행 adapter SHA를 별도 봉인했다.
BASE_MEMIT42658 reference는 RS6453/10000, PS11407/20000, NS49838/100000;
같은 endpoint/분모의 산술 비교만 프로그램에 연결했다. Cross-host/kernel parity 미확립.
과거 TF summary는 게시표에 존재하며 paired raw는 이번 입력에 없음.
따라서 baseline paired는 CPU recall 때 compatible raw 결속 전까지 NOT_AVAILABLE이며 꾸며 채우지 않는다.
기존 targets/weights/state/later-batch z 재사용0, C4 복구0, GSS/EN 재개0.

## 프로그램 관측과 저장

동일 fixed10k 순서 B100×100, 같은 RAM W/H로 계속한다. 매batch currentRPN/all-seen rewrite,
B1/5/10/20/30/40/50/60/70/80/90/100 full all-seenRPN.
RSPSNS strict NLL preference(tie실패), TF token-micro/prompt-macro/full-target strict,
true/new/desiredNLL·margin·분모를 동일 forward에서 기록한다.
CPU reducer는 at-write→final 및 W5 first500→W100 pairedlost/gained/ID,
active/superseded strata와 final raw-free Korean report를 작성한다.
예정 z10000/solve500/history-layer-append500은 runtime 실제 counters로 확인하게 했으며 아직 실측 아님.
`save_checkpoints=false`; editedW/H/delta/optimizer/RNG/resume disk0. RAM rollback만 허용.
`exact_resume=NOT_AVAILABLE`. 원자료 보존, 모델/CP/C4 전송0.

## 자원·제출 검산

1GPU/8CPU/121856MiB, ubuntu/gpu, exportNONE/Requeue0, wall168h. Cap1 prospective 설정.
제출 전 resource-only 확인에서 project active GPU0, admission 기존 projectjob [].
Dependency `None`. Held owner/fullargv/source/GPU/CPU/mem/node/dependency 검산 전부 PASS.
계획 hostpeak80GiB/GPUpeak72GiB, 12–48h 추정/7d wall은 실측 아님.
H 및 rollback 각3.828125GiB, C0 3.828125GiB, selected W0+entry2.1875GiB,
FP64 solve의 일시행렬을 포함해 계획했다. 실제 메모리/시간은 미관측.
디스크 요구22GiB 대비 제출 준비 관측 여유 약1.275TB. 별도 storage waiver0.
CPU10tests PASS, 단일 launcher memory policy PASS, frozen194members SHA 검산 PASS.
Owner검산이며 독립 red agent 미사용. No scientific GPU parity gate 추가0.

## 실행 경로

```sh
# 이미 등록·release된 동일 job을 재제출하는 명령이 아니다.
# provenance: source=/data/janghj/ODE-edit/local/memit-history-fixed10k/20260928-v1/frozen-source-v1/source
# output=/data/janghj/ODE-edit/local/memit-history-fixed10k/20260928-v1/attempt-v1/output
# submission/release receipts=/data/janghj/ODE-edit/local/memit-history-fixed10k/20260928-v1/attempt-v1
```

실행 재현 argv는 `runs/odeedit_memit_history_10k_s3_20260928/submission.json`에 보존했다.
이미 등록된 job을 자동 중복 제출하지 않는다. 완료 CPU 리뷰는 사용자 recall 시 수행한다.
NO_BROADCAST_NOT_REQUIRED. 상세 assets/CPU/resource/M0는 같은 task audit/report 경로에 있다.
