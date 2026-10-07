# GPT2-XL native baseline generation rerun 사실보고

현재 상태: **USER recall 후 실제 여섯 baseline + CPU collector 등록·held 검산·release 완료**. Bounded snapshot에서 MEMIT60928 RUNNING, 나머지 정상 Dependency PENDING이다. 실험 완료·B1 commit/다음 entry·metric 전체 전송은 아직 미관측이며 기다리지 않았다.

## 등록 재개 실제 근거

Recall nonce `USER-GH-SH1-SH2-SH4-BASELINE-GENERATION-REGISTER-RESUME-20261007-R1`, authority `6d6e2fdb531ecb3d6e5bc98db0f188001c6a7aa0`, envelope SHA `43099367d76e1f8a2b57b539c4a18aa17351e2194885db70e3516af19464fc16`. Direct owner ACK 후 현재 queue/accounting/로컬 제출 receipt를 exact task로 대조했고 이전 등록 없음이 확인됐다. 이번 recall에서 등록 pass1회만 수행했다. 새 attempt `local/gpt2xl-baselines-fluency-consistency-2k/20261007-v1/attempt-register-r1`; 원 attempt-v1/source/raw/error는 그대로 보존했다.

과학 source `6bc51602632b5a2dfb4c832479002b30b604b8eb`와 archive SHA `e502cb6a9e2e259aa074944dc834047a49a730750b7f404e90553ad5190f3fa9`, source332개는 bytes/SHA 동일하다. 등록-control source `4f068a938a5cce16ac999bb6daa4b583fb5a71ab`를 별도 결속했다. Config 변경은 attempt/run_instance/registration_recall뿐이다. 새 lock SHA `ec7fa67067edaff39586dab674f72bf393ccb77e9dce54e9f11ea86c81fbe62f`, config SHA `9738bb2a05f28cff347f260789d32db667d6bcb542142a5e9c4528f8e13cb4f2`. Shared reference READY/manifest/API도 원 SHA 그대로이다.

| Baseline | 실제 job | afterany dependency | bounded 초기 상태 |
|---|---|---|---|
| MEMIT | 60928 | 없음 | RUNNING/devbox |
| AlphaEdit | 60929 | 60928 | PENDING/Dependency |
| CAKE | 60930 | 60928 | PENDING/Dependency |
| AlphaEdit-BLUE | 60931 | 60929 | PENDING/Dependency |
| PRUNE | 60932 | 60930 | PENDING/Dependency |
| RECT | 60933 | 60931 | PENDING/Dependency |
| GPU0 CPU collector | 60934 | 신규60928–60933 모두 | PENDING/Dependency |

등록 직전 own GPU allocation/admitted queue0, 실제 신규 DAG width2/cap2 및 prerelease width2를 검산했다. Fresh root available1214976000B/RAID1438857592832B/inodes335305128. Node devbox/partition gpu/QoS lab_gpu_s1, 각GPU1/CPU8/65536MiB/48h request, collectorGPU0/CPU8/24576MiB/4h, exportNONE/Requeue0. 전체7 owner/fullargv/script bytes/source/input/resource/dependency held 검사 PASS 후 역순 release 모두 성공. 이전 취소 ID는 새 dependency에 없다. OURS/W0/타task 신규 취소·변경0.

기존 CPU154개 증거를 재사용했고, 등록-only CPU10개 PASS 및 실제 frozen production `run.locked` PASS(모델 load/native fit0). 독립 reviewer는 adapter/source332/archive/READY/configdiff/one-pass/cap/privacy 검토와 추가 metadata CPU5개 PASS를 수행했으며 scheduler/network/model 조회는 하지 않았다. Scientific source/GPU qualification을 새로 실행하거나 기존 method 계수를 바꾸지 않았다.

MEMIT [W&B run](https://wandb.ai/wkdguswns2256/layer%20allocation/runs/83b85657a45c4673)의 startup remote API에서 run ID/name=`server1-BASE_MEMIT-attempt-register-r1-job60928`, Config.job_id60928/step_id=-5 및 source/config/model/schema identity 일치를 검증한 own receipt가 생성됐다. 상태 READY_ONLINE은 여기서는 원 worker의 bounded remote identity/config 검산을 포함한다. Metric 전체/finish readback과 scientific completion은 **미관측**이며 다른 arm W&B는 아직 미시작이다. No recurring monitor/heartbeat/auto retry, sealed runner/collector는 원20batch를 자연 진행한다. Compact 상세 근거는 `audits/servers/server1/gpt2xl-baselines-fluency-consistency-2k/registration-resume-r1.json`.

## 이전 parent 수행 기록 — 역사 보존

아래의 미등록/저장공간 차단은 **이전 attempt-v1 당시 상태**이며 현재 등록 상태가 아니다. 원 실패/원 source/비용/receipt를 보존한다.

## 권한과 경계

Nonce `USER-GH-ALL-SH-BASELINE-FLUENCY-CONSISTENCY-RERUN-20261007-R1`, authority `b0cee1a302c10d48e206a31c7c5cf318ac36d8c3`. Actual app CWD `/mnt/raid5/janghj/ODE-edit`, session `01a04939-f93a-7b50-bca0-65438eab2062`, host devbox/server1. 과거 registry29e4는 역사값이다. 전용 non-main WT/branch에서 작업했고 dirty root, 다른 agent/source, OURS/PRICE/W0/setup/model/C0/P 및 기존 raw를 보존했다.

Baseline-only 취소는 source/owner/argv/script SHA와 실제 상태를 대조하여 collector→pending successor→active parent 순서로 수행했다. `60759,60741,60758,60757,60740` 취소 성공. 최종 정확9-ID active queue는 비어 있었다. `60150,60151,60152,60739`는 이미 COMPLETED여서 보존했다. 기존 60740의 과거 GPU 비용4703초, terminal GPU 비용은 MEMIT5663초/Alpha6483초/CAKE7173초이며 신규 실험 비용과 섞지 않는다. Pending 취소는 GPU 할당0. 이전 결과/부분결과/출력 삭제0, rescue CP0.

## 구현·공통 결속

공통 source `83535c6a47c552cc4e5c6385f3a587d752820150`, 최초 main 게시 `2828ab0938184ae863831c6e5e93fcf7a021eca4`. API는 `project/run_scripts/experiment_generation_eval/README.md`. `GenerationObserver.observe`는 exact state/runtime/occurrence별 실제 생성1회, `subset/read_observed`는 identity-checked CPU 재사용이다. 호출자는 ordered occurrence와 actual physical W identity 및 H/context guard를 결속한다. 원 native fit/writer/context/precision은 기존 sealed source를 읽기 재사용했다. 추가 fit/pilot/품질 gate/B21 없음.

Declared profile `cf-cake-prompt-inclusive-total100-eos-corrected-v1`: unpadded single-row full-prefix no-cache, topk5/temp1/topp1/총prompt+continuation100/native EOS. 원 CAKE generator의 byte/published-performance parity 또는 빠른 route라고 주장하지 않는다. Prompt truncate/overwrite0, 이미100이상이면 그대로 보존·continuation0·typed reason. Python/NumPy/CPU/이미 초기화된 CUDA RNG, W/H/context/hooks/parameter-buffer/training mode 비변이와 예외 restore를 검산했다.

Fluency는 native NLTK H2/3+2H3/3 bits, per-text→per-occurrence→valid macro. Consistency는 fixed vocab/IDF cosine, relation/target 전체 reference이다. 재fit/subject-only essence 필터 없음. Missing/zero vector/nonfinite는 typed missing, 유효한 measured0과 구분한다. Raw sums/counts/overlap costs는 local 보존; 공개는 scalar mean/count/reason만이다. R/P/N/harmonic/분모는 불변. W&B strict key/config, job/name/immutable identity, edits/pre/post state와 fit 축을 결속했다. SDK 접수와 remote delivery는 별도이며 이번 새 run의 실제 online 검증은 아직 없다.

W0 generation은 첫 실제 BASE_MEMIT cold run에서 fresh first2000을1회 측정한 후 atomic READY를 공유할 계획이다. 과거 RPN-only W0를 generation evidence로 대체하지 않는다. 모든 baseline current/pre·post100, milestone W5/10/15/20 seen prefix를 평가하며 overlap은 같은 raw CPU subset으로 재사용한다. 새 생성 case 상한 계획은 arm별8500 + 공유W0 2000 = 여섯 arm53000이며 실제 비용/속도/ETA 측정값은 아니다. 각 raw/phase에 forward/token/time 비용을 별도 보존하고 실패 전 반환된 관측도 double-count 없이 집계한다.

## Reference READY와 CPU 증거

Reference actual manifest `/mnt/raid5/janghj/ODE-edit/local/baseline-generation-eval-assets/20261007/reference-ready-r1/manifest.json`,6769B SHA `6d9a713ab7eaa10871f277e10a3974e0bd5b140c265be80052c279f258876ca8`. READY999B SHA `634ca5c70f54f7af5346849b3594ed92280cfbc3853d0def16a7eef86ae64203`, identity `75e595c7f26ec334830e9bb9ca6028098c19ea84a9509a5713985847683f8ea6`. 3개 공식 공개 reference를 단회 다운로드·SHA/schema 검산했으며 total968966959B. 정확 경로/size/SHA는 audit reference manifest에 있다. first2k prompt/reference coverage 각각2000/2000, missing0은 coverage 증거이지 과학 점수 PASS가 아니다. Vocabulary/finite IDF1380255, TF-IDF fitting0. NumPy2.2.6/SciPy1.15.3/sklearn1.7.2/NLTK3.6.5와 기존 active English punkt bytes/source를 결속했다. Peer는 exact files를 단회 pull하고 path override 가능; 다른 science env pin 변경0. Raw/model/stat/P/credential 대형전송0, NO_BROADCAST_NOT_REQUIRED 및 exact reference-pull 예외만 기록.

최종 owner CPU production fixtures154개 중152 PASS,2 SDK 의존 skip, 실패0. 테스트 source SHA receipt 보존. Independent reviewer1은 다른 작성자의 generation/tracking/source를 읽고19 unit·7 metadata observer·2 DAG checks PASS; 자신의 assets 구현은 제외했다. Reviewer2는 다른 작성자의 assets/native runner/collector32 CPU checks PASS; 자신의 공통 generation 수리는 다른 reviewer가 재확인했다. 실제 GPU/native fit/online 검증과 분리한다.

발견·수리한 기술 경계: 혼합-state endpoint/selected-row stale hash 재사용, startup manifest pointer/member/identity, native cosine1.0000000000000002의 FP64 endpoint rounding, `/tmp` ENOSPC와 CPU 준비의 잘못된 runtime key. 원 score/formula는 보존했고 cosine은 clamp하지 않았다. 고정4-ULP transport endpoint 허용만 명시하며1.000001은 거절한다. 원 품질/numerical threshold를 성능에 맞춰 바꾼 것이 아니다. RAID task-local TMPDIR를 제한 SDK sidecar에도 연결했다. 이전 CPU 준비/검산 기록은 남기며 scientific repeat-to-PASS0.

## 실제 등록 결과·차단 원인

실행 source `6bc51602632b5a2dfb4c832479002b30b604b8eb`.
Immutable attempt `/mnt/raid5/janghj/ODE-edit/local/gpt2xl-baselines-fluency-consistency-2k/20261007-v1/attempt-v1`.
Lock SHA `dd1690924118251ab302aeb885ffdd396976b6b863a731b7f54b120e5b135f35`, config SHA `713e164ce793ea7231d3b08b8bfab1a55adc08d74a8de62c290e6bd999f583aa`, source archive SHA `e502cb6a9e2e259aa074944dc834047a49a730750b7f404e90553ad5190f3fa9`. Archive CPU runtime binding332 source/137 native/3 original config members 검산 PASS, model forward/native apply0.

| Arm | 신규 job/dep | 실제 상태 |
|---|---|---|
| BASE_MEMIT | 없음 / 첫 head 계획 | 첫 sbatch I/O 실패 |
| BASE_ALPHAEDIT | 없음 / 새 MEMIT afterany 계획 | 미등록 |
| CAKE | 없음 / 새 MEMIT afterany 계획 | 미등록 |
| ALPHAEDIT_BLUE | 없음 / 새 Alpha afterany 계획 | 미등록 |
| PRUNE | 없음 / 새 CAKE afterany 계획 | 미등록 |
| RECT | 없음 / 새 AlphaBLUE afterany 계획 | 미등록 |
| CPU collector | 없음 / 새6 afterany 계획 | 미등록 |

Fresh own server1 GPU allocation/admitted queue0, prospective DAG width2/cap2. Other users' physical GPU2는 own project allocation으로 오인하지 않는다. 각GPU1/CPU8/65536MiB/48h request, CPUcollectorGPU0/CPU8/24576MiB/4h. Memory ceiling183296MiB, actual node/partition/QoS/disk/inodes 확인. 더 엄격 cap1이면 전체 직렬화하며 oldcancelled IDs를 resource gate로 사용하지 않는다. OURS/W0/타task source/job 변경0.

실제 제출1회에서 `sbatch: Batch job submission failed: I/O error writing script/environment to file`. 신규 ID 반환 없음, submitted receipt0, exact new task squeue empty. Slurmctld는 UP이나 StateSaveLocation `/var/spool/slurmctld`, SlurmdSpoolDir `/var/spool/slurmd`가 있는 `/dev/nvme0n1p2`의 일반 사용자 available bytes0/100%이다. Inodes는 소진되지 않았고 RAID에는 약1.2TB가 있다. `sacct`도 No space left on device로 실패하므로 내부 ID 소비 여부까지 인증하지 않는다. 이는 수치/방법/모델/C0/P failure나 GPU cap 부족이 아니다.

허가 없는 root/spool 삭제·이동·권한·서비스/config 변경은 하지 않았다. Sealed source/config/failure receipt 보존, 자동 재제출/monitor/heartbeat 없음. 관리자 권한으로 scheduler root 저장공간을 복구한 뒤 exact duplicate/immutable-attempt 검산을 거친 수동 owner recall이 필요하다. 기존 source/raw, 승인된새 C0/P와 W&B spool 보존. 새 실험을 완료 또는 Slurm PENDING이라고 쓰지 않는다.
## 2026-10-08 KV/cache repair — 실제 등록/release 완료

새 scientific source `199cfe5664355f6f1c9069c72396ec759bb25cec`, 공통 package tree `a10d88b555a96962567846dcb34a456c9e0d7c1c`. Own branch `codex/server1-gpt2xl-generation-cache-repair-20261008` 게시 확인. Config SHA `862daa74bd57a6afac756c52f31f963d60975d2123bee86fc8ce72dab50c9700`, lock SHA `c90853982ab07d685fa735b505128a370ee239b22e8614b49a65f2b5936b3589`. 전용 immutable attempt는 `/mnt/raid5/janghj/ODE-edit/local/gpt2xl-baselines-fluency-consistency-2k/20261007-v1/attempt-cache-repair-r1`이다.

| 역할 | 실제 신규 ID | afterany dependency | bounded 현재 상태 |
|---|---|---|---|
| MEMIT | 61148 | 없음 | PENDING / Resources |
| AlphaEdit | 61149 | 61148 | PENDING / Dependency |
| CAKE | 61150 | 61148 | PENDING / Dependency |
| AlphaEdit-BLUE | 61151 | 61149 | PENDING / Dependency |
| PRUNE | 61152 | 61150 | PENDING / Dependency |
| RECT | 61153 | 61151 | PENDING / Dependency |
| GPU0 CPU collector | 61154 | 새 GPU 여섯 ID 전체 | PENDING / Dependency |

최종 owner CPU177개 중175 PASS/2 SDK 의존 skip/실패0. 독립 blocker 수리 재검토12개 PASS. 실제 새 allocation/GPU qualification/W0 READY/W&B startup·finish·W20은 미관측이다. 사전 PLAN digest `83cee93d1b462f8e0b503f4692dd537430ee9084971aaa3a9150bcf4a2a4590b`는 actual receipt가 아니다. 새 source만 봉인했고 실행 중 archive hotpatch0.

전체7 held owner/원 argv/script bytes/source/config/native input/reference/자원/dependency 검사 뒤 역순 release했고, combined width2/effective cap2 및 memory policy PASS를 검산했다. GPU job1GPU/8CPU/65536MiB/48h, collectorGPU0/8CPU/24576MiB/4h, ExportNONE/Requeue0. Source archive/launchers/runtime/native closure/기존 config/reference SHA를 release 직전에 재검산했다. 취소된 old ID는 새 dependency에 없다. 물리 자원 부족은 정상 PENDING이며 기존 타job 강탈/취소/정책 cap 증액0. 단발 postrelease resource snapshot 뒤 agent monitoring을 종료하고 봉인 runner/collector가 자연 진행한다.

Nonce `USER-GH-SH1-SH2-BASELINE-GENERATION-KV-BATCH-REPAIR-20261008-R1`, authority `66cddb8fa9c09e475f0ae3f423635091c6a71e3d`. 새 task는 `gpt2xl-baselines-generation-cache-repair`이며 원 scientific parent/history를 보존한다. 실제 app CWD는 `/mnt/raid5/janghj/ODE-edit`, 전용 WT는 `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-gpt2xl-generation-cache-repair-20261008`이다.

정확 owner/source/argv/script/state 검산 뒤 `60934→60933→60932→60931→60930→60929→60928` 순서로 취소했고, 단발 postcheck에서 일곱 job 모두 CANCELLED 및 exact active queue empty를 확인했다. 원 MEMIT head의 parent allocation 비용25533 GPU-sec, 완결 atomic case 파일1001개, durable edit commit0, 전체 generation W0 READY 미완료를 보존했다. 이 파일 수는 재사용 검증 PASS 건수가 아니다. 초기 취소 receipt의 create-once 충돌은 원 실패와 이미 수행한 취소를 보존한 채 per-role receipt와 명시적 control continuation으로 수리했다. OURS/W0-only/FE/타서버 job 변경0, checkpoint/save/resume0.

공통 API는 `experiment_generation_eval/README.md`의 cache-repair 절이다. Reference→cached singleton→equal-length KV batch 순서의 최대8 고정 prompt qualification PLAN/tolerances를 CPU에서 먼저 결속하고, 실제 native GPU qualification은 첫 replacement job 내부에서 수행한다. 계획 MB8은 실제 full-width8 PASS를 뜻하지 않는다. 현재 선택군의 실제 최대 equal-length 폭6, native100 경계 부재를 명시한다. 검증하지 못한 폭은 사용하지 않으며 승인된 검증 singleton→reference fallback을 따른다. 속도/ETA/실제 token parity 및 GPU PASS는 아직 미관측이다.

이전 완료 W0 행은 원 source/runtime/route/path/SHA/size와 cold-state source-backed guard를 보존하는 명시적 compatibility reader만 재사용한다. 원 endpoint 전체 RAM/RNG guard는 완료되지 않아 미기록임을 밝히며, 새 실제 qualification 및 최종 endpoint 비변이 검산을 별도로 요구한다. 완료 원 행의 각 identity/metric/completeness를 검증하고 불명확·미완료 행만 새로 관측한다. 이는 원 RAM 편집 trajectory resume가 아니다. 새 mixed-provenance endpoint의 모든2000 case 검산 후에만 actual qualification/compatibility SHA를 결속한 atomic READY와 W0 전체 지표를 발행한다.

Generation 진행률은 완료 경계 callback의 `generation_progress/step` 축이며 edits/fit 축과 분리한다. 원문/token/IDs/tensor/API key/전체 env 업로드0. SDK 접수와 bounded remote readback은 구별하며, logging 실패가 과학 재fit 승인이나 offline PASS가 되지 않는다. Native 여섯 방법의 수식/hparams/dtype/solver/BS100×20와 RPN 의미는 변경하지 않는다.

독립 reviewer가 source/installed cache API와 CPU fixture68개를 검토해 old generation identity가 RPN identity에 잘못 연결된 제출 blocker를 찾았다. 정확 원 `BASE_MEMIT/generation/observer-identity.json` 결속으로 수리하고 실제 metadata regression을 추가했다. 수리 후 독립 CPU12개 PASS, 최종 owner CPU177개 검산을 수행했다. 실제 GPU/온라인 검증을 source/CPU PASS로 확대하지 않는다. 장기 완료 polling/heartbeat/자동 retry는 만들지 않는다.

아래 내용은 원 parent attempt의 역사 기록이며 현재 repair의 성공/완료를 뜻하지 않는다.
## 2026-10-08 직접 사용자 recall — R2 수리/재등록 준비

사용자 원문 `fail 된거 다시 처리해`에 따라 기존 실패 attempt를 보존하고 새 create-once `attempt-cache-repair-r2`를 준비한다. R1 `61148..61153` 여섯 GPU job은 모두 FAILED, GPU0 collector61154는 종료 COMPLETED(과학 성공이 아닌 실패 요약)이며 exact active queue는 비어 있다. 추가 취소0, OURS/W0-only/FE/다른 프로젝트/타서버 작업 변경0. R1 GPU parent allocation 비용은 합계143 GPU-sec(82+12+12+12+13+12); 이전 취소 attempt의25533 GPU-sec와 별도로 보존한다. 모든 R1 native fit/solve/history/commit 건수0.

첫 원인: W0Compatibility가 caller의 `{path,bytes,sha256,inode,mtime_ns}` member를 3필드 content member와 전체 dict 비교하여, 원 file의 SHA/size/path/inode/mtime가 모두 맞아도 `GENERATION_MEMBER_BYTES_IDENTITY`로 거절했다. 전체 member 동등성 대신 필수 core3필드 정확 검산과 선택적 inode/mtime 정확 검산으로 수리했다. Unknown field/잘못된 타입/bool metadata/상대경로/hash·size 변조/stat 경합은 계속 차단한다. 실제 R1 observer/config/runtime/coldguard/reference member를 production consumer로 검산했고 CPU9개 PASS, 독립 source/CPU 재검토 PASS이다. 원 raw/member/asset 덮어쓰기0, identity 검산 해제0.

R1 실제 qualification은 최대8 고정 prompt에서 reference PASS→cached singleton의 logits/probability/token/metric 비교 통과, strict topk/mask/position 비교 FAIL이었다. 따라서 원 고정 fallback에 따라 no-cache reference MB1을 선택했다. Batch는 미실행이며 폭8 미검증이다. 이것이 W0 terminal failure 원인은 아니며 이를 cache/batch PASS로 바꾸거나 tolerance를 완화하지 않는다. NewR2 source에 같은 고정 PLAN/예산/tolerance를 결속해 첫 실제 allocation 내부에서 qualification하고, actual receipt/compatibility SHA 및 complete2000 READY를 다시 검산한다. 추가 과학 fit/pilot/모델·stats·P 다운로드나 재계산0.

이번 직접 수동 recall에만 새 immutable authority/attempt를 결속한다. 원 nonce 중복 방어는 기본 그대로이며, 정확 prior R1 source/config/lock/7terminal IDs와 target R2를 대조한 명시 receipt가 없으면 재등록을 거절한다. 새 primary의 technical READY afterok와 자원 lane afterany를 구별해 head 실패 시 후속이 GPU를 할당받고 연쇄 실패하지 않게 한다. 품질 score/성능 gate/자동 retry가 아니다. 아래 R1 등록 시점의 PENDING 기록은 당시 snapshot으로 보존하며 현재 상태가 아니다.

R2 최종 owner CPU192개 중190 PASS/2 SDK 의존 skip/실패0, 모델/GPU 초기화0. 독립 member9개 및 수동 재등록/DAG17개 검산 PASS, 별도 실제 GPU qualification은 아직 미수행이다. 새 후속5개에만 `--kill-on-invalid-dep=yes`를 요청하여 technical prerequisite가 실패해 영구 불가능하면 GPU 없이 terminal CANCELLED가 되며, collector는 새6개 afterany로 실패·부분 관측을 회수한다. Primary/collector/기존job에는 이 flag를 적용하지 않는다. 실제 설치된 sbatch parser/man과 CPU argv 검사로 옵션/AND dependency 의미를 확인했고 별도 Slurm smoke/GPU pilot은 만들지 않았다.
