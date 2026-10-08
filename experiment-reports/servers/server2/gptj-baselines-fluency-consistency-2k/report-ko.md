# GPT-J native six-baseline fluency/consistency 준비·등록 보고

최신 상태(2026-10-08): `MANUAL_R2_SUBMISSION_HANDOFF; SIX_GPU_AND_COLLECTOR_RELEASED; RESOURCE_PENDING`.
이하 기존 등록 보고는 역사 기록이며, 현재 실행 상태로 읽지 않는다.
원 등록 상태: `SUBMISSION_HANDOFF; SIX_GPU_AND_CPU_COLLECTOR_RELEASED`.
Instruction/nonce `USER-GH-ALL-SH-BASELINE-FLUENCY-CONSISTENCY-RERUN-20261007-R1`,
SH2/server2/session `01a0493a-074c-7f91-9a13-769116326fef`.
원 dirty root와 기존 task/source/raw를 보존한다. 입력 READY·CPU fixture 통과는 실제 GPU/생성/편집 완료가 아니다.

## 통신과 실제 입력

사용자 지시에 따라 repo app-server 정책의 SSH→Unix socket WebSocket
initialize/initialized→thread/resume→idle turn/start를 사용했다.
GH accepted turn `01a1165b-2002-7783-8df5-b48abda9086c`,
request nonce `SH2-GH-GENERATION-READY-REQUEST-20261007-R1`.
원 bounded90초 terminal timeout을 보존하고, 같은 accepted turn 한정 read-only
recovery로 completed 및 nonce final ACK를 회수했다. 중복 send/steer/interrupt0,
모델/effort 변경0. 최종 relay nonce
`GH-SH2-GENERATION-READY-RELAY-20261007-R1`도 직접 수신 ACK했다.

초기 dynamic wrapper unavailable/Git 게시를 live 전달의 대안으로 취급한 처리는
owner 오류로 정정한다. 역사 request/main43b5ff84와 실패 기록은 그대로 보존했고,
Git 게시만으로 GH 직접 수신을 주장하지 않는다.
최종 응답은 `audits/servers/server2/gptj-baselines-fluency-consistency-2k/app-server-final-recovery.json`.

공통 source `83535c6a47c552cc4e5c6385f3a587d752820150`,
publication `2828ab0938184ae863831c6e5e93fcf7a021eca4`,
package tree `6b9ed049ddaf0cfd431d036125ea8aa7be3724d1`.
SH1의 별도 실행6bc51602와 혼동하지 않는다. SH2는 공통 source를 수정하지 않았다.

승인된 exact reference pull로 manifest/READY와 attribute_snippets.json/idf.npy/
tfidf_vocab.json만 수신했다. SOURCE_KEEP/nooverwrite/nodelete.
총 신규 수신968974727B, 10.281초, 각 size/fullSHA 일치.
Reference identity `75e595c7f26ec334830e9bb9ca6028098c19ea84a9509a5713985847683f8ea6`.
Manifest6769B SHA
`6d9a713ab7eaa10871f277e10a3974e0bd5b140c265be80052c279f258876ca8`.
상세 exact allowlist는 task audit/reference-pull-allowlist.json,
수신 원본은 ignored local/inputs/reference-r1 아래다.

Server2 실제 공통 load_assets:8.246초, peak RSS2837356544B.
NumPy2.2.6/SciPy1.15.3/sklearn1.7.2/NLTK3.6.5 및 active English punkt의
source/resource identity 일치. reference/generation prompt coverage 각각2000/2000,
누락0. 자산 refit/download·모델 load·새 GPU 계산0.
Final input config SHA
`51d822e131bcc096e43438dd699e87205fa4ecbcccd44388e673f16eae9af4c7`.
preparation-r1의 provisional/CPU44 원본은 보존했다.

## 기존 baseline 취소 대조

이번 exact reconcile에서 모두 terminal이므로 신규 scancel0이다.

| ID | 역할 | 상태 |
| --- | --- | --- |
|60656/60657/60658|stock MEMIT/AlphaEdit/collector|COMPLETED|
|60769/60770|CAKE/AlphaEdit-BLUE|FAILED|
|60771/60772/60773|PRUNE/RECT/collector|CANCELLED|

PRICE/ours/W0/FE·다른 서버/owner job 불변. 이 표는 전체 owner queue0 주장이 아니다.

## 구현·검증 수준

기존 stock56d3a445와 four-arm3a4a107b의 model/revision/native source/hparams/
C0/P/input/scorer를 exact 재사용한다. heavy assets는 과거 fullSHA+현재 stat,
작은 source는 새 SHA 수준이다. 모델/stat/P 복제·재계산0.

CAKE/BLUE guard는 stock GPT-J fc_out bias를 보존하며 기존 B1 forward의
shape/dtype/finite를 관측한다. 추가 forward/GEMM/fit0; actual affine
numerical parity는 NOT_MEASURED. PRUNE은 W20 평가 전에 명시
PRUNE_TERMINAL_BASE_FIX를 한 번 적용한다.

Thin bridge는 실제 SH1 load_assets→observe/subset/read_observed와
generation_payload를 사용한다. 원 rows/summary/sums/counts/missing reasons/
rows_path/work/physical state 및 raw receipt를 보존한다.
Cold W0 READY identity는 full cold model에 결속하고 arm-local H를 공유하지 않는다.
자기 H/context/cache/counter와 RNG finally restore/nonmutation는 별도 검사한다.
B1pre는 W0 CPU subset, milestone current는 prefix subset이다.
subset과 prefix는 동일 physical state/원 observation을 쓰지만 endpoint identity는 다르다.

Production logger는 SH1 공유 schema/client/worker를 read-only 재사용한다.
Private logger/CPU26는 역사 fixture이며 현재 transport가 아니다.
단위는 fluency bits/consistency cosine; missing mean은 omit, 지표0으로 대체하지 않는다.
actual Slurm jobID/run.name/config와 immutable runUUID를 봉인한다.
SDK 접수와 remote readback, 프로그램 종료와 science completion을 구분한다.
등록 후 bounded 대조에서 새 online startup·첫 actual write는 아직 미관측이다.

기존 CPU44 PASS는 historical preparation이다. 새 coupling tests는
bridge/shared mapping/runner/launcher/독립 collector에 한정한다.
처음 mock assertion이 prefix와 subset의 endpoint identity가 같다고 요구하여
6 subtest failure였고, 실제 공통 API의 distinct endpoint/same physical-state
계약에 맞게 task fixture를 수정했다. 원 failure receipt 보존;
수치 tolerance/과학조건 변경0. 새 검산 결과·source hash는 CPU integration receipt에 남긴다.
원 integration-r1 CPU38 PASS(4.723초)는 역사 receipt로 보존했다.
EOF 정리와 final CPU receipt 결속 후 같은 좁은 integration-r2를 수행했다.
최종 새 CPU38 tests PASS(0 failure/0 error), 3.990초, peak RSS850006016B,
threads1/CUDA initialized=false. prior bias/plan CPU는 unchanged source SHA로 재사용했다.
실제 모델/생성/native fit/network/Slurm는 이 검사에서 모두0이다.
독립 worker fixture/reducer 검토는 사용했으나 actual pretrained/GPU red PASS를 만들지 않았다.

## 실행·자원·저장 경계

6 cold arms 각first2000 BS100×20: unique2000/총12000 edit applications.
BASE_MEMIT이 fresh cold W0 generation sole publisher다.
기존 exact S2 GPU frontier 뒤 BASE_MEMIT, 그 종료 뒤 cap2 두 lane
BASE_ALPHAEDIT→ALPHAEDIT_BLUE→RECT 및 CAKE→PRUNE.
W0 공유 READY 때문에 첫 전체 chain은 직렬이며 GPU file polling/callback/new auto-submit0.
collector는 six GPU afterany. 더 엄격한 cap1이면 전량 직렬.

각 GPU1/CPU6/59392MiB/48h; collector0GPU/CPU6/24576MiB/4h.
48h는 요청 wall이며 ETA/완료 보장이 아니다. fresh owner/source/node/dependency와
현재 물리 VRAM/RAM/disk/QoS를 등록 직전 결속했다. 성능 afterok/gate0.
예상 raw reserve16GiB/arm, 동시32GiB; reference 약0.903GiB 별도.
실제 generation/fit/solve/eval/guard/IO/할당시간·peak·유효 분모는 runner/collector가 기록한다.

53600 case observation 참조(고정10 prompt이면536000 참조)에는 cache reuse가 포함되며
신규 model forward 수가 아니다. B1pre600case는 exact cold subset으로 재사용한다.
두 metric을 위해 재생성0. 입력 seed20261002 / observer seed20261007을 구분한다.

NoCP, exact_resume=NOT_AVAILABLE, z disk cache None.
모델/W/H/RNG/optimizer/복원 delta durable0. raw text/token/전체stdout/secret는 local-only,
Git/W&B 업로드0. 원 source/raw/실패비용 KEEP.
reference exact 단회 수신만 수행했으며 새 생성 raw를 타 서버로 복제하지 않는다.
그 외 NO_BROADCAST_NOT_REQUIRED. 신규 recurring monitor/heartbeat/automatic retry0.

## 원본·명령

Ignored root:
`/mnt/raid5/janghj/ODE-edit/local/gptj-baselines-fluency-consistency-2k/`.
reference inputs/reference-r1, preparation-r1 역사, preparation-r2/config.json 및 binding.json.
execution/source/lock/jobs는 아래 attempt-r1 실제 등록 receipt에 결속했다.
source freeze와 이후 report/main commit을 별도로 기록한다.
명령은 project/run_scripts/gptj_native_baselines/GENERATION_RERUN_README.md;
prepare/bind/submit은 create-once이며 같은 nonce를 중복 제출하지 않는다.

## 실제 등록·최신 recall 대조

Execution source `503081fa9bc6efc4dbdd461324fba522b8f36e8b`,
tree `5610249a31866d1568d1e473288dc5fc68a53d9d`.
Lock88993B SHA `7e38d6a189521d27b3bc06084c71d28d277628b0a9812856c1ad87c70a334cb1`.
이후 compact 보고 게시 commit은 execution identity를 대체하지 않는다.

2026-10-07 22:14 KST에 전량 held 등록, exact owner/Command/full argv/source/
input/W&B/noCP/resources/dependency 검사 후 모두 release했다.

| Arm/역할 | Actual job ID | afterany | 최신 recall 단발 상태 |
| --- | --- | --- | --- |
|BASE_MEMIT|60909|없음|RUNNING|
|BASE_ALPHAEDIT|60910|60909|PENDING/Dependency|
|CAKE|60911|60909|PENDING/Dependency|
|ALPHAEDIT_BLUE|60912|60910|PENDING/Dependency|
|PRUNE|60913|60911|PENDING/Dependency|
|RECT|60914|60912|PENDING/Dependency|
|CPU collector|60915|60909–60914 전체|PENDING/Dependency|

최초 release 직후 snapshot은 전부 PENDING/Reason=None이었다.
추가 USER recall nonce
`USER-GH-SH1-SH2-SH4-BASELINE-GENERATION-REGISTER-RESUME-20261007-R1`
수신 뒤 exact 기존7개를 한 번 대조한 결과 위와 같았다.
Owner janghj/node server2/Command와 lock source 일치, 신규 중복submit0/job변경0.
새 controller root 여유를 당사자 검증이나 GPU PASS로 대신하지 않았다.

등록 전 own S2 project GPU allocation/frontier0을 actual source/owner/node로 확인했고,
새 DAG 가능한 최대 동시GPU2다. 다른 owner/server allocation은 이 task 권한 밖으로 보호했다.
현재 node는 A6000 8개/각49140MiB, CPU64/RealMemory512000MiB이며,
held 검사 때 전체 owner AllocTRES는 GPU4/CPU22/196GiB였다.
새 2lane host 요청118784MiB, 당시 free disk471627784192B/inode443349587.
이는 node 전체 resource 관측이지 science peak나 ETA가 아니다.

Actual W&B startup/readback·첫write/B2연결·W20 completion은 `NOT_OBSERVED`.
등록 성공은 numerical/model/generation certification이 아니다.
완료 분모·native calls·generation·비용은 sealed runner/collector의 실제 결과로만 판단한다.
신규 recurring monitor/automatic retry0, agent는 bounded handoff 뒤 중지한다.

Compact audit: submission.json, registration-resume-reconcile.json,
cpu-integration-final.json. Original held inspection/full scheduler argv와 원 stdout는 ignored local에 보존한다.
GH로 실제 IDs를 app-server 전달하려 했으나 target active turn의 same-task identity가
미확립되어 `COMMUNICATION_HOLD_UNRELATED_OR_UNRESOLVED_ACTIVE_TURN_NO_STEER`였다.
메시지 실행/steer0·자동재전송0이며 Git 게시를 GH live 수신으로 주장하지 않는다.
이 사실은 science 등록 완료와 별도다. 이전 readiness request의 GH final ACK는 그대로 유효하다.

## 2026-10-08 KV/equal-length batching repair 전환

새 instruction/nonce
`USER-GH-SH1-SH2-BASELINE-GENERATION-KV-BATCH-REPAIR-20261008-R1`,
authority `66cddb8fa9c09e475f0ae3f423635091c6a71e3d`를 직접 수락했다.
새 운영 task는 `gptj-baselines-generation-cache-repair`이며 원 parent science는 바꾸지 않는다.
전용 clean worktree/branch에서만 작업하고 root dirty/source/raw와 OURS/W0-only/FE/타서버 job을 보존했다.

현재 owner/source/Command/node/상태를 각 취소 직전 대조한 뒤
collector60915 → RECT60914 → PRUNE60913 → BLUE60912 → CAKE60911 →
AlphaEdit60910 → RUNNING MEMIT60909 순서로 exact 취소했다.
7개 모두 `CANCELLED by 1025`, exact target queue empty를 확인했다.
60909 실제 할당25963GPU-sec=7.2119444444GPUh는 **원 실행 비용**으로 보존한다.
나머지6개는 elapsed0/할당0. 새 실행 비용과 합산하지 않는다.
취소/계정 증거는 `cache-repair/cancellation.json`이며 과거 RUNNING/PENDING 표를 덮지 않았다.

원 W0 raw는 396 complete case/3960 prompt records/9320143B가 검산됐다.
2000 planned 중1604는 완료행 없음이며 숫자0이나 completed로 채우지 않는다.
원 source83535c6a/runtime7bc4cf91/UNPADDED_FULL_PREFIX_NO_CACHE/payload/size/SHA를
그대로 보존한 ignored inventory는1045582B,
SHA `a2297ecf824f45d09adbda7cdbef26b88ff99d14d18e04462cc40be2205aee22`.
새 source로 재라벨하지 않고 actual qualification+compatibility manifest 후에만 새 endpoint에 연결한다.
현재 저장된 native commit0, complete generation endpoint0, full W0 READY0이다.
원 terminal/failure receipt와 RAM H/history 값은 `NOT_RECORDED`이며,
이를 측정된 history0 또는 science 완료로 해석하지 않는다.

SH2 변경은 repair-only identity/bind/held-submit control, source-bound 완료행 reader,
최대8 prompt token-length-only qualification PLAN/실제 GPU receipt validator,
collector/progress scalar 경계에 한정한다. 원 native 알고리즘/hparams/dtype/solver/입력은 그대로다.
SH1 공통 `experiment_generation_eval`은 read-only이고 중복 구현하지 않았다.
수신된 source에 PLAN/cohort/tolerance/API를 실제 commit으로 결속해 봉인했다.
**GPU actual receipt는 제출 선행조건이 아니다.** 첫 replacement BASE_MEMIT 내부에서
reference → singleton KV → equal-length batch를 각1회 측정하고 selected route/고정MB/
원 tolerance/token·EOS·seed·position·state/RNG/비용을 actual receipt로 저장한다.
MB8 실제 폭/row 종료 coverage가 부족하면 PASS를 만들지 않고 검증된 singleton을 선택한다.
OOM production retry/사후 tolerance 완화/추가 fit 또는 B1 pilot0.

원 main89734ff5에서 shared source/API 입력 대기였던 사실과 통신 receipt는 보존한다.
이후 main380d07ef에서 공유 source `199cfe5664355f6f1c9069c72396ec759bb25cec` /
package tree `a10d88b555a96962567846dcb34a456c9e0d7c1c` 게시를 확인하고 전체 API를 읽었다.
Owner bridge/mixed endpoint reader/production PLAN 결속 후 새7개 job을 정식 등록/release했다.
원 config aliases와 full-cold state 표현을 조용히 바꾸지 않고 명시 task-local compatibility를 구성한다.
추가 GH/사용자 승인이나 nonexistent actual qualification을 기다리는 gate는 없다.
입력 결속 후 fresh admission/held 등록을 수행했으며 새 source/IDs는 아래와 같다.
계획 자원은 GPU1/CPU6/59392MiB/48h, collector0GPU/CPU6/24576MiB/4h,
합산cap2 또는 stricter. 원 canceled IDs를 새 dependency로 재사용하지 않는다.

앱 서버 정책으로 SH1 related turn01a1180b-d980-7291-bd19-58b92b002daf에 API 요청을
정확 steer한 사실은 확인했지만 bounded final ACK timeout은 그대로 기록한다. 재전송0.
GH idle turn01a11818-f8a2-7202-b733-c2d9ee634cbb에는 취소/입력 상태를 전달해
completed+nonce final ACK를 실제 회수했다. GH는 READY 입력이 오면 동일 task로 중계한다고 답했다.
Git 게시를 live ACK나 source READY로 대신하지 않는다.

Production config SHA `706636712290a8c52c7f9e233f83c2a8c54d778a583bdc191cb7f9457756b362`,
private PLAN SHA `0cafe3c40cb4772aaa409aac03cf6cdbda10ea935c4c49ead8447d52cab68c83`,
shared PLAN SHA `bdcac9e9e2a877dc1fa521e393bddb88136fa7caace7a5ab93878846036126b3`.
8개 고정 prompt에서 같은 길이 폭6, length100 경계 미관측이다. MB8 전체 폭을
검증한 것으로 주장하지 않으며 actual route 판정에서 미충족 batch는 선택하지 않는다.
CPU 최종93 fixture PASS(0 failure/0 error), 12.104초, CUDA initialized=false;
모델/GPU/과학 생성/Slurm0. 이 개수에 기존 중복 component fixture를 합산하지 않는다.
앞선 CPU r1은 음성 SHA fixture의 RuntimeError/ValueError 기대 차이로 FAILED였고,
guard를 완화하지 않고 fixture 기대만 수정한 r2로 검산했다. 두 receipt 모두 보존했다.
배포 metadata torch2.9.1과 실제 runtime2.9.1+cu128의 표기 차이도 별도 기록했다.
과학 환경 pin/모듈/source를 바꾸지 않았으며 최초 CPU bind 실패는 출력 생성 전이었다.

실제 GPU qualification/새 W&B run/편집/W20는 미실행이다.
CPU fixture와 좁은 read-only control review만 수행했으며 결과·한계는 cache-repair audit에 분리한다.
NoCP/old source·raw KEEP/NO_BROADCAST_NOT_REQUIRED,
추가 다운로드·stats/P 생성·heavy transfer·신규 monitoring/heartbeat/자동 submit retry0.

### 캐시 수리 실제 등록 handoff

Execution source `2e6f6f553ac16cf82a171bcdaa7fb2ace87b4e1b`,
tree `8d1d197b3fea2bf5baaa043852d0f98f0e6a3dc6`.
Lock102251B SHA `ef6ae0ae74018ac0d66233e164cd13eac02c4f7f44ea5a9065a42357e4301684`.
이후 보고/main publication은 execution source와 분리한다.

| Arm/역할 | 실제 job | afterany |
| --- | --- | --- |
|BASE_MEMIT|61160|없음|
|BASE_ALPHAEDIT|61161|61160|
|CAKE|61162|61160|
|ALPHAEDIT_BLUE|61163|61161|
|PRUNE|61164|61162|
|RECT|61165|61163|
|GPU0 collector|61166|61160–61165 전체|

전량 held owner/Command/fullargv/script/source/config/PLAN/input/reference/W&B/noCP/
CPU/RAM/GPU/wall/dependency 검사 후 후속부터 모두 release했다.
등록 전 own Server2 project GPU allocation/frontier0, effective cap2를 실제 source/node로
확인했으며 과거 canceled609xx를 dependency에 재사용하지 않았다.
61160은 actual qualification 및 공유 W0 READY 생산 후 첫 cold native chain을 진행한다.
그 종료 뒤 두 lane이 READY/actual receipt/compatibility를 검산한다. source/READY 기술의존성은
failclosed이고 성능 afterok gate·GPU file polling은 없다.

단일 release 직후 snapshot은 **7개 모두 PENDING/Reason=None**이다.
이는 terminal science가 아니며 scheduler 후속 상태를 반복 조회하지 않았다.
첫 GPU qualification/selected route·고정MB/W0 READY/W&B startup/첫write/W20는
`NOT_OBSERVED`로 인계한다. 원 7.211944GPUh와 새 allocation 비용을 중복 합산하지 않는다.
최소 source/API/fixture seam은 분리 worker와 root가 검토했으나 actual GPU independent PASS는 없다.
현재 새 agent monitoring/heartbeat/자동 retry0, sealed runner/collector는 승인 범위로 자연 진행한다.

실제 IDs는 repo app-server 정책으로 GH idle turn
`01a11840-23c5-7351-8ae7-d411b37c9061`에 전달 접수됐다.
Nonce `SH2-GH-GENERATION-CACHE-SUBMISSION-HANDOFF-20261008-R1`; bounded completed 및
final nonce echo를 실제 회수했다(`cache-repair/app-server-handoff.json`).
GH는 실제 IDs/MB8 미검증/qualification 미관측 구분을 ACK했고 job polling은 하지 않았다.
이 통신 성공도 science/GPU/remote metric PASS와는 별개다.
Compact 감사 `cache-repair/submission.json`; 전체 held scheduler receipt/원 stdout/raw는 local KEEP.

Source+compact report는 own branch와 main의
`de80205f70f1a500cdd99622e36d649597cfa848`에 비강제 게시됐고 두 remote ref exact SHA를 확인했다.
첫 main push의 remote temporary pack/index-pack 오류는 보존했다. 같은 commit의 branch 게시를
실제 확인한 뒤 nonforce ref 업데이트로 main을 통합했으며 force/원격 관리/반복 retry는 없었다.
Latest main의 SH1 shared member 검산 수정도 보존했지만 이미 제출된 execution archive/config/lock는
hotpatch하지 않았다. 원 CPU93/source2e6f6f55 evidence와 이후 publication source를 구분한다.
당시 실제 장비는 RTX A6000 8개/각49140MiB, own project allocation0,
disk425102852096B·free inode443311699였다. 이는 실제 generation peak/ETA가 아니다.
Bounded submission 인계 뒤 추가 job 조회/agent monitoring 없이 USER recall을 기다린다.

## 최신 USER recall: r1 logger 실패와 별도 r2 준비

사용자 “server2에서 진행된 baseline들 실험 다시 올려봐. fail되었다.
Wandb에 실시간으로 기록하는것도 진행시켜”를 직접 수락했다.
대상은 바로 앞 여섯 GPT-J baseline(MEMIT/AlphaEdit/CAKE/AlphaEdit-BLUE/PRUNE/RECT)이며
FE/OURS/W0-only/다른 서버는 재개·변경하지 않는다.
전용 clean worktree/branch에서 기존 source/raw/spool/원 dirty root를 보존한다.

61160–61165는 모두 FAILED,61166 collector는 COMPLETED이나 science_complete=false다.
현재 exact7개 queue는 empty, 이미 terminal이므로 이번 scancel0.
각 GPU 할당10/9/9/9/8/8초, 합53GPU-sec(0.0147222222GPUh)는 실패한 r1 비용으로 분리한다.
원60909의25963GPU-sec는 그대로 별도 보존하며 중복 가산하지 않는다.

여섯 sidecar 모두 인증·원격 config/job name 확인 뒤 **READY_ONLINE**을 반환했다.
그 후 parent Tracker._read가 공유 identity.create에서 private config 키
`qualification_plan_sha256`을 검산하여 `ValueError: CONFIG_NOT_ALLOWLISTED`로 거절했다.
Broad catch가 이를 `LOGGING_DEGRADED_CONTROL`로 표시했고 모델 로드 전 종료했다.
따라서 이 실패를 “로그인 누락”으로 분류하지 않는다. 실제 fit/write/commit/history0,
새 actual qualification0. 원 online run UUID/URL/spool은 보존하며 rename/backfill0이다.

최소 수리는 외부 config 키를 기존 공유 schema가 지원하는
`generation_qualification_plan_sha256`으로 바꾼 것이다. 내부 PLAN 키·과학식·native 예산·입력·dtype는
그대로다. 실제 parent pipe-reader→immutable identity 생성 CPU seam을 추가하여 array index0,
signed step -5, PLAN binding, mismatch/privacy rejection와 overwrite 금지를 검사했다.
공유 tracking helper 수정0. SH1이 게시한 rich-member 검산 수리1413ac0f/treee6935b9a도 읽기전용 채택했다.
기존83535c6a raw의 source/runtime/route를 새source로 바꾸지 않는다.

`--profile r2`는 ignored `local/gptj-baselines-fluency-consistency-2k/cache-repair-r2/`에만
새 config/PLAN/source/lock/새W&B UUID를 만든다. r1 기본 경로와 모든 old archive/config는 불변이다.
현재 config SHA1924773f0bbb60addc0a9d1ace8281691a625da96a2099053a4c44e8a920411c,
private PLAN35c53b58…, shared PLANbdcac9e9…를 제출 전 고정했다.
변경 source/실제 parent identity와 기존 좁은 API/control fixture 총99 tests PASS,
11.853초/CUDA initialized=false; pretrained model/GPU/새 실제 qualification은 아직0이다.
독립 실제 GPU 인증이 아닌 owner CPU 검산과 bounded source 검토 수준이다.

각GPU1/CPU6/59392MiB/48h 및 GPU0 collector CPU6/24576MiB/4h,
합산cap2/더엄격현행 제한과 fresh frontier를 등록 직전 검사한다. wall은 ETA가 아니다.
BASE_MEMIT 안에서 actual qualification→compatible W0 READY를 만든 뒤 native trajectory를 진행하고,
종료 후 두 lane이 기술 READY를 검산한다. 추가 fit/pilot/성능 gate·자동 retry0.
W&B scalar-only online, 실제job번호 name/config, separate fit/progress/performance axes,
bounded finish/readback을 유지한다. CPU PASS/SDK 접수/실제 online 검증/실험완료를 별도로 기록한다.
NoCP/raw local KEEP/NO_BROADCAST_NOT_REQUIRED. 실제 등록 ID/새 online 상태는 등록 후 추가한다.
이번 source/RCA receipt: `cache-repair-r2/failure-and-source-review.json`.

### r2 실제 등록·자원 pending 인계

Execution source `f979efa69ce76a00b3e35c295452d01e14d1f13c`,
tree `ad51c8384277eb17aedb76632fc8b967657a0366`.
Lock103380B SHA `da68b2c1402d2c5badbde3250ff6601496e0dca956d97f1eb1ea28c9f22b08fb`.
이후 report/main commit은 이 실행 source를 대체하지 않는다.
6GPU+GPU0 collector 전량 held 검사 후 후속부터 모두 release 완료했다.

| Arm/역할 | 실제 job | afterany | bounded 현재 상태 |
| --- | --- | --- | --- |
|MEMIT|61364|없음|PENDING/ReqNodeNotAvail, May be reserved for other job|
|AlphaEdit|61365|61364|PENDING/Dependency|
|CAKE|61366|61364|PENDING/Dependency|
|AlphaEdit-BLUE|61367|61365|PENDING/Dependency|
|PRUNE|61368|61366|PENDING/Dependency|
|RECT|61369|61367|PENDING/Dependency|
|GPU0 collector|61370|61364–61369 전체|PENDING/Dependency|

실제 janghj/server2 source/Command/fullargv/owner/script/자원/입력/PLAN/추가USER authority/
W&B/noCP/dependency를 검사했다. 전부 새 실제 ID이며 stale611xx/609xx dependency0,
기존 job 변경·취소0. 등록 전 source-based own Server2 allocation0/frontier0이며
새 DAG 가능한 최대2GPU다. node RTX A6000 8개/각49140MiB를 현재 확인했다.
Disk free374937300992B/inode443276424; 각59392MiB/host ceiling60416MiB.
이 값은 실제 과학 peak나 ETA가 아니다.

최초 release snapshot의 Reason=None은 scheduler 초기값이다. 이후 한정 actual resource snapshot은
위처럼 node unavailable/dependency로 확인됐으며 임의로 빈 GPU나 ready로 해석하지 않는다.
새 online startup/immutable run URL/actual qualification/첫write/W20는 `NOT_OBSERVED`다.
첫 새 job의 bounded local startup receipt 검사에서도 아직 receipt가 생성되지 않았다.
로그인 실패를 새로 주장하지 않으며 actual SDK/auth 검산은 실행 안의 cheap startup에서 수행한다.
각 arm은 새UUID/attempt=cache-repair-r2와 실제job번호 run.name/config를 쓰고 scalar를 실시간 전송한다.
SDK accepted와 실제 remote readback, uploader finish와 science completion은 계속 분리한다.

Compact 등록 audit `cache-repair-r2/submission.json`, 전체 immutable source/config/lock/held/currentresource
receipts는 ignored 새attempt 아래 보존한다. 이번 direct USER recall에 대한 접수·제출 사실은
own status/server-head receipt에 기록했다. GH가 직접 수신했다는 주장은 새 app-server ACK 없이는 하지 않는다.
등록된 sealed 프로그램은 W0 qualification/관측·각2k 편집·평가·collector를 자연 진행한다.
Agent는 resourcepending 인계 후 멈추며 새 recurring monitor/heartbeat/automatic retry0.
