# GPT-J native six-baseline fluency/consistency 준비·등록 보고

상태: `READY_SOURCE_REFERENCE_CPU; SUBMISSION_NOT_YET_EXECUTED`.
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
새 online run/GPU 검증은 아직 미관측이다.

기존 CPU44 PASS는 historical preparation이다. 새 coupling tests는
bridge/shared mapping/runner/launcher/독립 collector에 한정한다.
처음 mock assertion이 prefix와 subset의 endpoint identity가 같다고 요구하여
6 subtest failure였고, 실제 공통 API의 distinct endpoint/same physical-state
계약에 맞게 task fixture를 수정했다. 원 failure receipt 보존;
수치 tolerance/과학조건 변경0. 새 검산 결과·source hash는 CPU integration receipt에 남긴다.
최종 새 CPU38 tests PASS(0 failure/0 error), 4.723초, peak RSS849752064B,
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
현재 물리 VRAM/RAM/disk/QoS를 등록 직전 결속한다. 성능 afterok/gate0.
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
execution/source/lock/jobs는 attempt-r1 등록 이후에만 확정한다.
source freeze와 이후 report/main commit을 별도로 기록한다.
명령은 project/run_scripts/gptj_native_baselines/GENERATION_RERUN_README.md;
prepare/bind/submit은 create-once이며 같은 nonce를 중복 제출하지 않는다.
