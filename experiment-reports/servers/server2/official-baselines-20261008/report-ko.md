# Server2 official baseline runner 구현·입력 결속

Instruction: `USER-OFFICIAL-BASELINES-20261008-R1`.
Owner: server2 / SH2 `01a0493a-074c-7f91-9a13-769116326fef`.

현재 상태는 **실제 qualification GPU6+CPU collector 등록·held 검사·release 완료**다.
이 보고는 CPU 준비를 GPU qualification 또는 2K 실험 완료로 표시하지 않는다.
기존 server2 실행/source/raw는 변경·취소하지 않았다. OURS는 이번 격자에 없다.

## 실행 source와 범위

전용 WT: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-official-baselines-20261008-r1`.
Branch: `codex/server2-official-baselines-20261008-r1`.
배포 정본은 main `537509729345c26efa641ac9f026993e5f5c036e`, 최초 official tree
`c750f75d14b722233810cc790cdc128d0edb965c`이다. 최종 실행 main commit/tree는 공통
factual/tracking source와 검토된 own runner가 통합된 뒤 별도로 동결한다. 최초 배포 SHA를
새 실행 source로 잘못 표시하지 않는다. 검토된 실행 source는 main
`18e7fbd99bc2692c9da9ebdc9f4598caba79a1f8`, official tree
`7f550b1a02b29173da91372e25dd9279c334504c`로 동결·archive했다.

GPT-J FT/MEMIT/AlphaEdit/AlphaEdit-BLUE/FE/SPHERE × CF/zsRE **12행**이다.
각 cold first2000, batch100×20, edit seed0, FP32/eager/TF32 off/autocast off다.
공통 준비 격자는 main36행+Qwen 보조5행이며 selected-grid alias를 제외하면40 편집 chain이다.

`native.py`는 algorithm/parser/request/call options를 `official.baselines.registry`에서 가져온다.
EasyEdit는 model/data/C0/P 경로에만 사용하며 external EasyEdit algorithm import는 없다.
공유 수학·hparams 파일은 수정하지 않았다.

## 실제 자산 준비

Ignored manifest:
`/mnt/raid5/janghj/ODE-edit/local/official-baselines-server2/20261008-r1/preparation-r2/assets.json`

- manifest SHA256: `83ebd082f4edef211c9ac5e38515a994b7f6430464c57fc64574c8bf89ea9ec8`.
- assets identity: `277c7db5a29aca48b1dd755b77a7043725c131eea30befddbc96c6d016b8ca0a`.
- 원 preparation-r1은 그대로 보존했다. 새 공유 source를 r1이라고 재표기하지 않는다.
- GPT-J revision: `47e169305d2e8376be1d31e765533382721b2cc1`.
- Python: `/mnt/raid5/janghj/EasyEdit/.venv/bin/python`; torch2.9.1+cu128 / Transformers4.57.1.
- CF fixed10k prefix와 EasyEdit zsRE 입력을 official stream/order/tokenizer locks에 대조했다.
  첫2000과20 batch 경계가 일치한다. model forward 없이 실제 tokenizer CPU 검사했다.
- 공유 factual 실제 `_plan`으로 GPT-J CF 26000 prompt-pair/52000 candidate, zsRE6000
  teacher-forced candidate를 전량 대조했다. concat-boundary/suffix/order PASS, 모델 forward0.
  receipt SHA: `6a3f60685f6f2410fd5ffdf38f51c8928c7532a54ed214c2177bc52c9806f9be`.
- C0 L3..8 NPZ header/raw FP32 mom2/count **54924275**, sample_size100000을 확인했다.
  공개 native `moment().float().cpu()` reduction과 cache key를 연결한다. 자동 수집은 차단했다.
- P full six FP32 slots L3..8를 결속했다. BLUE L3/L8는 physical slots0/5이며 H는 별도2슬롯이다.
  기존 fullSHA/finite/schema receipt와 현재 stat를 재사용했다. 원 P construction receipt는 없으므로
  threshold.02의 새 spectral 생성 증명을 주장하지 않는다. 새 생성으로 우회하지 않았다.
- FLU/CON reference/NLTK/scoring versions는 기존 SHA와 현물 경로를 유지한다.
  원문·tensor·model·dataset은 Git에 넣지 않는다.

## runner·checkpoint·평가 연결

`run.py`는 실제 model load → native apply → scheduled factual/generation → atomic checkpoint 흐름이다.
FT는 native 선택 fc_out weight와 bias를 모두 저장한다. Alpha/BLUE/SPHERE의 native cache_c와
context는 arm별로 독립이다. GPT-J initializer gap은 명시 native state 결속으로 처리하며 solver fork가 아니다.
BLUE Tensor/tuple source 연결과 actual native 동작은 GPU qualification에서 따로 확인한다.

이번 USER의 checkpoint 예외를 적용한다. W0 및 매 완료 batch의 selected FP32 W/native H/RNG/
context/cursor/identity를 최신1개로 보존하고 W20은 유지한다. 원 자산/옛 CP에는 삭제 권한이 없다.
재개는 실제 원 checkpoint 폴더를 진행시키되 새 attempt의 tracking/log/terminal을 분리한다.
checkpoint fsync 이후 ledger-write 사이 중단은 durable 실제 receipt로 metadata만 복구한다.

GPU qualification은 method별 continuous B1/B2/B3와 **실제 disk B2→B3**를 비교한다.
W/H/context/RNG/per-case metrics exact equality를 요구한다. method당 native100 요청 호출4회의 기술
검산 비용은 본20-batch 과학 chain 비용과 구분한다. 아직 실제 qualification receipt는 없다.
첫 B1 factual의 기존 첫 forward를 passive probe로 관측하여 native fc_out+bias,
GPT-J Tensor/tuple/readout27→ln_f→untied biased head, suffix/token/FP32 NLL 식을 대조한다.
새 forward/fit/write0이며 측정 전 plan/tolerance/source를 봉인한다. 이것은 **owner formula
control**이지 독립 원본 evaluator oracle PASS가 아니다. 원본 parity와 resume를 혼동하지 않는다.
`CF_original_evaluator_parity`는 별도 요구 상태이며 GH에 exact source/reference 범위를 요청했다.
게시본 수신으로 source 입력 gap은 해제했다. CF native/resume 실제 증거로 본등록을 허용하되,
새 W0에서 실제 original 비교가 성공한 뒤만 READY/후속이 열리도록 분리했다.

CF factual은 strict NLL/tie failure와 request macro E/G/S 및 raw/display rounding을 구분한다.
zsRE는 teacher-forced token request macro, W0 prediction agreement, loc_ans accuracy를 분리한다.
CF 생성은 새 비교의 **모델당 W0 1회 + 각 W20 chain 2K 1회**다. 중간 생성은 없다.
공식 native case-padded KV/global endpoint RNG/total100/noEOS를 사용하며 같은 생성 결과로 FLU/CON을
함께 계산한다. old W20-only/partial 관측을 새 비교 W0로 바꾸지 않는다.

## 검산·등록 상태

- 봉인 실행 source의 Server2 CPU fixtures: 97/97 PASS. 등록 제어 repair 포함105/105 PASS,
  control 전용26/26 PASS다. 실제 pretrained/GPU 관측0.
- official source verifier: source148, source integrity PASS, external task import0,
  GPU qualification NOT_RUN.
- 별도 scoped independent source review: run/native/submit/collect; reviewer가 작성한 assets/generation은
  독립 검토 주장에 포함하지 않는다. 발견된 resume/context 문제를 수리했다.
- `submit.py`는 published main+official tree gate, 실제 owned queue/DAG/cap2,
  held full argv/owner/source/memory/dependency/script 검사→release와 GPU0 collector를 연결한다.
  CF는 실제6방법 qualification aggregate 후 W0 technical READY를 결속한다.
  zsRE는 fresh W0 token reference→한 batch smoke→본 실행을 분리한다.
- 요청 자원 기본: GPU1/CPU6/59392MiB/48h(exportNONE/Requeue0), CPU collector6/24576MiB/4h.
  wall은 ETA가 아니다. 새 checkpoint/raw/error reserve 계획은128GiB이며 제출 직전 다시 검사한다.
- 한정 admission 관측에서 기존61535가 RUNNING,61534는 resource PENDING,61536..39는 dependency
  PENDING이었다. 전체 admitted graph를 보존하고 새2lane은 실제 surviving frontier 뒤에 연결한다.
  이 관측을 과학 진행률/결과 모니터링으로 사용하지 않는다.

첫 qualification FT job **61619**를 실제 held 등록했다. `PENDING/JobHeldUser`,
AllocTRES 없음·RunTime0이며 release 전에 dependency 검사에서 멈췄다.
Slurm은 요청 `afterany:61538:61539`를 같은 AND 의미인
`afterany:61538(unfulfilled),afterany:61539(unfulfilled)`로 저장했다.
제어부 정규화·동일 attempt 명시 continuation을 source
`fed0550c0b790431c04ec8a26865ba689c2a8236`으로 수리했다. 이미 등록된 FT를 재제출하지 않고
원 archive/source/failure receipt를 보존했다. 실행 source `18e7fbd`는 변경하지 않았다.
초기 execution lock SHA: `84c6f1397a42de671261f8a2b52ce5acd8efa57d01102d673c6d46cb6786929e`.
최초 부분 등록은 `registration-partial-r1.json`에 그대로 보존했다. 후속 단회 continuation에서
전체 held owner/source/full argv/input/resources/dependency 검산 후 아래 경로를 release했다.

| 경로 | 실제 job ID | afterany dependency |
| --- | --- | --- |
| FT | 61619 | 61538, 61539 |
| MEMIT | 61624 | 61538, 61539 |
| AlphaEdit | 61625 | 61619 |
| AlphaEdit-BLUE | 61626 | 61624 |
| FE | 61627 | 61625 |
| SPHERE | 61628 | 61626 |
| CPU collector | 61629 | 위 GPU6개 |

단1회 release 직후 snapshot은 전부 **PENDING**, reason은 당시 `None`이었다. 이를 별도로 관측하지
않은 `Resources` 이유 또는 RUNNING으로 바꾸지 않는다. 두 lane cap2, 기존61534..40/61428 및
다른 작업은 유지했다. 이 등록은 native/resume 기술 qualification이며 **CF/zsRE12개 2K 본실행
등록·완료가 아니다**. 실제 GPU qualification/resume/W&B startup은 아직 NOT_OBSERVED다.
source/lock/receipt와 GH app-server 직접 handoff는 `registration-handoff-r1.json`에 결속했다.
공유 tracking b10a87df / official tree85f2cb7b와 factual source596896ff/publication55afa07d/
official treee79c3939를 exact 대조해 병합했다. factual.py SHA256은
`2bc41883b9261d084a3b4b99e0920911789659401a6d9b2b0f0d801a446ab47b`다.
`official.tracking` 단일 readonly init/log/finish를 사용하며 logger 복제는 없다.
schema는 official-baselines-scalar-v1, CF W0_AND_W20_FIRST2000, zsRE CF-generation 생략이다.
공식 request-macro는 `official/*`, 실제 CF prompt/token9-field는 기존 R/P/N에 별도 기록한다.
milestone current100은 같은 prefix raw 마지막100을 CPU 집계하며 추가 forward0이다.
공식 일정에 없는 current/pre는 미측정으로 보존하며 값이나 zero를 만들지 않는다.
W20 progress는 공유 official_generation_progress를 통해 phase만 W20_generation으로 바꾼다.
기존 native 숫자 출력만20초/phase/batch 경계에 전달하며 fit 단조 cursor를 checkpoint에 묶는다.
rounded native loss/NLL/KL과 FT printed loss를 구분하고 추가 tensor conversion/GPU sync0이다.
CPU config/scalar PASS는 auth/원격 readback PASS가 아니다. 실제 init은 새 job 내부에서 수행한다.

source/API 및 원본 evaluator parity 범위 요청은 repo app-server direct로 SH1/GH에게 전달했다.
추가 USER 승인 요청은 아니다. actual held 등록은 published main/tree/현재cap/자원/input/argv를
결속하고 수행한다. 원본 parity 입력 미결속 상태는 CF 본실행에서 정확히 차단한다.
등록 전 준비 상태와 실제 등록 후 PENDING·online verified·GPU PASS를 구분한다.
qualification은 독립 원본 evaluator 없이도 native/resume 검산만 수행할 수 있다. 등록 단계의
계획과 실제 측정 receipt는 구분한다. 원본 evaluator 추가로 official tree가 바뀌면 기존
qualification을 새 source의 PASS로 재표기할 수 없다. CF W0 READY와 후속 본실행에는 실제 원본
proof의 source/member/SHA 결속이 필요하며 현재 PASS 문자열이나 owner 식 검산으로 대체하지 않는다.
오류 기록에서 checkpoint metadata가 손상돼도 최초 scientific 예외를 유지하도록 좁게 보완했다.

## 독립 원본 scorer 신규 source 입력

`GH-SH2-OFFICIAL-NATIVE-ORACLE-READY-20261009-R1`을 직접 수락했다. main
`34001ec0950f00b61e89be753494f40b9da6f70f`의 API/lock/envelope 전체와 SHA를 검산했다.
원 public AlphaEdit `test_batch_prediction` AST가 독립 model forward를 수행하는 공통
`official.evaluation.cf_native_reference`를 읽기 전용 재사용한다. owner logits/formula/resume를
원본 PASS로 대체하지 않는다. 새 source에 first4 사전고정 cohort/state/model/tokenizer/runtime와
NLL abs1e-4/rel1e-5, 집계1e-10pp, strict bits exact 조건을 결속한다.
원 qualification archive와 실행 bytes는 그대로 유지하며, 새로운 oracle 입력과 과거 실제 resume
증거의 producer provenance/consumer binding을 분리한다. 새 source 입력 수신·CPU 연결은 actual
GPU oracle PASS가 아니다. 실제 증거 전 CF/zsRE 본실행 gate를 성공으로 표시하지 않는다.

미래 source의 Server2 CPU137/137 PASS, oracle 독립공유 CPU17/17 PASS 및 source157 SHA/
external-task imports0을 확인했다. 실제 pretrained/GPU·원격 W&B 증거는 아니다.
새 `preparation-r3/assets.json` SHA는
`a91d01c42613606bc2784179cb098b6cde5cc3e9103d12d8321315f43760283c`, assets identity는
`7dbdd126d3155d0fd874f7adef01a8a706155796d8aeaebd2459e5a25548ebbc`다.
source provenance가 달라 전체 assets ID가 바뀌는 것을 숨기지 않는다. 모델/입력/C0/P/tokenizer/
runtime/reference의 physical 내용은 이전 manifest와 정확히 일치하며 생성/다운로드/모델 forward0이다.
179개 native computational source와15개 핵심 runner 함수 AST 및12 cell config가 동일한 조건을
사전 봉인한다. 과거 actual resume aggregate/원 per-case/cost/source/CP identity는 그대로 보존해
검산하며 consumer binding을 별도로 기록한다. display-only rounding 예외는 정확한 두 source SHA로
제한한다. 이 compatibility로 old CP를 새 source로 재개하지 않으며 새 chain은 cold W0부터 시작한다.
실제 qualification collector aggregate가 아직 없으므로 미래 본등록은 INPUT_PENDING이다.

## 보존·인계

원 dirty root/다른 worker/source/archive/raw/기존 jobs KEEP. 새 recurring monitor/heartbeat/retry는 없다.
`NO_BROADCAST_NOT_REQUIRED`: 같은 서버의 기존 exact 자산을 재사용하며 소형 source/report만 Git 공유한다.
대형 관측/CP/reference/tensor/credential은 local-only다.
