# Server2 official baseline runner 구현·입력 결속

Instruction: `USER-OFFICIAL-BASELINES-20261008-R1`.
Owner: server2 / SH2 `01a0493a-074c-7f91-9a13-769116326fef`.

현재 상태는 **공통 factual/tracking 및 runner main 게시, 실제 qualification 등록 제어 수리 중**이다.
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
CF main gate는 이 증거를 resume PASS로 대체하지 않는다.

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
제어부 정규화·동일 attempt 명시 continuation을 수리한다. 이미 등록된 FT를 재제출하지 않고
원 archive/source/failure receipt를 보존한다. 나머지 실제 IDs/release는 후속 증거로 분리한다.
초기 execution lock SHA: `84c6f1397a42de671261f8a2b52ce5acd8efa57d01102d673c6d46cb6786929e`.
등록 상태는 `registration-partial-r1.json`에 기록했다. 아직 실제 GPU/online PASS는 없다.
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
현재 상태를 scheduler PENDING·online verified·GPU PASS로 표시하지 않는다.
qualification은 독립 원본 evaluator 없이도 native/resume 검산만 수행할 수 있다. 등록 단계의
계획과 실제 측정 receipt는 구분한다. 원본 evaluator 추가로 official tree가 바뀌면 기존
qualification을 새 source의 PASS로 재표기할 수 없다. CF 본실행에는 실제 원본 proof의
source/member/SHA 결속이 필요하며 현재 PASS 문자열이나 owner 식 검산으로 대체하지 않는다.
오류 기록에서 checkpoint metadata가 손상돼도 최초 scientific 예외를 유지하도록 좁게 보완했다.

## 보존·인계

원 dirty root/다른 worker/source/archive/raw/기존 jobs KEEP. 새 recurring monitor/heartbeat/retry는 없다.
`NO_BROADCAST_NOT_REQUIRED`: 같은 서버의 기존 exact 자산을 재사용하며 소형 source/report만 Git 공유한다.
대형 관측/CP/reference/tensor/credential은 local-only다.
