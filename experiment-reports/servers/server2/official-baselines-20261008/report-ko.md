# Server2 official baseline runner 구현·입력 결속

Instruction: `USER-OFFICIAL-BASELINES-20261008-R1`.
Owner: server2 / SH2 `01a0493a-074c-7f91-9a13-769116326fef`.

현재 상태는 **실제 native runner/controller 구현 완료, 공통 source 입력 결속 중, 신규 Slurm 미제출**이다.
이 보고는 CPU 준비를 GPU qualification 또는 2K 실험 완료로 표시하지 않는다.
기존 server2 실행/source/raw는 변경·취소하지 않았다. OURS는 이번 격자에 없다.

## 실행 source와 범위

전용 WT: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-official-baselines-20261008-r1`.
Branch: `codex/server2-official-baselines-20261008-r1`.
배포 정본은 main `537509729345c26efa641ac9f026993e5f5c036e`, 최초 official tree
`c750f75d14b722233810cc790cdc128d0edb965c`이다. 최종 실행 main commit/tree는 공통
factual/tracking source와 검토된 own runner가 통합된 뒤 별도로 동결한다. 최초 배포 SHA를
새 실행 source로 잘못 표시하지 않는다.

GPT-J FT/MEMIT/AlphaEdit/AlphaEdit-BLUE/FE/SPHERE × CF/zsRE **12행**이다.
각 cold first2000, batch100×20, edit seed0, FP32/eager/TF32 off/autocast off다.
공통 준비 격자는 main36행+Qwen 보조5행이며 selected-grid alias를 제외하면40 편집 chain이다.

`native.py`는 algorithm/parser/request/call options를 `official.baselines.registry`에서 가져온다.
EasyEdit는 model/data/C0/P 경로에만 사용하며 external EasyEdit algorithm import는 없다.
공유 수학·hparams 파일은 수정하지 않았다.

## 실제 자산 준비

Ignored manifest:
`/mnt/raid5/janghj/ODE-edit/local/official-baselines-server2/20261008-r1/preparation-r1/assets.json`

- manifest SHA256: `630892fba38af485cfd05e3c45f8c6c36ac759003446897457a03f1f067a024a`.
- assets identity: `729e805b497a02bf49fbd9dd7b0f5448fb685eca347a047834d013b1797d5af1`.
- GPT-J revision: `47e169305d2e8376be1d31e765533382721b2cc1`.
- Python: `/mnt/raid5/janghj/EasyEdit/.venv/bin/python`; torch2.9.1+cu128 / Transformers4.57.1.
- CF fixed10k prefix와 EasyEdit zsRE 입력을 official stream/order/tokenizer locks에 대조했다.
  첫2000과20 batch 경계가 일치한다. model forward 없이 실제 tokenizer CPU 검사했다.
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

CF factual은 strict NLL/tie failure와 request macro E/G/S 및 raw/display rounding을 구분한다.
zsRE는 teacher-forced token request macro, W0 prediction agreement, loc_ans accuracy를 분리한다.
CF 생성은 새 비교의 **모델당 W0 1회 + 각 W20 chain 2K 1회**다. 중간 생성은 없다.
공식 native case-padded KV/global endpoint RNG/total100/noEOS를 사용하며 같은 생성 결과로 FLU/CON을
함께 계산한다. old W20-only/partial 관측을 새 비교 W0로 바꾸지 않는다.

## 검산·등록 상태

- Server2 전용 CPU fixtures: 58/58 PASS(실행 당시). 실제 GPU/model forward0.
- official source verifier: source134/python191, source integrity PASS, external task import0,
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

신규 actual job IDs: **없음**. source/config/execution lock/job/log 경로는 실제 봉인·등록 이후 기록한다.
공통 SH1 factual API `evaluate`/`build_zsre_w0_reference` 정보는 direct 수신했으나 main publication은
아직 결속 전이다. 공통 official W&B transport는 GH/SH1의 source/API 검토 중이다. 기존 helper는
old W20-only nonce와 price prompt-pair schema에 묶여 있어 새 W0+W20/request-macro를 조용히
오표기할 수 없다. helper를 복제하거나 offline PASS로 대체하지 않았다.

source/API readiness 요청은 repo app-server direct로 SH1과 GH에게 전달했다. 추가 USER 승인 요청이
아니며 입력이 결속되면 해당 source로 execution manifest를 만들고 actual held 등록을 진행한다.
현재 상태를 scheduler PENDING·online verified·GPU PASS로 표시하지 않는다.

## 보존·인계

원 dirty root/다른 worker/source/archive/raw/기존 jobs KEEP. 새 recurring monitor/heartbeat/retry는 없다.
`NO_BROADCAST_NOT_REQUIRED`: 같은 서버의 기존 exact 자산을 재사용하며 소형 source/report만 Git 공유한다.
대형 관측/CP/reference/tensor/credential은 local-only다.
