# BG-1 / C4 ours-first — 준비 및 G0_BLOCKED 사실 보고

Instruction: ODEEDIT-S06-BG1-C4-OURS-FIRST-SH4-V1. 상태 **G0_BLOCKED / CALIBRATION_MISSING / WAITING_USER_RESUME**.
이것은 BG-1 실행완료·성능 보고가 아니다. Scientific promotion=false.

## 결과와 blocker

동일 W0/order/native L4/L2=1의 N4 B1..B10 전 endpoint로 C4 D64 최대값을 고정해야 한다.
SH2가 lifelong main-cell-3 및 L4 one-shot38997의 봉인 commit20개를 확인했다.
두 경로 모두 B1/B5/B10만 저장됐고 **B2/3/4/6/7/8/9는 checkpoint=null**, 복원 가능한 weight delta journal도 없다.
Native target 또는 entry/endpoint hash만으로 누락 weight를 만들 수 없다.
기존 CP bytes 검증은 SH2 과거 fullSHA+현재 stat 결속 재사용이며 이번 S4의 신규 tensor검산/전송이 아니다.
일부 endpoint 최대값/따뜻한 W50 proxy/0 budget으로 치환하지 않았고 baseline editing rerun도0이다.
따라서 fixed b는 미정이고 **BG scientific job0, firstB100 미실행**이다.

## 완료된 독립 준비

|항목|관측|
|---|---|
|원문/참조|14문서, PROTOCOL1269줄, dispatch/native/fitter full-read; 원문 SHA 보존|
|C4 획득|정확2gzip 359,779,975B; train356317/validation45576행 전체 EOF·CRC·fullSHA|
|Reference|S64+Dev128+Reserve320+Report256=768, 실제 int64[768,257]|
|Token|자연256+BOS1; target[129,257), logits[128,256); decode-reencode0|
|Overlap|Wiki128+first1000 R/P/N 입력13128; 선택문서294528쌍 검사, 중복0|
|독립 CPU 테스트|39 PASS; native map/FD·VJP/C2/mass/ball/trust/candidate/guard|
|Teacher 준비|job47592 held검사→release; 마지막 PENDING; 정상기동/완료 아직 미관측|

Reference identity `f5791dd3c986261a252796bd5d7293ece46339d261609449dcfe674970bbfde0`. Member root `0c6aa4e2ddb350c61999580fe3efabaae880ec8e17aa208a8f7d85c026a6f1aa`.
모든 역할과 window는 첫 model loss 전에 CPU 봉인했다. Sampler는 답/score/future subject list를 받지 않는다.
mom2 원문, deferred MMLU/Audit 및 미래 stream overlap은 미검사이며 overlap-free로 주장하지 않는다.
PSL은 composition만 계산하고 선택에는 관여하지 않는다. 전체 C4 균등표본·semantic 중복부재도 주장하지 않는다.

## Source·구현 검증 경계

Teacher execution source `8b8f2a678128156852f4941dbaa3cb9023f18559`, tree `071bba51b57efd84499e42d0b08c4ce180392a91`;
archive `7630c27eebf800c21fdbbff8de0995fcc66c6d3c919793de7ef1a77da9fe83d3`, lock `174d3c3b738444082b59caf21d7bdd655fdc321b69c64a98d7835ce26accf968`.
Builder는 commit 전 실제 파일 SHA를 결속했으며 그 당시 mainHEAD에 신규 builder가 있었다고 쓰지 않았다.
Frozen source/model/tokenizer/4shards/index/dependencies/kernel/reference 포함1772 members를 CPU fullSHA 검산했다.
실제 teacher는 같은 lock을 model load 전에 다시 검증한다. FP32/eager/V128256/MB1, 8docs×24 mmap shards,
KL(p0||pW)는 vocab합→128position평균→document평균이다. Reserve/Report teacher는 생성하지 않는다.
동일 logp의 selfKL=0은 독립 W0 재실행 parity가 아니다.

`native_map`은 고정 native 직접 RHS solve와 RA 경로의 FP32 차이를 구분한다. Cholesky/inverse/SPD 대체0.
`correction`은 전체 D64에서 정한 단일 slope, 문서 mass, own anchor ball, 실제 executable trust,
RAW1/CORR1/CORR.5/CORR.25와 parent fallback을 CPU fixture로 검사했다.
**BG model adapter/persistent controller는 미완성·미검증이다.** 실제 Llama post-write gradient,
all-token model parity, branch restore, history exactlyonce/zero-write, resume, G0는 검증하지 않았다.
Toy CPU 결과를 model-level PASS로 승격하지 않는다. 기준 b/b_num·실제 BG 수치 tolerance도 아직 lock하지 않았다.
Red가 찾은 W0/server/calibration 우회·OOV/ID·source/tokenizer seal·환경버전 guard는 수정 후 CPU 검산했다.
새로운 scientific 구현·수리·제출은 다음 명시적 recall 이후이며 누락 calibration 권한/자산 해결도 필요하다.

## 비용·자원

실제 C4 CPU build 21.628s. 신규 editing/calibration forward0.
Teacher 마지막 관측 allocated/RunTime0; 완료 GPU시간/peakmemory/NLL/실제 teacher bytes는 **NOT_OBSERVED**.
계획 teacher tensor 12,608,077,824B(약11.7422GiB), 예상0.25–2GPUh/요청wall2h는 측정값이 아니다.
제출 시 project active/admitted0+teacher1≤cap2, 1GPU/8CPU/60416M, source별도/exportNONE.
당시 물리GPU8개는 다른 사용자에게 할당되어 있었다. 다른 job 선점/취소/변경0.
작업 디스크여유 255904006144B, 계획60GiB는 독점 filesystem 예약이 아니다.
Scheduler 기본 Requeue=1은 이 새 teacher47592만 Requeue=0으로 제출제어 정정했다.
Frozen source/과학설정/output·기존 job은 바꾸지 않았다. 자동 retry/agent callback/monitoring0.

## 인계와 재현

Local root `/data/janghj/ODE-edit/local/bg1-c4-ours-first/20260915-v1/attempt-v1`. Resume manifest `/data/janghj/ODE-edit/local/bg1-c4-ours-first/20260915-v1/attempt-v1/resume-manifest.json`, SHA `44bba7e96e2bb163ee812282a538fc49152ed1d6e89f5624e2c17f898cc9d99d`.
Teacher output 예상 `/data/janghj/ODE-edit/local/bg1-c4-ours-first/20260915-v1/attempt-v1/teacher-output-v1`; 종료/manifest 존재를 추가 관측하지 않는다.
원본 보고서/코드/모델/P/stats/dirty/중지 ORBODE 및 다른 paused task는 그대로 보존한다.
Reference/teacher/raw text/tokens/log는 local-only, NO_BROADCAST_NOT_REQUIRED. PNG 생성0(성능결과 없음).

재현 코드: `project/run_scripts/bg_tw_reference/README.md`; builder의 원 CLI 입력은 source-manifest 및 full-read에 결속.
`preparation_report --worktree <fresh-cleanW> --attempt <A> --control <A/new-report-control>`로 기존 입력과 새 출력 namespace에서 재생성한다.
경로/기록시각 필드가 달라질 수 있으므로 report 전체 byte-identical 재현은 주장하지 않는다.
CPU tests: `PYTHONPATH=<deps-transformers-4.44.2>:<W> /data/janghj/EasyEdit/.venv/bin/python -m unittest discover -s project/run_scripts/bg_tw_reference -t . -v`.
사용자 호출 전 polling/terminal대기/자동 결과분석/후속제출은 없으며 준비 job은 스케줄러에서 그대로 진행한다.
