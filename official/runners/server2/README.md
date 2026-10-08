# server2 실행 연결

## 2026-10-09 USER: zsRE 전용 지표 및 여섯 baseline

`execution --zsre-only` → `submit --stage zsre_pipeline`은 CF와 별도 새 source/attempt다.
MEMIT 첫 job 안에서 같은 모델 zsRE W0 prediction reference를 한 번 만들고
native B100 한 번 smoke를 수행한다. CPU reducer가 native/source/factual/checkpoint 증거를
검산한 뒤에만 새 Python/model의 cold MEMIT 20-batch chain을 시작한다.
나머지 FT/AlphaEdit/BLUE/FE/SPHERE는 producer 종료와 실제 smoke receipt를 검산한다.
GPU file polling/추가 자동 재시도/과학 성능 gate는 없다. 사전 계획은 actual PASS가 아니다.

기존 CF frontier 전체 뒤에 새 producer를 직렬화하고 이후 최대4 lane을 허용한다.
Server2 direct USER cap4만 적용하며 기존 CF jobs/source/archive는 불변이다.
zsRE는 FLU/CON을 실행하거나 CF generation config/metric을 기록하지 않는다.
W0 agreement와 loc_ans 정확도는 서로 다른 지표이며 W0 reference의 evaluation을 재사용한다.
checkpoint는 공식 latest1/finalW20 보존 계약이다.

## 2026-10-09 직접 USER: CF six-arm checkpoint-only 준비

`checkpoint_profile.py`의 명시 opt-in만 FT/MEMIT/AlphaEdit/BLUE/FE/SPHERE
CF first2000 BS100×20, project cap3, FLU/CON W0/W20 모두 미실행을 선택한다.
기존 profile/default와 frozen jobs는 바꾸지 않는다. factual 평가/native 수학은 유지한다.
`execution --checkpoint-only` → `submit --stage cf_checkpoint` 경로다.

각 GPU allocation의 `checkpoint_pipeline`은 실제 B3 대 B2→B3 qualification을
별도 모델 process에서 확인한 뒤, fresh cold model의 20-batch chain을 실행한다.
첫 FT job은 편집 전 shared factual W0와 독립 원본 scorer proof를 한 번 만든다.
나머지는 FT 종료 후 READY를 검증하고 3-lane DAG로 진행한다. GPU file polling은 없다.
따라서 첫 FT 동안에는 GPU1, 그 이후 최대GPU3이며 6개 scientific arm과 CPU collector다.
qualification의 추가 B3 replay는 원 parent 기술 검증이며 실제 비용을 별도 기록한다.

배치 W0..W20의 latest1 official checkpoint를 저장하며 최종 W20을 보존한다.
평가 미측정은 `DEFERRED_NOT_MEASURED`이지 점수0/평가완료가 아니다.
후속 2K checkpoint 평가가 예정된 consumer이므로 archive/delete gate는 닫혀 있다.
재개용 native W/H/RNG/context/cursor와 원 base/runtime/input/source binding을 유지한다.

공통 `official.tracking`은 읽기 전용이다. 이 caller는 실제 미실행을
`generation_schedule=DEFERRED_CHECKPOINT_EVALUATION`로 기록하며,
이를 승인하지 않는 옛 schema에서는 준비/시작을 차단한다. 켜진 schedule로 위장하거나
logger를 복제하지 않는다. 공통 schedule 게시 전에는 **미제출**이며 CPU fixture는
실제 native/resume/GPU/온라인 PASS가 아니다.

이 폴더는 server2가 소유하는 EasyEdit 자산 연결과 runner 경로다. `assignment.json`의 모델·방법을 담당한다.

- `prepare.py`는 기존 자산의 존재/선택적 SHA를 확인한다. GPU runner 완료를 뜻하지 않는다.
- 담당자가 `run.py`, 제출 스크립트와 재개 검증을 이 폴더에 구현한다. 공통 알고리즘·평가기는 `official/` 내부 모듈만 import한다.
- `assets.example.json`을 참고해 `assets.local.json` 또는 ignored local manifest를 만든다. EasyEdit는 자산 경로로만 사용한다.
- 데이터, C0, P, weights, raw 출력은 원래 자산 경로/ignored local에 두고 복사하거나 Git에 넣지 않는다.
- 실행 source는 `main`의 commit과 `official/` tree SHA로 고정한다. 기존 실행을 취소하거나 코드를 바꾸지 않는다.
- 공유 코드 수정이 필요하면 원인을 보고하고 공통 수정으로 통합한다. 서버별 수치 구현 fork를 만들지 않는다.

## Server2 실제 자산 결속

`python -m official.runners.server2.assets`는 기존 Server2 현물을 읽어
`/mnt/raid5/janghj/ODE-edit/local/official-baselines-server2/20261008-r1/preparation-r1/assets.json`을
create-once로 준비한다. `assets.prepare(out)`의 결과를 runner가 사용하고,
실행 직전에 `assets.verify(manifest)`로 source/input/runtime 및 봉인된 자산 stat를 다시 확인한다.
EasyEdit 알고리즘 코드 import·모델 load·GPU·다운로드·C0/P 생성은 하지 않는다.

- GPT-J revision은 `47e169305d2e8376be1d31e765533382721b2cc1`이며 tokenizer 여섯 파일을 새 SHA로 대조한다.
- CF는 기존 fixed10k prefix, zsRE는 `EasyEdit/data/zsre/zsre_mend_eval.json`을 사용한다.
  `official/hparams/*-stream.lock.json`과 새 정규화 first2k stream/20개 경계를 정확히 대조한다.
  subject-last/offset/BOS/token 길이 CPU 검산은 두 stream 모두 수행한다.
- C0는 L3..8의 raw FP32 `mom2_sum`이다. 각 NPZ header와 scalar count **54924275**,
  sample_size **100000**을 새로 읽는다. sample_size를 count로 대신하지 않는다.
  1 GiB 행렬은 다시 load하지 않고 prior finite/schema/fullSHA와 unchanged stat를 재사용한다.
- P는 full six `[6,16384,16384]`이며 physical L3..8→slot0..5, BLUE의 L3/L8→slot0/5다.
  prior finite/six-slot/fullSHA 증거와 현재 unchanged stat를 결속한다. threshold .02는 공개
  hparams 및 역사 task 계약의 값이다. 원 P construction receipt는 없으므로 새 eigen 검증 또는
  spectral 생성 provenance가 입증됐다고 주장하지 않는다. 재계산으로 이 한계를 숨기지 않는다.
- FLU/CON의 기존 reference 세 파일, 수신 receipt, manifest/READY, NLTK resource와 scoring
  versions를 결속한다. 새 비교의 W0는 새 실제 모델 관측이며 기존 W20-only raw 재사용이 아니다.

Manifest의 `ASSETS_BOUND_CPU_ONLY`는 자산 준비 상태다. 실제 native smoke, BLUE output/layout,
연속 B3와 B2→B3 resume 및 평가 parity는 별도 GPU 증거가 필요하다.

## 신규 CF 독립 원본 reference 결속

`oracle.plan(manifest, records[:4])`은 shared `cf_native_reference`/원 source/lock과
ordered first4, 고정 tolerance를 actual 관측 전에 봉인한다. 새 `execution.prepare`가
manifest에 이 계획을 넣는다. 새 CF W0는 call-local `use_cache=False` 및 실제 물리 state를
canonical first4 관측 전에 기록하고, 수정 없는 원 `test_batch_prediction`의 **독립 forward**와
비교한다. 실패/CPU fixture/미관측은 raw를 보존하고 READY를 차단한다. GPU smoke PASS만
새 complete W0 READY에 proof/canonical/state member SHA로 결속한다. 일반 full2k parity와는
다른 engineering scope이며 추가 fit/edit/generation은 없다.

이 연결은 미래 immutable source만 적용한다. 기존18e7fbd qualification source/archive는
변경하지 않는다. 과거 native/resume producer proof와 새 independent scorer proof를 각각
검산하고 consumer compatibility를 별도로 기록한다. old raw/source/identity를 새 source로
relabel하거나 actual receipt 없는 plan을 GPU PASS로 사용하지 않는다.

`execution.prepare(..., qualification_producer_attempt=<exact old attempt>)` 또는 CLI
`--qualification-producer-attempt`는 `qualification_input.plan`의 native source/input
compatibility를 봉인한다. 입력은 기존18e7fbd registration-qualification-r1으로 한정한다.
179개 계산 source 및15개 핵심 함수 AST/모델·token·stream·C0/P·runtime·reference·12config를
대조하고 old/new 전체 source와 assets manifest IDs를 별도로 남긴다. 원 display rounding은
과거 원뜻으로 검산한다. 이 native compatibility는 old checkpoint를 새 source로 재개할
권한이 아니다. 본 chain은 새 cold W0/checkpoint identity에서 시작한다.

`submit --stage cf`는 실제 여섯 native/resume proof와 완료된 CPU collector/atomic terminal이
있어야 등록한다. 새 W0 oracle actual PASS를 그 W0를 제출하기 전에 요구하는 순환 gate는 없다.
대신 사전 봉인한 oracle PLAN이 필요하고, W0 actual PASS 이후만 READY/후속이 열리며 reader와
collector가 member SHA를 재검산한다. 원 qualification receipt가 없으면 INPUT_PENDING이며
CPU fixture·label·단순 Slurm terminal로 성공을 만들지 않는다. 새 monitor/retry는 없다.
