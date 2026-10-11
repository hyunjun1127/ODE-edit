# PRICE 공식 구현과 baseline 실험

**2026-10-08 기준 배포 경로는 저장소 최상위 `official/`이다.** CAKE 저장소처럼
방법 구현, hparams, 평가 코드와 실행 연결을 한 폴더 안에서 관리한다. 새 baseline
실험은 `main`에 게시된 이 폴더의 코드·설정과 정확한 commit/tree SHA를 사용한다.
각 서버의 EasyEdit 설치는 데이터·모델·C0·projector 자산 경로로 활용한다.

**FE의 현재 실행 기준(2026-10-11 사용자 정정):** [`runners/fe_author_history.py`](runners/fe_author_history.py)의
수정 EasyEdit 구현 + 원저자 hparams + persistent history다. FP32 모델로 각 배치 진입 시 현재 100건의 z만
계산하며 미래 edit의 W0 target을 선계산하지 않는다. `fe_original.py`의 author W0-fixed/BF16 경로는
이력·참고 실험이며 main baseline이 아니다. [붕괴 조사와 검증 범위](../experiment-reports/global/fe-collapse-repair-20261011/report-ko.md).

초기(2026-10-08) 실험 범위는 **baseline 6개 × 모델 3개 × 데이터셋 2개 = 36개 행**이다.
PRICE(ours) 구현과 hparams도 배포하지만 이번 실행 목록에서 제외한다.
이 문서의 준비 완료 표시는 CPU/source 확인이며 실제 모델 실험의 완료를 뜻하지 않는다.

**2026-10-09 ours 설정 분리:** [`hparams/ours/`](hparams/ours/README.md)에 모델별
`writer.json`·`price.json`과 arm override를 둔다. `official.ours.config.resolve()`가
검증한 불변 설정 하나를 계산 경로 전체에 전달한다. 기본값은 유지하며 Qwen arm은 파일만
준비했다. 이전 `hparams/PRICE/`는 deprecated이고 baseline 본 실험 설정은 그대로 사용한다.

## 폴더와 실행 책임

```text
official/
  ours/                  PRICE ridge/CAP075 핵심 구현, M1 helper
  baselines/             EasyEdit·SPHERE·BLUE의 고정된 구현과 공통 registry
  hparams/               방법별 모델 설정, 원본 설정, source/stream/generation lock
  evaluation/            공통 지표 집계와 server1 native FLU/CON 구현
  experiments/           실행 목록·CF/zsRE stream 준비, 공통 checkpoint
  runners/server1/       server1 EasyEdit 자산 연결과 실행 runner
  runners/server2/       server2 EasyEdit 자산 연결과 실행 runner
  runners/server3/       server3 EasyEdit 자산 연결과 실행 runner
  runners/server4/       server4 EasyEdit 자산 연결과 실행 runner
  tests/                 설정·데이터·source·checkpoint CPU 검증
  tools/verify.py         배포 파일 SHA와 외부 task 코드 import 검사
  SOURCES.json            원본 commit/경로/SHA, 배포 SHA와 변경 내역
```

서버별 `prepare.py`는 자산 점검 진입점이다. 담당자는 자신의 `runners/serverN/`에
실제 `run.py`와 제출·재개 연결을 작성한다. 공통 baseline/PRICE/평가 코드의 수치
변경은 공유 코드로 통합하고 서버별 복사본을 만들지 않는다. 기존 실행과 frozen source는
보존한다. main에 새 구현을 게시했다고 이미 실행 중인 job의 source가 바뀌지는 않는다.

| 담당 | 모델 | 방법 | 본 실험 행 |
| --- | --- | --- | ---: |
| server1 | Llama3 | FT, MEMIT, MEMIT-FE | 6 |
| server2 | GPT-J | 6개 전부 | 12 |
| server3 | Qwen2.5 | 6개 전부, BLUE 격자·강도 대조 | 12 + 보조 5 |
| server4 | Llama3 | AlphaEdit, AlphaEdit-BLUE, AlphaEdit+SPHERE | 6 |

## 고정 실험 범위

| 축 | 값 |
| --- | --- |
| 모델 | Meta-Llama-3-8B-Instruct, Qwen2.5-7B-Instruct, GPT-J-6B |
| 데이터 | CounterFact(CF), zsRE, 각각 기존 파일 순서의 첫 2,000건 |
| 방법 | FT, MEMIT, AlphaEdit, AlphaEdit-BLUE, FE(MEMIT-FE), AlphaEdit+SPHERE |
| 순차 편집 | 20 batch × 100건, 동일 순서·batch 경계, edit seed 0 |
| 정밀도 | baseline FP32 모델·가중치; FE ridge solve FP64, TF32 off |
| CF 지표 | Efficacy, Generalization, Specificity, Score, Fluency, Consistency |
| zsRE 지표 | Efficacy, Generalization, Specificity=loc_ans 정답 정확도; W0 예측 보존율 별도 |
| 누적 평가 | W0, 0.5K, 1K, 1.5K, 2K; 해당 시점까지 전체 편집 요청 |
| 생성 평가 | CF W0는 모델당 1회, W20은 실행당 2,000건 전체 1회 |
| 본문 제외 | ours 실행, GPT2-XL, CAKE, PRUNE, RECT |

Qwen BLUE CF L2 `{1,10,95}` 3개와 clamp 0.75 대조 2개를 별도 기록한다.
선택된 격자 실행을 본문 Qwen BLUE CF 행으로 재사용하므로 **41개 논리 행,
중복 실행을 제외한 편집 chain 40개**다. W0 6개 모델·데이터셋 평가와 smoke/resume
검증은 별도다. GPT2-XL 기존 결과는 부록 기록으로 보존한다. PRUNE·RECT의 세 모델
CF 2K는 본 실험 이후 부록 단계로 두고 이번 제출 목록에는 넣지 않는다.

## Baseline hparams와 출처

EasyEdit에 공개 설정이 있으면 우선 사용한다. Llama3 MEMIT은 SPHERE 저장소,
AlphaEdit-BLUE는 BLUE 저장소를 사용한다. 원본 **17개 파일**과 SHA를
[`sources.lock.json`](hparams/sources.lock.json)에 보존했다. Qwen BLUE는 공개 설정이
없어 BLUE schema에 Qwen AlphaEdit 값을 대응한 자체 설정임을 명시한다.

| 방법 | Llama3 | GPT-J | Qwen2.5 |
| --- | --- | --- | --- |
| FT | L21, lr 5e-4, 25 epoch | L21, lr 5e-4, 25 epoch | L27, lr 5e-4, 25 epoch |
| MEMIT | L4–8, clamp .75, lr .1 | L3–8, clamp .75, lr .5 | L4–8, clamp 4, lr .5, decay .001 |
| AlphaEdit | L4–8, clamp .75, lr .1, L2 1 | L3–8, clamp .75, lr .5, L2 10 | L4–8, clamp 4, lr .5, L2 1 |
| AlphaEdit-BLUE | [4,8], clamp .75, L2 1 | [3,8], clamp .75, L2 95 | [4,8], clamp 4, L2 격자 |
| FE (author hparams + history) | L4–8, clamp .75, 35 steps, lr .1, decay .5 | L3–8, clamp .75, 25 steps, lr .5, decay .5 | L4–8, clamp 1, 35 steps, lr .5, decay .001 |
| AlphaEdit+SPHERE | AlphaEdit 설정, L2 1 | AlphaEdit 설정, L2 10 | AlphaEdit 설정, L2 1 |

- FE는 Liu et al.의 [From Backward Spreading to Forward Replay](https://arxiv.org/abs/2605.00358)이며,
  현재 baseline은 수정 EasyEdit FE 구현에 commit `478134df`의 모델별 hparams와 누적 history를 적용한다.
  `hparams/MEMIT_FE_HISTORY_AUTHOR/`에 원본 SHA를 기록하며, W0 전체 target 선계산은 사용하지 않는다.
- SPHERE는 AlphaEdit 기반이며 `cumulative_ratio=0.5`(η), `suppression_strength=0.5`(α)다.
- FT는 `prompt_last`, 단일 층, norm 제약 없음이다. 공개 **내부 batch_size=1**을 유지한다.
  외부 편집 batch 100과 구분하며, native optimizer는 외부 batch마다 새로 만든다.
  따라서 `num_steps=25`는 100건 합계 25개 optimizer step이라는 뜻이 아니다.
- 일부 YAML의 `model_name`은 base 모델처럼 보이는 로컬 경로다. 배포 설정은 Llama/Qwen
  모두 명시한 Instruct ID와 revision으로 연결하고 이 override를 [`profiles.json`](hparams/profiles.json)에 기록한다.
- zsRE에도 같은 hparams를 사용한다. 이 pin의 선택 파일에는 zsRE 전용 설정을 사용하지 않는다.
- Qwen BLUE는 CF W20의 반올림 전 Score 최대값을 선택한다. 동률은 Specificity,
  Efficacy, Generalization, 작은 L2 순이다. 선택값을 zsRE에 그대로 적용한다.
  본문에 **CF-tuned 자체 설정**으로 표시하고 격자 전부를 공개한다. 별도 held-out 선택은 아니다.
- 강도 대조는 AlphaEdit 및 선택된 BLUE 설정의 clamp만 .75로 바꾼다.
  결과를 본 뒤 공개 설정 행을 바꾸지 않는다.

원본 pin: [EasyEdit 4c109870](https://github.com/zjunlp/EasyEdit/tree/4c109870955a4522ac3d7cf10ad00f34de8e4f0d),
[SPHERE 0c28c000](https://github.com/PlusLabNLP/SPHERE/tree/0c28c000a1482e73573928f63a66e0b1b88ac949),
[BLUE 311b076a](https://github.com/xpq-tech/BLUE/tree/311b076a92e4ed0f14f5c8b4909732da781bc5f7).

## FE writer 비교 진단

`runners/fe_author_history.py`의 기본 backend는 `EASYEDIT_HISTORY`다.
`writer_backend: AUTHOR_WRITER_CURRENT_BATCH_Z`는 저자 FE `478134df`의
`batch_edit` writer·history에 우리 `compute_z`를 연결하는 별도 진단이다.
원본 writer SHA, 저자 source 전체 manifest, hparams, 고정 context와 모델 revision을 검증한다.
z는 각 배치 진입 모델에서 현재100건만 계산하며 미래 target cache는 허용하지 않는다.
저자 FP64 Gram→FP32 history 누적과 가중치 대입을 유지하고, 배치 실패 시 W/H를 복원한다.
`observe_batches`는 z/K/현재 출력과 case ID를 기록한다. `retain_milestones`는
500·1,000·1,500·2,000 edits checkpoint를 rolling 저장과 독립적으로 보존한다.
GPT-J C0 수치 보정은 `covariance_policy: project_roundoff`로 명시해야 한다.
[수치 결함·진단 상태](../experiment-reports/global/fe-collapse-repair-20261011/report-ko.md).

## FLU/CON: 기존 자산 유지, 현재 server1 코드 기준

**`attribute_snippets.json`, `idf.npy`, `tfidf_vocab.json`은 기존 자산과 SHA를 그대로
사용한다. 개선 대상은 실행 코드다.** 현재 server1 job **61519 / 61520 / 61521**의
source `adb244e6f9c86b54f73bd6d8fb833b338f470ded`를 실제 execution lock과 대조했고,
그 생성·평가 코드를 `evaluation/generation/`에 포함했다. 기준은
[`generation.lock.json`](hparams/generation.lock.json)이다.

runner 작성 시 이 실행본을 읽고 **CAKE와 BLUE의 `util/generate.py` 및
`experiments/py/eval_utils_counterfact.py`의 연산 방식을 함께 참고한다.** 다음을 공통으로 적용한다.

- 한 case의 `generation_prompts`를 padded batch로 묶고, call-local KV cache와
  incremental decoding을 사용한다. 매 토큰마다 전체 prefix를 다시 계산하는 이전 경로를 기본으로 쓰지 않는다.
- top-k 5, prompt 포함 padded-array width 100, prompt당 생성 1개, EOS 조기 종료 없음.
  endpoint 시작에 seed **20261007**을 한 번 설정하고 case 순서대로 RNG를 소비한다.
- CAKE의 누적 cache+query attention mask와 decode 규칙을 따른다. BLUE의 query-only
  mask 및 `skip_special_tokens` 차이를 문서화하며 두 구현을 byte-exact하다고 부르지 않는다.
- 생성은 한 번 수행해 FLU와 CON이 같은 원문을 공유한다. vocab/IDF/vectorizer는 재사용하며
  생성문에 TF-IDF를 다시 fit하지 않는다. CPU 재집계에 GPU 생성을 반복하지 않는다.
- Fluency는 NLTK word 2·3-gram entropy의 `H2/3 + 2H3/3`(bits),
  Consistency는 생성문과 relation/new-target의 전체 reference snippets 간 고정 TF-IDF cosine이다.
- 원문·token·요청별 결과, profile/seed/source/순서/weight identity, 실제 시간·forward/token 수를
  ignored local에 저장한다. 이전 생성 profile의 점수·raw를 새 profile 결과로 재표시하지 않는다.
- server1의 현재 실행은 W20-only다. **이번 새 비교 실험의 W0는 모델당 별도 1회**로
  공유하며 중간 W5/W10/W15 생성은 하지 않는다. 실제 속도 향상 수치는 아직 미측정이다.

## 데이터·평가기 고정

- [x] CF 2K 순서는 기존 fixed-10K prefix와 동일: ordered ID SHA
  `0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4`.
- [x] CF·zsRE 파일 SHA, 정규화 stream SHA와 20개 batch 경계를 각각
  [`cf-stream.lock.json`](hparams/cf-stream.lock.json), [`zsre-stream.lock.json`](hparams/zsre-stream.lock.json)에 기록.
- [x] zsRE는 AlphaEdit 원본처럼 `answers[0]`을 target으로 사용한다. `alt`로 임의 변경하지 않는다.
  첫 2K에서 모든 subject가 prompt에 정확히 한 번 등장한다. 원본 순서를 유지하고 요청을 제외하지 않는다.
- [x] tokenizer CPU 점검: CF lookup 0은 Llama 0건, Qwen 7건(B2·3·7·10·17·18·20),
  GPT-J 6건(B3·5·7·10·17·20). zsRE는 세 모델 모두 0건. offset 검산 불일치 0건.
- [ ] 서버마다 모델 revision, tokenizer 파일 SHA/BOS 동작, C0 Wikipedia 10만 표본과
  projector threshold .02의 실제 provenance를 검증한다. 파일 존재만으로 완료 처리하지 않는다.
- [ ] Qwen lookup 0 요청의 baseline 성공/실패를 개별 기록한다.
- [ ] zsRE neighborhood의 token별 W0 예측을 저장한다.
- [ ] Llama AlphaEdit CF 2K의 원본 평가기와 공통 평가기를 실제 가중치에서 비교한다.

CF 성공은 E/G에서 `new NLL < true NLL`, S에서 `true NLL < new NLL`이며 동률은 실패다.
요청 내 prompt 평균을 먼저 구하고 요청 간 macro 평균을 낸 뒤 E/G/S의 조화평균을 Score로 쓴다.
원본 AlphaEdit summarize는 소수점 둘째 자리로 반올림한 E/G/S의 조화평균을 표시하므로
원시 Score와 원본 표시용 Score를 함께 보존한다. neighbor의 new/true NLL도 별도 보존한다.

zsRE E/G는 teacher-forced token 정확도의 요청별 평균이다. Specificity(본표 Loc)는
**loc_ans 정답 token 정확도를 요청 안에서 평균한 뒤 요청 간 평균**한다.
`Specificity_loc_ans`는 같은 값의 호환 별칭이며, W0 예측 보존율은
`W0_prediction_agreement` 보조지표로 분리한다. 잘못된 W0 답의 보존은 Loc 성공이 아니다.
2026-10-09 사용자 정정 지시가 이전 W0-agreement 표 정의를 대체한다. frozen 원자료는
수정하지 않고 저장된 token correctness를 CPU 재집계하며, tokenization의 원본 논문
완전 재현을 이 정정만으로 주장하지 않는다.

## Checkpoint와 실행 순서

각 batch의 편집과 예정된 평가가 끝난 뒤 FP32 편집 가중치, 완료 batch, context templates,
Python/NumPy/Torch CPU·CUDA RNG, 평가 cursor와 요청 기록을 commit한다.
AlphaEdit·BLUE·SPHERE는 native `cache_c`도 저장하고 P는 고정 자산에서 다시 읽는다.
FT·MEMIT는 이 초기 runner 계약에서 batch 사이에 가중치만 이어가며 FT optimizer는 이어가지 않는다.
현재 FE는 FP32 가중치와 누적 FP32 history, RNG, 고정 context, 배치 cursor를 함께 저장한다.
W0 전체 target cache를 현재 FE의 checkpoint 의존성에 넣지 않는다. 수치 보정 C0 사용 시 별도 SHA에 결속한다.

`experiments/checkpoint.py`는 임시 저장 → hash/fsync → atomic pointer 갱신 → 이전
checkpoint 삭제 순서를 제공한다. W0부터 시작하고 최신 1개만 유지하며 W20은 보존한다.
code/config/stream/model/tokenizer/assets identity가 다르면 resume를 거부한다.
기존 task runner의 `noCP`는 이번 새 실행 계약에 적용하지 않는다.

- [x] 공통 checkpoint의 CPU B2 중단→B3 재개, RNG·가중치·history 일치와 쓰기 실패 보존 시험.
- [ ] **각 native 방법의 실제 runner**에서 B1–B3 연속 실행과 B2 재개 실행의 weight hash·지표 일치 확인.
- [ ] W20 생성이 실패하면 마지막 commit checkpoint에서 해당 batch를 재실행하는 경로 확인.
- [ ] 디스크 예산: dense cache_c만 Llama 약 4.1GB, Qwen 7.2GB, GPT-J 6.4GB(각 전체 층,
  decimal GB). 가중치·atomic 교체 중 두 checkpoint·동시 실행 수·최종 W20 보존분을 추가한다.

1. CPU source/hparams/stream/tokenizer/자산 검증과 서버별 runner 구현.
2. 방법별 native smoke, CF 평가기 대조, W0 및 resume 검증.
3. Llama/GPT-J CF 재현 확인 후 CF 18개 행과 Qwen 격자·대조.
4. 모델당 zsRE 1 batch smoke 후 zsRE 18개 행.
5. 최종 표·Qwen 격자 공개, 기존 GPT2-XL 및 후속 PRUNE/RECT 부록 정리.

재현 수치 차이 **1%p**를 원인 점검 기준으로 사전 기록한다. stream·평가 정의·hparams가
다르면 발표치 동등성의 증거로 해석하지 않는다. Qwen AlphaEdit 1K의 기존
99.2/94.4/69.8도 같은 조건인지 먼저 확인한다. 실제 제출은 서버별 최신 GPU cap과
기존 allocation을 반영한다. 이 CPU 준비 작업에서 신규 GPU job은 제출하지 않았다.

## 준비 명령

저장소 루트에서 실행한다. GPU 없이 가능한 검증이다.

```bash
python3 -m official.tools.verify
python3 -m official.experiments.prepare matrix --output local/official-baselines/configuration
python3 -m official.experiments.prepare stream --dataset cf --source /path/to/fixed-10k/counterfact.json --output local/official-baselines/streams
python3 -m official.experiments.prepare stream --dataset zsre --source /path/to/zsre_mend_eval.json --output local/official-baselines/streams
python3 -m official.runners.server1.prepare --easyedit-root /path/to/EasyEdit --model llama3 --reference-manifest /path/to/manifest.json --output local/official-baselines/assets.json
CUDA_VISIBLE_DEVICES='' python3 -m unittest discover -s official/tests -v
```

tokenizer 점검은 `official.experiments.prepare audit-tokenizer --stream ... --tokenizer
/path/to/local/snapshot --model qwen25 --output local/...json`으로 실행한다. 모델 forward는 없다.
`--hash-large`는 서버 자산 점검에서 실제 대형 파일 hash를 계산한다. 참조 자산을 다운로드·복사하지 않는다.

공통 native API는 `official.baselines.registry.implementation`, `hparams`, `requests`다.
server runner는 여기서 로드한 함수에 동일 모델을 순차 전달하고 native history/context를
관리한다. BLUE tensor/tuple 호환, C0/P layer 매핑, zsRE token-prefix 평가와 실제 resume의
GPU 검증은 runner 통합 항목이다. CPU import 성공을 실제 2K 실행 성공으로 표시하지 않는다.

## 공통 호환성 보완 (2026-10-09)

`SH1-GH-OFFICIAL-COMPAT-REVIEW-20261008-R1`에 따라 SPHERE의 Tensor/tuple/list
hidden container 호환과 누적 KV attention mask를 보완했다. FE의 별도 native context
generator와 loss/target-fit 수학은 바꾸지 않았다. 공식 generation은 GPT2/GPT-J에 더해
Llama/Qwen2의 full-context native cache 및 연속 position/cache 좌표를 지원한다.
top-k 5, case-batch, endpoint global RNG, padded 총길이 100, no-EOS, CAKE decode 정의는
그대로다. 지원하지 않는 cache/API는 typed failure이며 다른 생성 방법으로 대체하지 않는다.

공통 W&B API는 `official.tracking.init` → `Tracker.log` → `Tracker.finish`다.
각 서버는 [tracking 계약](tracking/README.md)의 동일 implementation을 읽기 전용으로
사용하고 서버별 logger를 복제하지 않는다. 공식 request-macro 점수는 기존 PRICE의
prompt-pair R/P/N scalar와 별도 namespace/schema로 기록한다. CF generation은 모델별
W0 한 번 및 chain별 W20 한 번이며 zsRE에 CF generation 지표를 넣지 않는다.

정확한 upstream bytes와 변경 후 SHA는 `SOURCES.json`에 결속한다. CPU fixture와
fake SDK 검산은 실제 pretrained/GPU qualification 또는 W&B online/readback PASS가
아니다. 기존 봉인 job/source는 hotpatch·취소·재시작하지 않는다.

## CAKE 3모델 × CF·zsRE 2K (2026-10-11)

`baselines/cake/`는 CAKE 저자 commit `0b378234862bd76c69f58404ef84c27d5f4bf9ef`의
층별 residual 배분·projected solve·batch 후 history 누적을 사용한다. target/key 계산은
같은 식의 `baselines/blue/` 호환 helper를 사용한다. 원본 함수 AST와 두 batch의
rectangular Linear writer/history를 CPU에서 대조한다.

`hparams/CAKE/{llama3,gptj,qwen25}.json`은 이 실행의 설정이다. Llama와 GPT-J는
저자 CAKE 설정 그대로이며 L2는 각각 10·30이다. Qwen2.5는 공식 AlphaEdit 편집
설정을 유지하고, 공개 known facts로 측정한 job 63264의 L4–L8 causal score와
저자의 temperature 0.1을 쓴다. 원본·측정 출처는 `hparams/CAKE/sources.json`에 있다.

server1 runner는 `python -m official.runners.server1.cake_run --config CONFIG --lock LOCK`이다.
EasyEdit 경로에서는 모델·stream·기존 nullspace projector 자산만 읽는다. 수치 코드는
main의 `official/`을 frozen archive로 묶어 실행한다. 공용 context는 Llama `cf14b857`,
GPT-J `b454d83e`, Qwen `5c01bc1a`다. 2K를 100건씩 순차 편집하고 native history를 켠다.
FP32/eager, seed 0, TF32 off이며, CF와 zsRE는 각각의 공식 평가기를 사용한다.
CF FLU/CON은 저장한 W20을 이용한 후속 평가로 미룬다.

각 실행은 500·1,000·1,500·2,000 edits마다 편집층 가중치, native history, RNG,
context, 원모델 revision과 source/config/asset identity를 저장한다. 출력의
`milestones/edit-0500`, `edit-1000`, `edit-1500`, `edit-2000`을 모두 보존하며,
`checkpoint/`의 최신 파일 교체로 삭제되지 않는다. 각 디렉터리는 공통
`official.experiments.checkpoint.load`로 읽을 수 있다. 편집하지 않은 가중치는
기록된 원모델 snapshot에서 복원한다. `--resume`은 같은 source/config의 최신
checkpoint에서 재개하며 이미 완료된 W20은 다시 편집하지 않는다.

등록기는 `python -m official.runners.server1.cake_submit prepare|register`다.
server1의 기존 작업 frontier 뒤에 CF·zsRE 두 lane을 붙이고, 각 lane은
Llama → GPT-J → Qwen 순서로 한 GPU씩 실행한다. 새 실행의 동시 GPU 수는 최대 2이며,
held 상태에서 실제 job/script/resource/dependency를 확인한 뒤 release한다.
