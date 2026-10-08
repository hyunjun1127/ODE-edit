# server2 실행 연결

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
