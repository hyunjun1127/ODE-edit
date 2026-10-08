# server1 실행 연결

이 폴더는 server1가 소유하는 EasyEdit 자산 연결과 runner 경로다. `assignment.json`의 모델·방법을 담당한다.

- `prepare.py`는 기존 자산의 존재/선택적 SHA를 확인한다. GPU runner 완료를 뜻하지 않는다.
- 담당자가 `run.py`, 제출 스크립트와 재개 검증을 이 폴더에 구현한다. 공통 알고리즘·평가기는 `official/` 내부 모듈만 import한다.
- `assets.example.json`을 참고해 `assets.local.json` 또는 ignored local manifest를 만든다. EasyEdit는 자산 경로로만 사용한다.
- 데이터, C0, P, weights, raw 출력은 원래 자산 경로/ignored local에 두고 복사하거나 Git에 넣지 않는다.
- 실행 source는 `main`의 commit과 `official/` tree SHA로 고정한다. 기존 실행을 취소하거나 코드를 바꾸지 않는다.
- 공유 코드 수정이 필요하면 원인을 보고하고 공통 수정으로 통합한다. 서버별 수치 구현 fork를 만들지 않는다.

## 실제 실행 연결

`assets.py`는 모델·tokenizer·CF/zsRE·층4..8의 원 NPZ sum/count·generation
reference/NLTK를 읽기 전용으로 결속한다. `prepare_runtime.py`는 FT/MEMIT/MEMIT_FE의
CF/zsRE 여섯 config와 아직 존재하지 않는 qualification/W0 READY 경로를 생성한다.
모델 또는 통계 다운로드/재계산은 없다. 큰 자산과 checkpoint/raw는 ignored local에만 둔다.

`native.py`는 `official.baselines.registry`의 실제 구현/parser/requests/call_options를
사용한다. FT는 native prompt_last·단일층21·내부batch1·norm_constraint=False,
MEMIT은 beta_hse=0/save_weights=False, MEMIT_FE는 native clamp4를 유지한다.
이 세 방법에는 native history H가 없으며 반환 weights_copy를 H로 해석하지 않는다.

`run.py`의 GPU qualification은 방법별 실제 BS100 B1–B3 연속 실행과
B2 종료→새 프로세스의 pinned checkpoint 복원→B3를 비교한다. selected FP32 W,
context/native cursor/RNG 및 동일 first300 factual raw가 정확히 일치해야 READY가 된다.
연속 B3에서는 기존 first300 canonical raw를 보존한 채 원본 CF evaluator의
독립 forward와 대조한다. 허용차·원본 SHA·matched-subset 범위는 PLAN에 봉인한다.
별도 first4/canonical 재평가는 없고 full2k parity를 주장하지 않는다.
CPU fixture는 이 GPU 검증을 대신하지 않는다. qualification 자체는 추가 2k science chain이나
생성 평가가 아니며, 다른 trajectory와 독립적인 검증용 cold model이다.

CF factual schedule은 정본 `official/hparams/contract.json`의 all_seen
W0/W5/W10/W15/W20이다. 실제로 측정하지 않은 current/pre/current/post는 만들지 않는다.
CF scores는 strict NLL request-macro의 `official/*` namespace에 기록하고 실제 prompt/token
분모의 nine-field diagnostics와 구분한다. zsRE는 teacher-forced request-macro,
같은 모델 W0 token prediction agreement와 `Specificity_loc_ans`를 별도로 기록한다.

CF 생성은 native CAKE case-batched KV/global-RNG/topk5/padded total100/noEOS profile로
모델 W0 한 번, 각 CF chain W20 한 번만 실행한다. 한 텍스트를 entropy/TF-IDF에 공유한다.
중간 생성, W20 full2000을 current100으로 복사, 옛 파생-profile raw의 relabel은 없다.

## checkpoint와 제출

공식 사용자 계약에 따라 W0부터 매 batch의 편집·예정 평가 완료 뒤 selected FP32 W,
native context/cursor/RNG/identity를 최신 1개 checkpoint로 저장한다. W20은 보존한다.
W20 generation 실패는 B19 checkpoint를 유지한다. 재개는 같은 source/config/input의
명시적인 `--resume`만 허용하며 완료된 W20의 중복 재개는 control과 runtime 모두 거절한다.
다른 작업의 과거 noCP/source/raw/job은 변경하지 않는다.

`submit.py`는 실제 main commit/official tree를 archive로 봉인하고, 전체 owner의 실제 GPU
allocation과 admitted DAG 폭을 검사한 뒤 held 등록→exact owner/argv/script/resource/input/
source/dependency 검사→release를 수행한다. 정상 dependency/resource PENDING은 허용한다.
정확한 technical qualification/W0/smoke 의존성은 afterok, science 실행 순서와 GPU0 collector는
afterany이다. 기존 job 변경·취소·자동 재제출·장기 polling은 없다.
새 DAG만 `--kill-on-invalid-dep=yes`를 사용해 기술 prerequisite 실패를 영구 PENDING으로
남기지 않는다. 취소된 downstream도 실패/미관측으로 집계하며 과학 성공으로 표시하지 않는다.

전체 DAG는 qualification3 → 공유 cold W0 → CF3 → zsRE 1batch smoke → zsRE3 → GPU0 collector다.
처음 qualification 3개는 현재 owner의 기존 GPU frontier 뒤로 연결한다.
이는 과학 여섯 행 외에 명시 승인된 qualification/W0/smoke 작업이 있다는 뜻이며,
held 검사에서 미래 GPU/native parity 또는 온라인 PASS를 주장하지 않는다.

W&B는 `official.tracking` 단일 배포를 읽기 전용으로 사용한다. 실제 Slurm job 번호가 이름과
config에 결속되며 각 invocation은 고유 UUID/spool/immutable identity를 가진다.
SDK 접수, bounded remote readback, CPU audit와 scientific completion은 별개다.
자격증명·코드·console·text/token/raw/tensor는 업로드하지 않는다.
