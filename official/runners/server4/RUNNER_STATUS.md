# server4 native runner 준비 상태

USER-OFFICIAL-BASELINES-20261008-R1. 실제 실행 준비 코드이며 현재 `READY`는 발급하지 않았다.
새 baseline 제출 0. 기존 baseline 60917–60923은 최신 사용자 지시에 따라 취소했다.

## 최신 oracle/W0 relay 반영 (2026-10-09)

`GH-SH4-OFFICIAL-ORACLE-W0-BINDING-20261009-R1`, main `34001ec0` 채택.
W0 담당 문의는 SH1 single producer / SH4 verified reader로 해소됐다.
reader 구현과 실제 READY는 여전히 `SOURCE_INPUT_PENDING`; full W0 중복 job0.

`oracle.py`는 공통 비교기만 호출한다. canonical raw identity를 그대로 두고 별도의
model/token/state binding을 넘긴다. qualification의 AlphaEdit first4와 production
AlphaEdit W20 full2K를 분리했으며 original forward 없는 CPU PASS를 승계하지 않는다.
`qualify.py`는 parent가 요구한 연속B3 대 coldB2→별도 프로세스B3 세 구간을
한 GPU에서 순차 실행하는 실제 CLI다. method당 native edit call6/batch100이며
추가 fullW0/generation/자동retry0. **이번 갱신에서 실행한 것은 CPU wiring뿐이다.**

현 blocker: 공통 원 NumPy display reducer와 Python-round logger 검산 불일치를
동일 source의 fixed2000 CPU fixture로 재현했다(`OFFICIAL_DISPLAY_SCORE_MISMATCH`).
원본 수치/공통 logger 수정, display 누락/가짜 READY 없이 GH/SH1 검토를 요청한다.
19개 own CPU tests에는 이 blocker 재현 test가 포함되므로 전체 실행 READY PASS가 아니다.

단발 scheduler 검산 시61418/61598/61618이 각GPU1/server4/ownrepo 할당이었다.
canonical2/local3의 effective2보다 현재3개다. 기존 실행 그대로 보존하고 새 등록0.
예정 qualification은 세 existing job 뒤의 serial lane이나 현재 dependency는 만들지 않았다.
원본 parity/resume, 실제 online 및 full2K 결과는 미관측이다.

## 코드와 입력

- `bind_assets.py`: 실제 EasyEdit 자산, prior SHA+현재 inode/mtime/size, CF·zsRE SHA와
  공식 정규화 stream 확인, 6개 config 생성. 외부 EasyEdit 알고리즘 import 없음.
- `native.py`: registry의 implementation/hparams/requests/call_options 사용. AE/SPHERE의
  module-global P/cache_c/context 및 BLUE 명시 cache_c 연결. BLUE projector는 원래
  `[4,5,6,7,8]`에서 물리 L4/L8에 해당하는 index0/4로 선택한다.
- `run.py`: cold FP32, TF32 off, native batch100, scheduled evaluation 후 atomic checkpoint.
  `--stop-after 2` + 별도 프로세스 `--resume --stop-after 3`가 실제 재개 경로다.
  편집/평가 실패는 이전 완료 checkpoint를 보존한다. 동일 W20 재개 시 state identity를
  검증한 observation만 재사용하며, 이전 source의 noCP 실행에는 이 경로를 적용하지 않는다.
- `submit.py`: config/source/main/READY와 5분 이내 admission 필요. 실제 sbatch held 등록,
  exact owner/command/GPU/CPU/memory/export/dependency 검사 후 release한다. 자동 retry 없음.
- `resume_check.py`: 실제 독립 cold 연속 B3와 B2→B3 receipt의 W/cache/context/RNG와
  factual 결과를 비교한다. CPU fixture를 실제 native receipt로 사용하지 않는다.

## 2026-10-09 공유 입력 결속

main `55afa07d2555718c4ad8b2e483db8a260aa94e35`와 GH 공통 `b10a87df`를 병합했다.
SH1 factual source `596896ff`의 실제 evaluate/build_zsre_w0_reference signature와
SHA를 `observe.verify_api()`로 확인하며 READY에 동일 receipt를 요구한다.
BLUE hidden/KV 및 Llama native generation 수정은 공통 게시본 그대로 사용한다.
기존 proposal patch를 별도로 적용하거나 evaluator/logger를 복제하지 않았다.

- `observe.py`: 실제 SH1 API 호출, current100의 occurrence/case identity 검산 및
  all-seen으로부터 독립 request-macro 재집계, W0 row/token/runtime identity 결속.
  zsRE는 reference의 `evaluation`을 그대로 재사용한다. W0 추가 forward 없음.
- `tracking.py`: 공통 `official.tracking` init/log/finish만 사용. request-macro는
  `official/*`, CF generation은 W0/W20 전용, zsRE는 generation config/metric 없음.
  GPU model load 전에 online startup을 확인한다. CPU adapter PASS는 remote PASS가 아니다.
- observer raw는 attempt별 local 경로, 재개 시 다른 시도의 timing/raw를 덮어쓰지 않는다.
- `check_factual_inputs.py`: 실제 Llama tokenizer로 두 first2K의 정본 token plan을
  CPU에서 확인. GPU/model forward 0, 입력 계획은 측정 성적이 아니다.
- CPU 52 PASS = 새 caller9 + 기존 own wiring3 + SH1 factual16 + 공통 fake SDK24.
  source148 SHA/209 Python AST/external-task imports0 PASS.

## 남은 실행 gate

1. CF 모델 공통 W0 factual/generation 및 zsRE W0 prediction producer/receipt 미결속.
   사용자 지시로 GH 직접 문의했으나 앱 tool unavailable/SSH 인증 제약으로 미전달.
   원격 credential/known_hosts 변경 또는 검증 우회는 하지 않았다.
2. 실제 native CF parity/B3 대 B2-resume 및 zsRE smoke는 NOT_RUN.
3. production main integration/exact freeze와 현재 admission/disk reserve 봉인은 미완료.
   준비 manifest는 READY가 아니며 boolean을 임의 PASS로 채우지 않는다.

현재 준비 자산은 ignored `local/official-baselines-20261008/preparation-v2/`.
v1 및 원 archive/raw는 보존했다. 정책 작업에서 GPU/Slurm/W&B 새 run을 만들지 않았다.

예전 CP·W0·source/raw를 삭제/전송/재계산하지 않았다. source/static/CPU wiring PASS는
실제 GPU qualification이 아니다. 허위 READY 파일로 위 gate를 우회하지 않는다.
