# server4 native runner 준비 상태

USER-OFFICIAL-BASELINES-20261008-R1. 실제 실행 준비 코드이며 현재 `READY`는 발급하지 않았다.
새 baseline 제출 0. 기존 baseline 60917–60923은 최신 사용자 지시에 따라 취소했다.

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

## 실행을 막는 미결 항목

1. SH1 `official.evaluation.factual` 및 실제 API 미게시. 현재 caller는 제안 API를 표시하고
   fail-closed한다. SH1이 게시한 실제 API에 맞춰 변경해야 한다. 임의 evaluator 복제 없음.
2. 공유 `native_generator.py`의 model_type guard가 gpt2/gptj만 허용한다. Llama는 현재
   `NATIVE_GENERATION_MODEL_FAMILY_UNSUPPORTED`. 공통 소유자의 Llama 검증/수정 필요.
3. BLUE compute_z의 tuple 전제. 감사 폴더의 검토용 patch는 적용하지 않았다.
4. 새 CF W0 모델당1회 공유 producer/receipt와 zsRE W0 예측 identity 미결속.
5. official-only W&B adapter 연결, native 실제 smoke/CF parity/B2→B3 resume 미실행.
6. 소스 검토/main 통합 및 정확 commit/tree freeze, 동시 checkpoint/output 용량 reserve 미봉인.

예전 CP·W0·source/raw를 삭제/전송/재계산하지 않았다. source/static/CPU wiring PASS는
실제 GPU qualification이 아니다. 허위 READY 파일로 위 gate를 우회하지 않는다.
