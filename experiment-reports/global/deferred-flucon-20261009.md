# CF FLU/CON 후속 2K checkpoint 평가 지원 — 2026-10-09

사용자 지시: FLU/CON은 편집 실행 중 수행하지 않고 나중에 최종 2K checkpoint에서 평가한다.
입력: main `868e5ad5`의 Server2 checkpoint-only caller와 차단 보고서.
구현 기준 main: `67d00d4b`. 변경 branch: `codex/deferred-flucon-20261009`.

## 공통 API

`official.tracking`의 official CF config에서
`generation_schedule=DEFERRED_CHECKPOINT_EVALUATION`을 명시하면 생성 설정 없이 시작할 수 있다.
세 모델·네 서버 모두 동일한 공통 검증기를 사용한다. Server2의 기존 `cf_checkpoint` 경로는
이미 이 값을 생성하므로 runner 계산 코드를 바꿀 필요가 없다.

- official instruction/dataset/model/source/config 검증은 유지한다.
- deferred config는 generation profile/schema/seed/reference SHA/source SHA/repair 및 qualification metadata와 혼용할 수 없다.
- factual E/G/L/Score, native 진단과 fit 값은 정상 기록한다.
- deferred run에서 FLU/CON 점수·counts·생성 progress·phase는 거부한다. 0으로 대신 기록할 수 없다.
- 일정 누락/알 수 없는 일정은 여전히 오류다. 기존 enabled CF 검증과 zsRE·legacy 제약은 유지한다.
- W&B config에 deferred 일정이 남고, factual readback 성공은 FLU/CON 완료를 뜻하지 않는다.

최종 W20 checkpoint 저장·consumer pending·consumer 완료 전 삭제 금지는 Server2의 기존 caller가 담당한다.
후속 FLU/CON은 별도 평가 run으로 실제 checkpoint/source/config/sample identity와 생성 설정을 기록해야 한다.
이번 변경은 공통 검증기 지원이며 후속 GPU evaluator 제출이나 다른 서버 runner의 일정 자동 변경은 수행하지 않았다.
기존 FLU/CON 자산과 공식 지표 정의는 변경하지 않았다.

## CPU 검증

- `python -m unittest official.tracking.test_transport official.runners.server2.test_checkpoint_profile -q`: **32 PASS**.
  실제 Server2 config → shared schema → production worker → fake SDK → factual readback 통합 검사 포함.
  기존 20-batch fixture의 생성 호출 0 및 최종 checkpoint 보존 검사를 포함한다.
- `python -m unittest discover -s official/tests -q`: **70 PASS**.
- Server2 전체 discovery: **128 tests 실행, 개별 test 실패 0**, 별도 `test_qualification_input`의
  `setUpClass`에서 과거 producer source가 없어 오류 1. 해당 클래스는 미검증이며 전체 PASS로 보고하지 않는다.
  필요한 경로: `/mnt/raid5/janghj/ODE-edit/local/official-baselines-server2/20261008-r1/registration-qualification-r1/source/official`.
- `python3 -m official.tools.verify`: source 157개 SHA 및 Python 233개 검사 PASS.
- `git diff --check`: PASS.

Python: `/mnt/raid5/janghj/EasyEdit/.venv/bin/python`, `CUDA_VISIBLE_DEVICES=''`.
실제 GPU·Slurm·온라인 W&B 호출 0. 로컬 receipt와 실제 제출·평가 완료는 별도 확인이 필요하다.
