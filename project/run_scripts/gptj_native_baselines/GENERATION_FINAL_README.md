# Server2 GPT-J 최종 W20 generation 일정

최신 USER의 최종 2K 한 번 평가 및 현재 영향받은 모든 Server2 실험 변경 지시에 따른
명시 `final-v1` profile이다. 이전 runner/attempt/source/raw는 역사 기록으로 보존한다.
공통 generation/tracking source와 공유 EasyEdit/CAKE/BLUE 환경은 수정하지 않는다.

## 실행 범위

- MEMIT/AlphaEdit/CAKE/AlphaEdit-BLUE/PRUNE/RECT 독립 cold first2000 BS100×20.
- R/P/N W0/current pre·post/W5·10·15·20 all-seen/retention 일정과 native hparams는 그대로다.
- W0/current/W5·10·15 generation은 실행하지 않는다. 생성 W0 READY/old compatibility는 선행조건이 아니다.
- 실제 native20 commit 및 PRUNE terminal transform 뒤 W20 first2000에서 generation 한 번만 수행한다.
- 같은 생성문으로 entropy `H2/3 + 2H3/3` 및 기존 참조 TF-IDF cosine을 함께 계산한다.
- 미측정 중간 generation을 0 또는 observer PASS로 채우지 않는다. Final generation 실패는 이미 기록된 20 commit을 지우거나 rollback하지 않는다.
- 최초 BASE_MEMIT의 고정 최대8 prompt×3 route qualification은 별도 기술 작업이다. 새 fit/편집 pilot/미니 baseline은 아니다.

## 새 소스·정책·로깅

`generation_final_common`이 원 parent/repair envelope의 exact bytes와 최신
`control/generation-metric-policy.json`의 W20-only 정책, task-local USER override를 결속한다.
원 native/parser/target/solver/FP32·FP64/seed/context/sampler/reference/noCP는 변경하지 않는다.
과거 8600 case observation/arm + shared W0 2000에서, 새 계획은 2000/arm·여섯 arm 총12000이다.
동일 모델·경로의 실측 없이 이 비율을 시간 가속률/ETA로 주장하지 않는다.

W&B는 `wkdguswns2256` / 공백을 유지한 `layer allocation`의 online scalar run이다.
`generation_schedule=W20_ONLY_FIRST2000`, actual job ID와 run.name `job<id>` 및
새 UUID/attempt `final-generation-v1`을 기록한다. Shared immutable identity validator가
새 schedule key를 아직 등록하지 않아, 이 final profile만 task-private strict identity extension을
사용한다. 기존 profile은 원 shared identity create 경로를 유지한다. CPU의 실제 parent pipe-reader
fixture로 config echo/array0/signed step/privacy/nooverwrite를 검산하며 실제 online PASS로 표현하지 않는다.
진행 phase는 `W20_generation` / 별도 `generation_progress/step` 축이며, 최종 과학 점수는
`all_seen/post/*` / edits2000 단 한 payload다. RPN·fit 축과 섞지 않는다.

## 명령과 저장

전용 clean non-main WT에서만 실행한다. Create-once 명령이며 자동 재제출은 없다.

```bash
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.gptj_native_baselines.generation_final_bind
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.gptj_native_baselines.generation_final_cpu_checks --out /mnt/raid5/janghj/ODE-edit/local/gptj-baselines-fluency-consistency-2k/final-generation-v1/cpu-integration-final.json
# 소스 commit 후 전량 held 등록/검사/release; 이 명령이 실제 등록을 수행한다.
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.gptj_native_baselines.generation_final_submit
```

GPU launcher는 기존 `generation_run`의 명시 final dispatch, collector는 `generation_final_collect`다.
Qualification PLAN은 제출 전 고정하고 실제 receipt는 첫 GPU job에서 생성한다.
각 GPU1/CPU6/59392MiB/48h; collector GPU0/CPU6/24576MiB/4h. Wall은 ETA가 아니다.
합산 cap2 및 더 엄격한 현행 제한을 fresh source-based admission으로 검사한다.
BASE_MEMIT → 두 lane(AlphaEdit→BLUE→RECT, CAKE→PRUNE), collector afterany 전체6.
자원 대기는 정식 PENDING이며 agent freeGPU polling/heartbeat/automatic retry는 없다.
NoCP / exact_resume=NOT_AVAILABLE. Raw text/token/전체 stdout와 원본 실패·취소 자료는 ignored local KEEP,
Git/W&B에는 compact scalar/source/manifest만 기록한다. NO_BROADCAST_NOT_REQUIRED.
