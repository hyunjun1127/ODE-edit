# JLZ v12 shared-budget 2k 실행

Nonce: `ODEEDIT-USER-GH-SH4-JLZ-V12-SHARED-BUDGET-2K-20261004-R1`

현재 상태: **파일럿 완료, 본선 RESOURCE_PENDING 초기 인계**. 본선 초기 및 W20 결과는 아직 NOT_OBSERVED / NOT_MEASURED다. 2026-10-04 10:26:55 UTC에 본선 dependency 해제와 Resources 대기, 서버 Slurm GPU 8/8 할당을 확인했다. 이후 agent 모니터링·자동 재개는 중단했다.

## 결속과 실행

- 정본 27파일 523,263B SHA/size 검증, method/experiment/implementation/TeX/telemetry/reference 전체 정독. 2,000행 입력 순서와 파일럿 분리 검증.
- reference CPU 29 PASS, 생산 CPU 회귀 7 PASS. 작은 임의 CPU Llama는 실제 대상 Llama/GPU 검증이 아니다. 독립 reviewer 없이 SH4 owner audit와 명시적 red-team checklist를 수행했다.
- 실행 source `7852d66e`; publication source와 실행 archive는 별도다. 실제 immutable source/config/launcher SHA는 submission receipt의 execution lock에 결속했다.
- 파일럿 `58172` → 본선 `58173` → collector `58174`. afterany는 실패 수집을 위한 연결이며, 본선은 같은 source/config의 파일럿 READY 없이는 모델을 로드하지 않는다. task 동시 GPU 1, project 최대 3, 각 GPU job 8CPU/59392MiB, export NONE/Requeue 0.
- 실제 파일럿 BS2×2: 2 commit/10 layer-history append, W/H/RNG 연결 및 observer 비변이 확인. cached/full loss·gradient·canonical capture, native K, absolute-D chain rule 및 scaled-epsilon update 검산 통과. 이는 작은 파일럿 범위이며 main B100 PASS가 아니다. 원 결과와 기술 수치는 actual-pilot audit에 보존했다.
- 파일럿 parent allocation 410 GPU초, 프로그램 실측 404.21초, peak VRAM 38,248,571,904B, peak RSS 34,730,432KiB. 요청 wall 4h 및 본선 7days는 ETA가 아니다.
- 새 V12_MAIN 한 경로만 cold W0/H0, BS100×20. 추가 arm/baseline/B100 fit/checkpoint 없음. 중단된 v11은 재개하지 않았고 다른 job은 변경하지 않았다.
- 기존 W0 first2000 raw 26,000행의 모델·입력·token·evaluator·runtime identity를 결속했다. 본선 cold 모델 hash까지 일치할 때만 재사용하며, 불일치를 기존 평가 PASS로 대체하지 않는다.
- 파일럿/본선 원자료: `/data/janghj/ODE-edit/local/jlz-v12-shared-budget/20261004-v1/attempt-r1/`. raw/prompt/tensor/fullstdout은 Git에 넣지 않는다.

## 구현 경계

요청 독립 EfficiencyAdam, FP64 공유 group-L1 projection, 요청별 before-backward stop/freeze, 동일 평가 forward의 terminal z를 구현했다. Writer는 각 층의 실제 하층 write 이후 K/h를 다시 측정하고 z−h를 native ridge로 적용한다. 마지막 actual mean key로 history를 층당 한 번만 누적한다.

모델/공유 환경은 수정하지 않았다. 기존 venv에 없던 JSON Schema 검증기는 시스템 설치본에서 task 전용 dependencies-r1로 복사하고 SHA 봉인했다. 최초 import 실패와 수정은 GPU 실행 이전의 CPU 단계다.

## 산출물과 해석

현재 scientific 결과값은 NOT_MEASURED다. sealed collector는 pre/post current 및 W5/10/15/20 allseen, ACC/RS/PS/NS/NLL, 두 부호 margin, paired/cohort/state/history/count/cost를 CPU로 검산한다. 20 commit와 모든 endpoint가 없으면 PARTIAL이며 scheduler 종료를 과학 완료로 간주하지 않는다.

동일 조건 baseline의 GH matched comparison receipt는 아직 결속하지 않았으므로 matched baseline은 NOT_AVAILABLE다. 신규 baseline fit은 없다. 낮은 성능·집중·KKT·고정 예산 미수렴은 실행 gate가 아니다.

정식 본선 자원 대기를 확인해 agent polling을 중단했다(`monitoring_active=false`, `automatic_resume=false`). 완료 상세 리뷰는 사용자 recall 때 수행하고, 이미 등록한 runner/collector만 W20까지 자연 진행한다. 낮은 성능을 이유로 재시도하거나 500에서 중단하지 않는다.
