# 서버별 runner와 EasyEdit 자산 연결

사용자 지시(2026-10-08): 저장소 최상위 `official/`을 main에 배포하고 모든 새 baseline
실험의 코드 기준으로 사용한다. 각 서버는 이 폴더 안 자신의 하위 경로에서 EasyEdit 자산을
연결하는 runner를 작성한다. GPU가 없는 mock 실행으로 실제 실험을 대체하지 않는다.

| 소유자 | 쓰기 경로 | 실행 배정 |
| --- | --- | --- |
| server1 | `server1/` | Llama3 FT/MEMIT/FE, CF·zsRE |
| server2 | `server2/` | GPT-J 6개 방법, CF·zsRE |
| server3 | `server3/` | Qwen 6개 방법, CF·zsRE + BLUE L2 격자와 clamp 대조 |
| server4 | `server4/` | Llama3 AlphaEdit/BLUE/SPHERE, CF·zsRE |
| rent | `rent/` | 미배정 (A100 Kubernetes; job 번호 KST `DDHHMM`, [rent/README.md](rent/README.md)) |

공통 factual evaluator의 최초 통합은 server1이 `official/evaluation/factual.py`와 해당
테스트를 담당한다. 다른 서버는 입력·메트릭 계약을 공유하고 공통 파일을 동시에 수정하지 않는다.
공통 API나 수치 구현 수정이 필요하면 근거·patch를 보고하고 GH가 통합한다.

## Runner 요구 사항

1. `main`에서 isolated branch/worktree를 만들고 본인 `serverN/`에 실제 `run.py`,
   자산 연결, Slurm 제출·resume 경로를 작성한다. 다른 작업자의 변경을 되돌리지 않는다.
2. `official.baselines.registry`의 `implementation`, `hparams`, `requests`, `call_options`를
   사용한다. 서버 EasyEdit를 `sys.path`에 넣거나 EasyEdit의 알고리즘 코드를 import하지 않는다.
   Llama MEMIT은 `beta_hse=0, save_weights=False`를 명시하고 공통 checkpoint로 저장한다.
3. 각 서버 EasyEdit 경로의 기존 데이터·C0·P·모델을 읽는다. 샘플 순서, 파일 SHA와 모델 revision을
   고정한다. upstream layer_stats의 하드코딩 경로나 자동 재계산으로 우회하지 않는다.
4. 외부 batch는 100, 20회다. FP32, 동일 stream, seed와 평가 시점을 유지한다.
   AlphaEdit/SPHERE의 module-global cache와 BLUE의 명시적 cache_c는 batch 사이에 이어지고
   새 run에서는 독립 초기화된다. MEMIT/FE에 history를 새로 추가하지 않는다.
5. context template은 native 방법으로 시작 때 한 번 생성해 저장한다. resume 때 재생성하지 않는다.
   FT optimizer는 외부 batch마다 native 코드에서 새로 만들며 optimizer state를 재개하지 않는다.
6. FLU/CON은 `official.evaluation.generation.native_observer.NativeGenerationObserver`를
   연결한다. server1 현재 실행 소스와 CAKE/BLUE 연산 방식을 읽는다. 고정 reference 자산,
   case-batched KV, global endpoint RNG, total100 및 공통 지표 의미를 유지한다.
7. CF는 엄격한 NLL 비교와 요청별 macro, zsRE는 token 정확도와 W0 예측 일치를 구현한다.
   원본 AlphaEdit의 loc_ans 정확도도 별도 저장한다. raw/new/true NLL과 생성 원문은 ignored local에 둔다.
8. `official.experiments.checkpoint`에 W/cache_c/RNG/context/cursor/identity를 연결한다.
   checkpoint는 W0부터 기록하고 각 batch 편집·예정 평가 완료 뒤 최신 1개만 유지한다.
   실제 B2 중단→B3 재개와 연속 B3의 weight hash/지표를 방법별 검증한다.
9. 모든 과학 runner/평가 변경을 검토 후 main에 통합하고 **실제 제출은 main의 정확한 commit 및
   official tree SHA**를 동결한다. CPU mock PASS와 실제 모델 smoke/PASS를 구분한다.
10. 기존 실행은 유지하며 최신 서버별 GPU cap 및 현재 allocation을 확인한다. 본 실험은 CF를
    먼저 진행하고 zsRE는 모델별 1 batch smoke 뒤 실행한다. 메모리·디스크 예산은 실제 자산 크기로 계산한다.

현재 `prepare.py`는 CPU 자산 점검 도구이며 실제 GPU runner는 서버 담당자의 구현 범위다.
server별 `assignment.json`과 `assets.example.json`에 시작점을 제공한다.

## 인계 산출물

- 구현 commit, native apply/import source 목록, config/stream/tokenizer/assets SHA.
- CPU 검사와 실제 native smoke/resume 결과(각각 표시), Slurm job/자원·로그·출력 위치.
- compact Korean 보고서는 `experiment-reports/servers/<server>/official-baselines-20261008/`에 기록.
- raw 파일은 `local/official-baselines/...` 등 ignored execution path에 저장.
- 기존 artifact broadcast 정책을 적용하되 model/데이터 대형 복사·삭제를 새로 수행하지 않는다.
- 실패 원인은 재현된 exception과 source 근거로 기록하고, 설정을 결과에 맞춰 조정하지 않는다.
