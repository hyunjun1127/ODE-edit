# W&B 모델별 First 2k Saved View

## Qwen 과거 결과 가용성 추가 확인 — 아직 reference 업로드 전

사용자 Qwen baseline/W0 확인 요청으로 기존 저장소 자료를 추가 조사했다. 최초 W&B inventory의 NOT_AVAILABLE은 당시 **원격 imported run 부재**였으며 로컬 결과 부재를 의미하지 않는다.

| 자료 | 실제 측정 범위 | 확인된 source |
|---|---|---|
| Qwen Official MEMIT | 누적 1k/1.5k/2k/3k/5k/7.5k/10k | server4 `official-layer-realization-debt-lifelong-b100x100-2026-09-03-v6/counterfact-primary-checkpoints.csv` |
| Qwen Official AlphaEdit | 위와 같은 7 endpoint | 동일 source, QA arm |
| Qwen Alpha O_NATIVE / JV_NATIVE | current B1–B10, 누적 W1/W5/W10; 최대1k | server2 `alpha-native-response-v31-sequential-routing-2026-09-06-v1/main-four-terminal-v1` |
| 과거 Qwen W0 | 동일1k cohort 1 endpoint | 위 server2 `final_metrics.csv`, PRE_EDIT_ORIGINAL_W0 |
| 현재 PRICE Qwen W0 | first2k raw26,000 rows, current100/prefix 재집계 가능 | job60004 W0; 동일 token/order/state CPU 검산 |

Qwen MEMIT-H/BLUE/CAKE의 10k 결과는 이 조사에서 확인되지 않았다. 원래 native lr/clamp/runtime 및 stream/order identity와 현재 PRICE의 차이를 historical reference로 명시해야 한다. 특히 server2 strict PS/NS request-cluster 수치를 prompt strict accuracy로 치환하면 안 된다. 사용자 최신 확인 요청 시점에는 위 Qwen reference를 **원격 업로드하지 않았으며** CPU preparation만 수행했다. 1k 결과를 2k/10k로 확장하지 않는다.

## 2026-10-07 사용자 지적 후 비교 연결 교정

직전 분리는 새 run을 baseline/PRICE-59768과 같은 패널에서 비교할 수 없게 만든 불완전한 처리였다. 사용자 `실시간 자동 동기화` 승인에 따라 별도 CPU 비교 브리지를 추가했다. 원본 GPU run의 frozen source와 logger는 그대로 유지한다.

확정된 commit 및 관측 raw의 case/order/token/state/분모를 검사하고 독립 재집계한 뒤 `current/pre`, `current/post`, `all_seen/post`의 기존 metric 이름과 % 단위로 비교용 `[comparison]` run에 기록한다. 원본 run ID/job/source/config SHA를 연결한다. 원본 run 동시 쓰기, 추가 모델 평가, scheduler 변경은 없다. 기존 baseline·PRICE-59768 기록도 수정하지 않는다. 동일 그래프·지표 정의를 맞춘 것이며 역사 baseline과 실제 runtime/seed/history까지 동일하다는 주장은 아니다.

최초 원격 readback 확인: MEMIT LLAMA_CAP075 job60001 B1–B4, QWEN_CAP075 job60004 B1. current 분모는 각각 R100/P200/N1000. 현재 새 누적 W5는 아직 관측되지 않아 all-seen에 점을 만들지 않았다. 도달 이후 실제 cumulative 관측만 동기화한다. `Live legacy logger — endpoint denominators differ` 섹션은 모델별 비교 view에서 제거했다. 학습 loss와 native logger는 원본 run에서 계속 볼 수 있다.

CPU 브리지는 정확히 현재 등록된 MEMIT/Alpha 12개 job을 대상으로 60초마다 확정 파일만 확인한다. 자체 종료는 모든 대상 terminal/W20 확인 또는 최대168시간이다. 이는 사용자 승인된 로깅 동기화이며 GPU 실험 polling/quality gate/재제출 기능은 없다. 미제출 GPT-J나 향후 별도 task는 실제 submission lock을 새로 등록해야 한다. pending arm에 가상 점이나 성공 run을 만들지 않는다. CPU metadata/기록 raw 테스트4개 PASS 및 원격 scalar 전수 readback PASS. 브리지 초기 SDK `Summary.update` 호출 형식 오류는 수리했으며 이미 업로드된 point를 검산·재사용해 중복 생성하지 않았다.

실행과 소스는 [bindings](../../../../audits/servers/server4/wandb-model-views/comparison-bindings.json), [view receipt](../../../../audits/servers/server4/wandb-model-views/comparison-overlay-receipt.json), ignored local `/data/janghj/ODE-edit/local/wandb-comparison-bridge/{launch.json,status.json}`에 분리 기록한다. 기존 아래 설명과 `receipt.json`은 최초 view 분리 시점의 역사 기록이다.

사용자 요청에 따라 원격 Saved View 3개를 추가하고 기존 비교 view를 모델별 탐색 화면으로 변경했다. GraphQL 재조회로 저장된 spec 일치, 서버 run 필터로 모델 간 교집합 0을 확인했다. 브라우저 렌더링은 별도 확인하지 않았다.

| 모델 | Saved View | 확인된 run 수 | imported baseline / W0 |
|---|---|---:|---|
| Llama-3-8B-Instruct | [First 2k](https://forge.coreweave.com/wandb/wkdguswns2256/layer%20allocation?nw=pricefirst2kllama) | 13 | baseline 9개 + W0 |
| Qwen2.5-7B-Instruct | [First 2k](https://forge.coreweave.com/wandb/wkdguswns2256/layer%20allocation?nw=pricefirst2kqwen) | 2 | NOT_AVAILABLE |
| GPT-J-6B | [First 2k](https://forge.coreweave.com/wandb/wkdguswns2256/layer%20allocation?nw=pricefirst2kgptj) | 0 | NOT_AVAILABLE |

기존 [Ours and Baselines - First 2k Comparison](https://forge.coreweave.com/wandb/wkdguswns2256/layer%20allocation?nw=y9vet5iwjzr)은 위 세 모델 바로가기를 제공한다. 아래 기존 곡선은 Llama 전용임을 명시했다. 기본 workspace, Full History 10k, Batch vs Cumulative view는 원 spec 그대로 보존했다. 위 run 수는 설정 시점의 포함 개수이며 성공/완료 개수가 아니다.

## 비교 범위와 결측

현재 imported MEMIT/MEMIT-H/MEMIT-BLUE/AlphaEdit/AlphaEdit-BLUE/CAKE/v12-MEMIT/v13-MD/v13-CD 및 W0는 Llama 자료다. 기존 baseline 업로더의 source report 연결과 모델 identity를 확인했다. 근거는 `blue-native-lifelong-comprehensive-review-2026-09-11-v1/source-config-compatibility.csv`, server3 MEMIT-H `asset-preflight.json`, CAKE/v12/v13 보고서의 모델 identity다. 같은 모델이라도 runtime/history/cohort 차이는 historical comparison이며 matched causal 비교로 승격하지 않았다. Qwen/GPT-J baseline을 Llama 자료로 대체하거나 0으로 채우지 않았다.

현재/current100과 누적/all-seen 패널 및 RS/PS/NS/harmonic, TF 지표는 기존 namespace를 유지했다. x축은 0–2000 edits다. 봉인된 생산 logger의 `eval/*`는 current와 milestone 누적 endpoint가 섞여 있으므로 별도 live 패널과 R/P/N 분모 패널에만 표시한다. baseline의 % 단위와 live fraction 단위를 혼합하지 않았다. 누락된 split-endpoint metric은 새 평가나 임의 재구성으로 채우지 않았다.

## 새 run 자동 적용

Saved View는 `config.model_family` 또는 `config.model_profile`의 LLAMA/QWEN/GPTJ 값, 또는 정확한 model-prefixed `config.arm`을 사용한다. 현재 공통 helper whitelist를 변경하지 않고 호출부 `cap_tracking.model_scoped_arm`을 추가했다. 기존 LLAMA/QWEN arm 문자열은 불변이며 GPT-J 새 caller가 공통 `start()`를 사용하면 `GPTJ_MEMIT_CAP075`/`GPTJ_ALPHA_CAP075` 등의 6개 identity로 자동 분류된다. 미확인 모델은 추정하지 않고 모델/arm 불일치는 명시 오류다.

CPU 메타데이터 테스트 4개 PASS. 이는 모델/GPU qualification이 아니다. SH1 소유 공통 helper는 수정하지 않았다. 향후 일반 실험 caller도 모델 식별자를 반드시 제공해야 하며, 이 패치는 아직 구현되지 않은 임의 caller까지 자동 수정하지 않는다. split endpoint metric schema 확대는 SH1 연계 사항으로 남아 있고 직접 전달 도구는 사용 불가여서 전달 완료로 기록하지 않았다.

## 보존 및 감사

기존 run config/history 수정 0, 신규 science run 0, Slurm 변경 0, 모델 load/forward 0. 기존 frozen archive와 등록 job은 변경하지 않았다. 원 view와 run 메타데이터 백업은 `/data/janghj/ODE-edit/local/wandb-model-views/`에 보존한다. compact [receipt](../../../../audits/servers/server4/wandb-model-views/receipt.json)에 원격 view ID/SHA와 membership을 기록했다. 검토 수준은 owner audit이며 독립 reviewer를 사용하지 않았다. NO_BROADCAST_NOT_REQUIRED: 소형 source/receipt/report만 Git 게시, credential/SDK spool/raw는 미전송.
