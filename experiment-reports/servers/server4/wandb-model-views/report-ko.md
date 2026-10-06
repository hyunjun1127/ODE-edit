# W&B 모델별 First 2k Saved View

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
