# PRICE 모델별 hparam 외부화 보고 — 2026-10-09

지시: `USER-PRICE-MODEL-HPARAM-CONFIG-20261009-R1`.
branch: `codex/price-model-hparam-config-20261009`.
시작점은 당시 `origin/main`의 `12eff7e3`이며 main 직접 push는 하지 않는다.
최종 구현 commit은 이 보고서를 포함하는 branch/PR commit으로 식별한다.

## 구현 결과

[`official/hparams/ours/`](../../hparams/ours/README.md)에 모델별 writer/price 두 파일,
schema와 네 개의 Qwen override를 배치했다. 설정 폴더에는 실행 결과·감사 자료를 섞지 않고
이 보고서 디렉터리에 검증·SHA·GPU 계획을 모았다.

[`config.py`](../../ours/config.py)는 torch 없이 JSON schema의 사용된 키워드와 정의역,
모델 결속, 중복/미지/누락 key를 검사하고 중첩 불변 Mapping을 반환한다.
baseline 파일은 resolve 시 읽지 않으므로 ours writer가 baseline 변경을 자동 추종하지 않는다.
모든 기본값을 파일에 적었고, K_eval과 legacy scale alias만 동일 설정에서 도출한다.

변경 파일과 SHA 전체는 [files.sha256.json](files.sha256.json),
원본/변경 후 배포 source SHA는 [source-changes.json](source-changes.json)에 있다.
SOURCES.json은 **ours 계산 모듈 19개 + official README 1개**의 설치 SHA만 갱신했다.
upstream commit/원문 SHA는 유지하고 변경 전 설치 SHA도 남겼다.
baseline hparams 18개와 profiles.json·sources.lock.json **20개는 bytes 불변**이다.
기존 PRICE JSON 세 개도 bytes를 보존하고 directory README에서 deprecated로 표시했다.

## 외부화 대응표

이전 줄 번호는 사용자 지시문 및 frozen 9ecdf834 기준이다. 새 코드의 줄 번호는 파일 링크와
SHA 목록을 기준으로 확인한다. 경로의 `core/`는 `official/ours/core/`를 뜻한다.

| 이전 위치 | 새 설정/흐름 |
| --- | --- |
| cap_controller.py:14,17 고정 β seal | price.beta_base; resolver의 유한·양수 검사 |
| cap_controller.py:14,16, cap_price.py:115,118 | price.c와 cap_mode; native 양수, none이면 nullable cap |
| cap_controller.py:18, cap_price.py:114,116, GPT-J price.py:119–124 | price.beta_max_scale; 두 price 경로와 controller가 동일 값을 사용 |
| cap_fit.py:32 | price.n_exp, price.K_grace; default fallback 제거 |
| cap_fit.py:32 / jlz_v12r/subject.py:38 | price.tau_F; subject와 controller의 단일 입력 |
| cap_fit.py:34 / GPT-J fit.py:34 / 두 Adam constructor | price.lr, eps, betas를 명시 전달 |
| jlz_v12r/optimizer.py:12 / realized subject의 norm 계수 | price.lambda_N |
| jlz_v12r/subject.py:36 / routes.py:12 / realized subject COEF | price.lambda_KL |
| 추가 발견: jlz_pilot/prompts.py native_loss의 KL default | 동일 config.lambda_KL; 별도 scalar default 제거 |
| 두 fit loop의 25/24, controller·Adam의 24 | price.max_updates; K_eval=max_updates+1, CONFIGURED_UPDATE_BUDGET |
| frozen cap_profile의 모델별 eligible/anchor/nll/λ_C dict | writer.hparams.layers의 순서와 마지막 층, v_loss_layer, mom2_update_weight |
| 추가 발견: realized/writer_coupled geometry의 λ_C=15000 default 3곳 | 필수 인자로 바꾸고 기존 a.profile.lambda_C 흐름 유지 |
| GPT-J adapter의 고정 L3–8, L27 readout | writer에서 해석된 eligible_layers·nll_layer, 실제 모델 깊이 범위 검사 |

GPT-J의 조기 nll readout도 실제 full/masked 경로에 연결했다. 기본 L27에서는 같은 tensor를
반환한다. shape·dtype·nonfinite·entry identity 및 `ACTIVE_FORWARD_BACKWARD_MASK_IDENTITY`
검사는 유지했다. 코드의 `SEALED_CAP_BASE_KNOBS`와 과학 knob literal은 제거했다.

## CPU 검증

기존 EasyEdit 가상환경에서 실행했다. 환경 설치/업그레이드 및 GPU 제출은 없다.

```bash
CUDA_VISIBLE_DEVICES='' /mnt/raid5/janghj/EasyEdit/.venv/bin/python \
  -m unittest discover -s official/tests -v
python3 -m official.tools.verify
git diff --check
```

결과: **70 tests PASS**, source 157개 SHA PASS, Python AST 230개 검사 및 외부 task import 0.
70개에는 기존 검사와 신규 설정/수치 회귀 14개가 포함된다.

- 세 모델의 해석된 기본값은 frozen cap_profile(LLAMA/QWEN) 및 보존한 GPT-J PRICE JSON과 일치.
- frozen source `9ecdf8342c52ee7e6a85a6ed741fc4f5468bc1ef`에서 oracle 원문 14개를 추출했다.
  **실제 61598 execution.lock의 source_members SHA와 14개 모두 대조했다.**
  oracle importer는 원문을 수정하지 않고 고유 namespace로 실행한다. task 운영 모듈을 실행하지
  않기 위해 cap_common의 literal ARMS/MODELS와 비수치 require/hash 유틸리티만 분리한다.
- controller 25후보 상태/확장 궤적, projection, Adam 출력·moment·counter, analytic norm의
  loss/gradient, subject 및 native helper의 forward/backward가 원문과 **tensor bytes 동일**.
- realized subject의 loss 계수·loss/gradient도 동일. 각 비교는 CPU의 작은 합성 입력이다.
  전체 모델·CUDA·실물 B1의 동일성으로 확대 해석하지 않는다.
- 네 arm에서 선언한 knob·파생 alias·identity만 변경된다. 잘못된 β/grace/타입/nonfinite,
  unknown/missing/duplicate key와 모델이 다른 arm은 거부된다.
- τ_F=.02에서 subject/controller 활성 mask가 같고, free cap·확장 scale·lr·λ_N이 실제 경로에 반영된다.
- 두 fit 경로에서 max_updates=1/3/27을 실행해 2/4/28 후보와 마지막 no-backward를 확인했다.
- W&B connector는 fake run으로 전체 설정 및 hash 전달을 검사했다. 온라인 W&B 전송은 미검증이다.
- GPT-J의 설정된 readout 8/20/27 및 hook 해제를 작은 모형으로 검사했다.

세 기본 config hash:

```text
llama3 b1dc6e64e44a2bd157143281f500a3f4014b5c4526fa7e711e96f02512746d06
qwen25 930251a302603cc31bcfd2d6c46d4aac9183023cf9d73a06b78d0786d0d3d3f6
gptj   f5b6888c7611b97af13470c97628528d6ec82ff08ff02debc43357541e6b09a8
```

수치 seal 대신 oracle fixture에 기본 hash를 기록해 의도하지 않은 기본값 변경을 검사한다.
arm 파일 변경·추가는 해당 arm의 identity를 바꾸며 기본 profile hash는 바꾸지 않는다.

잔존 literal 검색:

```bash
rg -n 'SEALED_CAP_BASE_KNOBS|\.75|\.0625|\.05|15000|25_24_BUDGET|range\(25\)|==24|>=24|<=24' \
  official/ours --glob '*.py'
```

출력 0행(exit 1). λ_N의 `.5 / a.square()`도 제거했다. 아래의 .5는 다른 수학적 계수다.

## 남긴 수치 상수와 제안

AST로 `official/ours`의 모든 숫자 literal **886개 위치**를
[constants-audit.json](constants-audit.json)에 기록했다. index·shape·bytes·정확한 수학 계수까지
포함한 전체 목록이다. 과학적 의미가 있는 나머지 항목을 다음처럼 분류했다.

| 남긴 값/위치 | 의미와 이번 판단 | 이후 제안 |
| --- | --- | --- |
| 두 price.py의 relative floor 1e-6, absolute floor 1e-12 | 가격 정규화 수식이며 이번 knob 표 밖 | 독립 민감도 실험을 설계할 때 별도 price schema로 이동 검토 |
| price range 1e6, 허용 오차 1e-8 | 위 floor 정의와 roundoff 검증의 결합 | floor와 함께 버전 관리하고 단독으로 느슨하게 하지 않기 |
| denominator >1e-8, LOO abs 1e-8 / rel 1e-6 | 잘못된 cached operator를 거부하는 수치 검증 | 일반 sweep hparam으로 노출하지 않고 수치 정책으로 유지 권고 |
| projection KKT 1e-10, FP32 stored budget 1e-6 | 해의 실현 오차 검사; projection 수식 불변 | 정밀도/rounding 변경 시 별도 qualification |
| geometry solve residual 1e-8, divisor floor 1e-30, symmetry/negative tolerance 1e-12 | factorization·잔차 수치 정책 | 고정 유지 권고; 변경하면 solver 재검증 |
| weight/context 합 검증 1e-7 | 정규화 assertion | 고정 유지 권고 |
| GPT-J hidden parity atol 2e-5 / rtol 2e-4 | 실제 모델 full/masked 오차 한계 | backend별 검증 범위로 관리 |
| AlphaEdit projector cutoff .02, λ_alpha=10 seal, projector 6개 | 역사적 Alpha writer 자산/설정 계약이며 MEMIT writer 파일의 대상 아님 | 별도 Alpha writer 설정·projector provenance를 정한 뒤 외부화 검토 |
| realized subject allocation=.1 | 별도 historical objective의 계수; 현재 PRICE fit에서는 사용하지 않음 | 그 objective를 다시 실행할 때 별도 schema에 명시 권고 |
| .5 in sqrt 미분/대칭화 | λ_N이 아닌 정확한 수학적 계수 | 상수로 유지 |
| 128/256/2048 및 microbatch 1/2 | 행 tile, dual solver·CPU/GPU 분할 등 계산 정책 | 처리 순서가 바뀔 수 있으므로 별도 성능 설정·재현 검증 대상으로 분리 권고 |
| GPT-J 4096/28/16384/50400, M1 prefix 5, token ignore -100 | 모델 architecture·pack/tokenization 계약 | 해당 모델/데이터 계약의 검증 상수로 유지 |

위 표 밖의 정책을 이번 refactor에서 튜닝하지 않았다. JSON arm 네 개도 실행하지 않았다.

## Collector/W&B와 실제 GPU 비교

fit 및 entry-price에 `resolved_config`, `config_sha256`, `config_arm`을 추가하고
기존 scale/cap 필드 이름은 alias로 보존했다. 수치 기본값은 같지만 provenance 필드가
추가되어 전체 record hash가 과거와 같지는 않다. `25_24_BUDGET` 오류 식별자는
`CONFIGURED_UPDATE_BUDGET`으로 바뀌었다. 새 collector는 후보 수를 receipt에서 읽고
새 arm 이름을 허용해야 한다.

W&B `config.ours`에 전체 설정을 넣는 connector를 제공한다. bare dict를 조립하던 runner는
`resolve()` 및 `plain()`으로 이행해야 하고, cached price의 config hash가 다르면 fit을 거부한다.
공식 ours의 MEMIT writer 설정만 이번에 제공한다. 기존 Alpha writer 변형 실행 설정 및
M1 GPU harness의 승인/통합을 완료했다고 주장하지 않는다.

실제 B1은 [GPU-SMOKE-PLAN.md](GPU-SMOKE-PLAN.md)의 GPU 1개·약 20분 계획만 작성했다.
실물 비교 기준 파일을 읽고 W/H 기대 hash는 확인했지만, refactor의 GPU smoke는 **NOT RUN**이다.
main 병합 및 GPU 실행 전 이 구분을 유지한다.
