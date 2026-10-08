# 모델별 PRICE(ours) 설정

`official.ours.config.resolve(model, arm=None)`가 이 폴더만 읽는다. 설정은 중첩 항목까지
불변인 Mapping이며, JSON으로 저장할 때는 `plain(config)`를 사용한다. 새 runner는
이 설정을 adapter에 한 번 전달하고 entry·engine·controller·optimizer·loss가 공유한다.

```text
ours/
  schema.json
  llama3/{writer,price}.json
  qwen25/{writer,price}.json
  gptj/{writer,price}.json
  arms/qwen25-beta150.json
  arms/qwen25-beta300-cap.json
  arms/qwen25-beta300-free.json
  arms/qwen25-tau002-grace8.json
```

`writer.json`의 `hparams`는 해당 모델의 `official/hparams/MEMIT/<model>.json`을 복사한
독립 snapshot이다. `source.path`와 `source.sha256`은 복사 당시 출처다. resolver는
baseline 파일을 다시 읽지 않는다. ours의 writer 설정을 직접 수정하면 해석된 설정 hash가
바뀌며, 원본 snapshot 출처는 그대로 남는다. baseline 36행의 설정에는 영향이 없다.

`price.json`은 모든 PRICE knob의 값을 명시한다. 기존 `../PRICE/*.json`은 역사 기록으로
보존하며 deprecated다. resolver의 fallback이나 기본값 공급원으로 사용하지 않는다.

## Native MEMIT와 ours의 기본값

| 모델 | native clamp | native v_lr | native v_weight_decay | ours β / c / 확장 scale | ours lr | ours λ_N |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Llama3 | .75 | .1 | .5 | .75 / .75 / .75 | .1 | .5 |
| Qwen2.5 | 4 | .5 | .001 | .75 / .75 / .75 | .1 | .5 |
| GPT-J | .75 | .5 | .5 | .75 / .75 / .75 | .5 | .5 |

세 모델 모두 λ_KL=.0625, τ_F=.05, n_exp=4, K_grace=12, max_updates=24,
eps=1e-8, betas=[.9,.999], cap_mode=native다. 후보 수 `K_eval=25`는
`max_updates+1`에서 도출한다. Qwen의 기본값을 native MEMIT 값으로 바꾸는 작업은 아니다.

Native clamp는 마지막 편집 층(현재 L8)에서 하나의 δ에 적용하는 ‖δ‖/‖h‖ 상한이다.
PRICE β는 여러 층 R의 가격 가중 L1 합 `Σ_l π_l ‖R_l‖ / ‖h_l‖`의 상한이고,
c는 각 층의 `‖R_l‖ ≤ c‖h_l‖` 상한이다. 수치가 같아도 같은 편집 강도를 뜻하지 않는다.
λ_N도 native weight decay와 norm의 정의·적용 위치가 다르다.

| writer snapshot 항목 | 해석된 profile | Llama3 | Qwen2.5 | GPT-J |
| --- | --- | --- | --- | --- |
| layers | eligible_layers | 4–8 | 4–8 | 3–8 |
| layers의 마지막 값 | anchor_layer, native_anchor_layer | 8 | 8 | 8 |
| v_loss_layer | nll_layer, nll_readout_layer | 31 | 27 | 27 |
| mom2_update_weight | lambda_C | 15000 | 15000 | 15000 |

writer의 v_lr, clamp_norm_factor, v_weight_decay, kl_factor, v_num_grad_steps는 native
출처 비교를 위해 보존한다. ours optimizer/loss/budget은 price.json의 명시된 값을 쓴다.
현재 제공하는 writer snapshot은 MEMIT이다. 역사적 AlphaEdit writer 변형의 L2/projector
설정은 이번에 추가하지 않았으며, 그 변형을 실행 가능한 새 설정으로 승인한 것은 아니다.

## 정의역과 override

- β_base, τ_F, lr, eps > 0; beta_max_scale, λ_N, λ_KL ≥ 0. 모든 수는 유한해야 한다.
- native cap일 때 c > 0. uncapped에서는 c=null을 사용할 수 있다.
- n_exp, K_grace는 0 이상 정수, max_updates는 1 이상 정수이며 K_grace < max_updates다.
- Adam betas는 원소 두 개이고 각각 [0,1)이다. bool을 정수로 취급하지 않는다.
- `K_eval`, `beta_max_native_scale`은 각각 max_updates, beta_max_scale의 읽기 전용 alias다.
  파일에 alias를 별도 작성하면 거부한다. τ_F와 λ_KL도 중복 설정 경로를 두지 않는다.
- unknown/missing/duplicate key와 다른 모델의 arm을 거부한다. `schema.json`의 타입·범위
  규칙은 추가 패키지 없이 검증하고, 층 순서·모델 결속·grace 관계는 resolver에서 검사한다.

arm 파일은 `{"base":"qwen25","override":{...}}` 형식이다. schema에 명시된 PRICE knob만
덮어쓸 수 있고 writer/architecture/source를 arm으로 바꾸지 않는다. 파일 이름이 실행 arm 이름이다.

| arm | β / scale | c | cap_mode | lr | λ_N | 추가 변경 |
| --- | --- | --- | --- | --- | --- | --- |
| qwen25-beta150 | 1.5 / 1.5 | .75 | native | .1 | .5 | 없음 |
| qwen25-beta300-cap | 3 / 3 | .75 | native | .25 | .05 | 없음 |
| qwen25-beta300-free | 3 / 3 | null | none | .25 | .05 | 없음 |
| qwen25-tau002-grace8 | .75 / .75 | .75 | native | .1 | .5 | τ_F=.02, K_grace=8 |

위 arm은 설정 파일만 추가했으며 sweep/GPU 실행은 하지 않았다.

## 코드 연결과 기록

```python
from official.ours.config import resolve, plain, wandb_config
from official.ours.core.jlz_interference_l1.cap_adapter import Adapter
from official.ours.core.jlz_v12r.entry import prepare_entry
from official.ours.core.jlz_interference_l1.cap_fit import fit

config = resolve("qwen25")  # arm="qwen25-beta150"은 명시적 실험 설정 변경
adapter = Adapter(model, config)
entry = prepare_entry(adapter, bench, pack, history, stats)
# run은 호출자가 만든 W&B run. None이면 tracker를 호출하지 않는다.
result = fit(adapter, entry, wandb_run=run)
resolved_json = plain(config)
```

GPT-J는 같은 resolver와 `jlz_price_gptj.adapter/entry/fit`을 사용한다. CPU 확인:

```bash
python3 -m official.ours.config qwen25
python3 -m official.ours.config qwen25 --arm qwen25-beta300-free
python3 -m official.tools.verify
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s official/tests -v
```

fit receipt와 entry-price에는 `resolved_config`, `config_sha256`, `config_arm`이 추가된다.
hash는 hash 필드 자체를 제외한 해석된 전체 설정의 canonical JSON(sorted keys, ASCII,
compact separators, nonfinite 금지) SHA256이다. writer/price/schema/arm 파일의 원문 hash도
그 안에 포함되어 source formatting 변경까지 추적한다.

W&B에는 `wandb.init(config=wandb_config(config))` 또는 `fit(..., wandb_run=run)`으로
동일한 전체 설정을 `config.ours`에 남긴다. `fit` 연결은 `allow_val_change=False`로 갱신한다.
기존 스칼라 telemetry allowlist를 바꾸지 않으며 W&B 온라인 전송 자체는 CPU 검증 범위가 아니다.
별도 offline receipt와 W&B config를 같은 `metadata(config)`에서 만든다.

기존 `native_c`, `beta_max_native_scale`, `local_caps`, `K_eval`, `arm` 등의 읽기 필드는
보존하지만 기록값은 실제 사용한 설정이다. 새 provenance 필드 때문에 record/config hash는
역사 기록과 달라진다. 기본값의 **수치 출력**과 hash가 포함된 **기록 전체의 bytes**는 구분한다.
옛 collector가 CAP075/CAP100/FREE100만 허용한다면 새 arm 문자열을 허용하고
`config_arm`으로 그룹화해야 한다. 고정 25/24가 아닌 receipt의 K_eval/max_updates를 읽는다.

bare dict profile이나 개별 scalar 인자를 받던 내부 API는 `ResolvedConfig` 입력으로 바뀌었다.
옛 runner는 새 adapter/entry/fit 결속을 해야 하며, frozen runner에 새 모듈을 주입하지 않는다.
M1/M2/M3 및 endpoint rounding의 기존 의미는 이번 knob 외부화에서 변경하지 않는다.
61598과의 실제 M1 실행 비교는 별도 [GPU smoke 계획](../../docs/price-hparam-config-20261009/GPU-SMOKE-PLAN.md)을 따른다.

검증·변경 SHA·남긴 상수의 전체 보고는 [구현 보고서](../../docs/price-hparam-config-20261009/README.md)를 참조한다.
