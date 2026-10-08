# GPT-J native connector: source/CPU 검토

이번 구현 소유 범위는 `official/runners/server2/native.py`와 전용 CPU fixture다.
공통 scientific implementation/hparams/solver를 수정하거나 외부 EasyEdit 알고리즘을
import하지 않았다. 실제 GPT-J forward·GPU·Slurm 실행 검증은 이 기록에 포함하지 않는다.

## 원 source와 연결

- 여섯 방법의 implementation, hparams parser, request schema, call options는
  `official.baselines.registry`에서 가져온다. device=0, 기존 stats/P 경로만 명시적으로
  배포 override한다. 외부 batch100, 20회와 native fit/solver/hyperparameter는 유지한다.
- FT는 native substring 선택 그대로 L21 `fc_out.weight` 및 **`fc_out.bias`**를 checkpoint한다.
  native optimizer는 매 apply에서 새로 만들어지며 batch 사이 optimizer는 저장하지 않는다.
- MEMIT/FE는 native COV_CACHE만 기존 C0에서 초기화한다. `official`의 CombinedStat.load와
  SecondMoment.moment().float().cpu() 경로를 사용해 FP32 raw mom2/count54924275를 그대로
  결속한다. H를 추가하지 않으며 layer_stats/dataset collector/자동 재계산을 호출하지 않는다.
- AlphaEdit/SPHERE/BLUE만 native H를 저장한다. 앞의 두 방법은 module-global P/cache_c,
  BLUE는 apply에 전달하고 반환되는 cache_c를 그대로 이어간다. fullsix P는 L3..8이며
  BLUE는 immutable physical slot0/5를 두-slot view로 전달한다.
- 각 engine은 native context/COV/P/H module-global을 자기 상태로 활성화하고 호출 종료 시
  원 module-global/model.config._name_or_path를 복구한다. 새 cold engine과 resume의 context는
  구분한다. `contexts()`는 cold에서 native 생성 1회, restore는 저장된 context를 설치하여
  재생성하지 않는다. W/H/RNG/cursor의 실제 B2→B3 비교는 별도 runner 검증이다.

## 발견한 initializer gap과 처리 경계

공통 AlphaEdit_main.py 83–90행 및 SPHERE_main.py 80–87행의 cache_c 초기화 분기에는
GPT-J가 없다. source에서 GPT-J일 때 cache_c가 정의되지 않을 수 있다. connector는
이번 계약에 명시된 native checkpoint state 관리로 six-slot FP32 zero H를 처음부터 만들고
native `cache_c_new=True`, 기존 P와 `P_loaded=True`를 결속한다. solver/history 수식은
수정하지 않으며 원 initializer byte는 불변이다. 실제 native 첫 write와 H append 결과는
GPU 실행에서 확인해야 한다. 이 연결을 upstream byte-identical initialization이라고 하지 않는다.

| source | SHA256 |
| --- | --- |
| official AlphaEdit_main.py | 1016bf13b4521dddd134f54884f00c1a7fb95c33e477c9ba2fe9e0f275e2fb9a |
| official SPHERE_main.py | 19934699a4604ad5ea87f5ad897d09514b0684b48b8f9fd88a9fc6750f0b208d |
| official blue/AlphaEdit/compute_z.py | 9ce6cddb251b4d27cde1f89c4570f25d556fbdba0ecfd55bb8b115e65d7cef8f |
| official runningstats.py | cae826b6471b6b5a4dd1f5da3adbd9cf697b01ae80585d59e727ca64f4ec47ae |
| S2 transformers GPT-J modeling_gptj.py | 2fea54b39fba964e2c7516bbd66e20eff3296c06fe540405d0020dd8e5b8264b |

현재 S2 Transformers4.57.1 GPTJBlock.forward는 `(hidden_states, attn_weights)`를 반환한다.
BLUE compute_z의 `cur_out[0]` 및 loss `.output[0]` 경로는 이 tuple 계약과 정적으로 일치한다.
MLP fc_out는 Tensor 경로다. Tensor/tuple에 대한 실제 native hook/input/output guard는
모델 실행에서 기록해야 하며 이 정적 일치를 actual GPU PASS로 표시하지 않는다.
BLUE repr_tools의 GPT-J block8 ln_1 별도 capture forward는 원 code 그대로 유지한다.

## 실제 수행한 CPU 검사

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`
`/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m unittest official.runners.server2.test_native -v`

11개 unittest PASS. 여섯 실제 native parser/import와 connector의 작은 FP32 fixture를 사용했다.
검사 범위: registry schema/options, FT bias 선택, BLUE physical P0/5 및 반환 H carry,
독립 global/context 관리, strict100·contiguous batch, nonselected mutation 차단,
weight/history/context 복구 및 missing-state 거부, model family/revision 구조 거부,
공통 atomic checkpoint의 실제 파일 저장/읽기를 통한 CPU fixture B2→B3 및 RNG 일치.
fixture의 작은 input width는 명시적으로 mock했으며 실제 GPT-J GPU qualification,
실제 numerical parity/resume, 성능 재현 또는 2K 완료를 증명하지 않는다.
