# GPT-2 XL: ours × MEMIT/AlphaEdit writer × 세 arm, SH1 실행 계약

2026-10-07 KST. Task `jlz-price-gpt2xl-2k`, nonce `USER-GH-SH1-JLZ-PRICE-GPT2XL-SIXARM-2K-20261007-R1`.

## 최신 사용자 권한과 범위

> server1에도 gpt2-xl 6arm 진행시키자. ours와 memit,alphaedit writer가 호환이 되도록 llama코드를 참조하여 설계하여 server1에 task 진행시키자. server1의 gpu cap은 2이다.

> 이전 easyedit 코드에서 alphaedit과 memit이 gpt2-xl에 run 올리는 방식을 참고해서 진행해

> easyedit의 hparam도 사용해라.

최신 hparam 지시가 앞서 검토한 Llama 계수 유지보다 우선한다. 실행 권한은 구현·Slurm 제출·본실험까지 포함하며 추가 승인은 필요하지 않다. 연구 세션 `01a10dc1-4cd1-74b0-a951-3e043882782b`가 설계와 GH 전달을 맡고, GH `01a04939-8873-7673-8dca-4c7fc5e31af0`가 정본 게시와 SH1 직접 배정을 맡는다. SH1은 `01a04939-f93a-7b50-bca0-65438eab2062`, live CWD `/mnt/raid5/janghj/ODE-edit`다. inventory의 예전 worktree 경로와 live 세션을 혼동하지 않고 exact ID/CWD/origin을 확인한다.

이번 여섯 cell은 **모두 ours**이며 writer만 MEMIT 또는 AlphaEdit다. stock baseline을 여섯 개 더 추가하는 뜻이 아니다.

| writer별 arm | beta_base | layer cap |
|---|---:|---|
| CAP075 | 0.75 | 0.75 a |
| CAP100 | 1.00 | 0.75 a |
| FREE100 | 1.00 | 없음 |

MEMIT 세 arm과 AlphaEdit 세 arm 각각 cold W0/H0, BS100×20=2,000 edits다. 같은 CounterFact first2000·순서·seed20261002를 사용한다. FREE075/FLAT/REVERSE는 제외한다. edited W/H, 가격, 실현량을 cell 간 공유하지 않는다. server1 전체 project/owner GPU cap2 안에서 각각 GPU1을 사용한다. server4 job 종료 의존성이나 자원 제한을 server1에 이식하지 않는다.

## EasyEdit 실행과 hparams를 실제로 결속

입력 정본은 `/mnt/raid5/janghj/EasyEdit/hparams/MEMIT/gpt2-xl.yaml`과 `hparams/AlphaEdit/gpt2-xl.yaml`의 원 bytes다. 이 proposal의 `native-hparams/` 사본과 SHA를 봉인한다. SH1은 두 native HyperParams parser의 실제 해석과 자신의 effective config를 대조해 field별 소비·mapping·native 미사용 상태를 기록한다. YAML만 저장하고 실행 default를 유지하는 것은 금지한다.

검토한 실행 경로는 `examples/run_AlphaEdit_editing.py`의 HyperParams.from_hparams → BaseEditor.from_hparams → batch_edit(sequential_edit=True), `easyeditor/editors/editor.py`의 GPT 모델/tokenizer 로딩, 두 writer의 apply/execute/compute_z/compute_ks다. method runner는 이 native 입출력·hparam 의미를 가져오고, z 배분과 실현 응답 최적화는 ours로 유지한다. stock compute_z를 별도로 fitting하거나 native remaining-layer divisor를 ours 요청에 다시 적용하지 않는다. 기존 shell launcher의 환경변수·고정 GPU 번호를 복사하지 않고 server1 Slurm launcher로 결속한다.

| 항목 | 이번 실제 값·사용 |
|---|---|
| layers / fact_token | L13–17 / subject_last |
| anchor / v_loss_layer | 마지막 eligible L17 / readout47 |
| v_lr | 0.5; 실제 EfficiencyAdamAbs 생성자에 전달 |
| v_num_grad_steps | 20; native loop와 같은 최대 20회 평가·19회 갱신 |
| clamp_norm_factor | 0.75; native_c와 CAP arm의 layer cap 및 budget ceiling scale |
| v_weight_decay / kl_factor | 0.5 / 0.0625; 실제 norm·KL 항에 연결 |
| MEMIT mom2_update_weight | 20000; 15000 default를 사용하지 않음 |
| AlphaEdit L2 | 10; Llama/GPT-J 코드의 1을 사용하지 않음 |
| AlphaEdit nullspace_threshold | .02; 준비된 고정 projector bytes 재사용 |
| mom2 | Wikipedia, 100000 documents, FP32 native sum/count |
| rewrite / layer / attention | transformer.h.{}.mlp.c_proj / transformer.h.{} / transformer.h.{}.attn |
| ln_f | transformer.ln_f |
| lm_head_module | MEMIT: transformer.wte; AlphaEdit: lm_head. 실제 tied weight의 shape·alias를 확인 |

native compute_z는 `range(v_num_grad_steps)`의 마지막 평가 후 backward 전에 종료한다. 따라서 20회 갱신에 terminal 평가를 하나 더 붙이지 않는다. ours의 early-active/controller·grace12 own updates·n_exp4·F threshold .05·EfficiencyAdam의 layer scaling과 moment 방식은 method 정의로 유지한다. native stock optimizer로 method를 대체하지 않는다. hparam mapping과 ours에만 존재하는 항목을 별도로 기록한다. beta_max는 `max(beta_base, clamp_norm_factor * max_layer pi)`이며 FREE100은 실제 uncapped projector다. base1.0은 요청된 실험 arm이며 native clamp 값을 1.0으로 바꾸지 않는다.

Alpha YAML에도 mom2_update_weight=20000이 있지만 inspected native Alpha solve는 이 값을 normal equation에 쓰지 않는다. 이 필드는 native 미사용으로 기록하며 Alpha 방정식에 20000 C0를 추가하지 않는다. native relative model/stats/P 경로와 device/model_parallel은 준비된 절대경로·Slurm logical cuda:0로 해석하고 과학적 값은 바꾸지 않는다.

## 자산과 native 입력

준비 완료 근거는 `experiment-reports/servers/server1/gpt2-xl-wikipedia-alpha-projector/completed-repair-r2/{report-ko.md,asset-manifest.json}`다. READY SHA는 `755cb20cbc631058d22f4d81b9538f4b4893b62fa4652d05d30c5d362b21e657`다. SH1이 실제 파일 identity를 재확인한다. stats/projector를 다시 생성하는 작업은 포함하지 않는다.

- 모델: openai-community/gpt2-xl, revision `15ea56dee5df4983c59b2538573817e1667135e2`, 48 blocks, hidden1600, intermediate6400, vocab50257, max_positions1024.
- checkpoint: `/mnt/raid5/janghj/.cache/huggingface/hub/git-checkouts/openai-community--gpt2-xl/model.safetensors`, SHA `0f8b28eb05a8075f48b61b6f35332978c74fc7763fa9fb4051a1c30511736a6a`.
- N: `/mnt/raid5/janghj/EasyEdit/examples/null_space_project_gpt2-xl.pt`, FP32 `[5,6400,6400]`, SHA `187d83367339de1373b33ce39b101bb8c81445de9b3b9e5bf612f55b8a9608eb`. stack index0–4는 L13–17이다. FP64 재고유분해·threshold 재조정·basis 교체를 하지 않는다. L14/L16의 역사 FP32/FP64 nullity 차이는 기존 보고대로 보존한다.
- C0: 같은 manifest의 L13–17 mom2 NPZ; 각 count=44,068,071 masked token vectors. 문서 수100000으로 나누지 않는다.
- runtime 후보: EasyEdit `.venv`, torch2.9.1/transformers4.57.1의 기존 native GPT2 경로. 실제 runtime/config/source hash를 봉인한다. FP32 model, FP64 geometry, autocast/TF32 off, eval/frozen/use_cache=False. 준비 자산의 과거 미기록 TF32 상태를 소급해서 채우지 않는다.

GPT2Tokenizer, pad=eos, right padding을 사용한다. native leading-space target 처리, add_special_tokens=False, BOS/UNK 첫 토큰 처리, target-prefix 입력과 정확한 prediction positions, subject_last lookup, KL `{} is a`를 두 writer에서 대조한다. mean key는 FP32로 context-group mean을 낸 뒤 group mean을 내는 native 순서를 보존한다. Llama의 token IDs·context template·observer identity를 가져오지 않는다.

편집용 GPT2 context는 자산 준비에서 생성되지 않았다. 동일 cold GPT2/runtime/native generator provenance를 가진 기존 cache가 있으면 검증 후 사용한다. 없으면 **첫 실제 MEMIT_CAP075 cell의 cold 준비에서만** native get_context_templates로 `{}` + 5개 생성 context를 만든다(5 prompts, n_gen_per_prompt1, max_out_len10). RNG를 복원하고 context/20 packs/evaluator identity READY를 원자적으로 봉인한다. 이는 입력 준비이며 toy/별도 pilot가 아니다. 다른 cell은 그 READY를 재사용하고 중복 생성·부분파일 사용을 금지한다. 같은 상태·runtime·token/row identity의 cold W0만 재사용 가능하다.

모든 rewrite/KL/평가 true·new 입력 길이≤1024와 lookup/labels를 CPU 준비에서 확인한다. 초과 row를 조용히 자르거나 제거하지 않는다. W0/head/teacher/evaluator를 모두 GPT2 경로에 연결하고 평가가 이미 final ln_f를 거친 hidden을 반환하면 ln_f를 다시 적용하지 않는다.

## GPT2 adapter와 weight 방향

Llama ours 코드의 배분·손실·pullback·controller를 참조한다. `jlz_price_gptj`의 projection/attention/local_linear/compose/observer 추상화는 연결 구조 참고이며 **GPT-J의 parallel attention/MLP 잔차를 복사하지 않는다**. GPT2 native 4.57.1은 `attn(ln_1(x))+x` → `ln_2` → c_fc/activation → c_proj(+bias)/dropout → residual add의 직렬 구조다. wte+wpe와 native mask/position 처리는 실제 native full forward에서 capture한 첫 편집 block 경계를 재사용한다.

수학적 operator는 output×input으로 정의하지만 실제 Conv1D 저장 weight는 **input6400×output1600**이다.

```text
R_l: [1600,B],  K_l,Q_l: [6400,B],  C0_l,H_l,N_l: [6400,6400]
DeltaW_logical = R_l Q_l^T                    # [1600,6400]
DeltaW_native  = native match_shape(DeltaW_logical, W_native.shape)
W_candidate_native = W_entry_native + DeltaW_native.to(FP32)
```

실제 `a.weights`/guard/rollback/commit/hash는 native Parameter와 native payload를 기준으로 유지한다. adapter에서 input_dim/output_dim을 명시하고 `weight.shape[0]`을 R 차원, `shape[1]`을 H 차원으로 간주하는 Llama/GPT-J 코드들을 수정한다. engine.zeros/R 검산, H 초기화, source cold hash, 비용·실현 텔레메트리까지 포함한다. 논리 weight가 필요하면 명시적으로 native `.T`를 읽되 Parameter를 view로 교체하지 않는다.

local projection은 실제 Conv1D와 같은 flattened `addmm(bias, key, W_native)` 후 reshape를 사용한다. fixed c_proj bias는 수정하지 않으며 그 bias가 포함된 실제 FP32 candidate 출력과 entry 출력의 차이로 v를 구한다. canonical payload를 다시 연산하지 않고 terminal에 평가한 native bytes를 그대로 copy한다. bias와 attention·embedding·LayerNorm·head는 guard 대상이며 GPT2의 정상 wte/lm_head tying을 잘못된 편집 alias로 판단하지 않는다.

fresh upper BUILD, subject의 층별 block-output injection, full off-owner same-layer pullback, final rewrite mean-key H once를 유지한다. first-layer cache와 raw-context lookup부터 upper stage·native observer·commit 이후 key 검사까지 같은 adapter를 사용한다. 단순 module 문자열 치환이나 commit 전치만으로 호환 완료를 주장하지 않는다.

## 두 writer의 식과 hparam 전달

```text
MEMIT: A0 = 20000 C0 + H
       Q = (A0 + K K^T)^(-1) K

Alpha: A0 = 10 I + N H
       Q = (10 I + N (H + K K^T))^(-1) N K

공통: M = Q^T K, DeltaW_logical = R Q^T
```

H는 ours lifelong의 arm별 누적 mean-key covariance이며 batch마다 final native key로 한 번만 갱신한다. zero/미달 요청도 포함한다. Alpha의 native global cache를 arm 간 공유하지 않는다. MEMIT stock에 없는 ours의 history·배분은 method 항목으로 명시한다.

MEMIT은 기존 same-A Cholesky/LU fallback 경로와 residual≤1e-8을 사용한다. Alpha는 일반적으로 비대칭인 A0의 LU를 layer/batch당 한 번 만들고 `Y=A0^-1 NK`, `Q=Y(I+K^T Y)^-1`을 사용한다. candidate별 dense factorization/SVD, jitter/symmetrization, exact writer, full reverse는 추가하지 않는다.

Alpha L2=10은 diagonal만 바꾸는 수정이 아니다. residual은 `10 Q + N(HQ+K(K^T Q)) - NK`, LOO는 `(10I+NH)q_minus + NK_minus(K_minus^T q_minus) - Nk_r`로 바뀐다. price/commit/collector metadata와 검산에도 λ=10을 전달한다. `lambda_alpha=1`, `q+...`, `add_(1.)`가 남으면 실패다.

가격은 각 writer own-entry M에서 `X[r,j]=(a_r/a_j)M[r,j]/(1-M[j,j])`, off-diagonal RMS, 기존 floor/min-layer 정규화로 계산한다. batch 동안 고정하고 다음 batch 갱신한다. 층별 배분은 가격의 출력이며 L13을 강제하지 않는다. 약한 actuation이 싸게 보이는 경우를 위해 M_rr, ||Nk||/||k||, raw kappa/pi/floor 수와 실제 실현량을 함께 보고한다.

norm에는 native v_weight_decay=.5를 실제 전달하고 `sum_l ||R_lr|| / a_star,r^2`의 ours 확장을 유지한다. native의 total-loss 조기 종료와 ours의 F/controller 정의를 동일하다고 주장하지 않는다. native KL 계수와 reduction은 보존하고 평가 P/N을 optimizer/controller에 넣지 않는다.

## 구현 시 반드시 고칠 hardcode

현재 Llama/GPT-J reference에는 `EfficiencyAdamAbs(R)`의 default lr=.1, `range(25)`, terminal24, controller/optimizer 최대24, collector의 lr=.1/25·24/Alpha λ1 검산, event/storage/fit-axis bound25가 남아 있다. **effective hparams 하나에서 lr=.5/K_eval20/max_updates19/λC20000/λAlpha10을 생성해 producer부터 collector까지 전달**한다. 기존 별도 실험의 default·frozen source는 변경하지 않는다.

새 권장 namespace는 `project/run_scripts/jlz_price_gpt2xl/**`다. adapter/profile/hparams mapping, entry/builder/engine, parameterized fit/controller/optimizer, geometry/price/telemetry, run/commit/observer, prepare/freeze/submit/collect/tracking을 SH1이 소유한다. 공통 모듈 변경은 필요한 최소 backward-compatible parameterization으로 제한한다. SH1이 진행 중인 `experiment_tracking/**` 공통 helper와 새 caller를 같은 frozen source에 결속한다. SH4 등의 source를 병렬 수정하지 않는다.

## 실제 본실험 안의 검산과 게시

별도 toy, 작은 batch pilot, reference fitting, full-builder backward qualification은 생략한다. native loader/module/shape/hparam/input/static 검산 후 실제 각 cell B1에 다음을 통합한다.

1. R=0 candidate의 native full capture 대 masked/staged hidden·key·head identity. 이미 수행하는 entry forward와 cached candidate를 사용한다. 기존 원소 기준 atol2e-5/rtol2e-4를 사전 고정하고 실패하면 원인을 기록한다.
2. stored/native transpose, bias 보존, 실제 materialized response, writer normal equation과 LOO residual, 동일 Q의 pullback shape·whole-owner coupling을 확인한다.
3. repaired ZERO/CAP endpoint·tie, FP64 KKT/FP32 feasibility, exact-zero와 tiny 진단 분리, 20/19 controller/optimizer/event 일치, lr=.5와 실제 writer λ를 검산한다.
4. terminal native bytes exact copy, 기존 commit forward에서 final key parity와 H once, B2 own-state continuity. subject objective와 all-token actual loss의 차이는 진단이며 일치 강제 대상이 아니다.

성능이 낮다고 별도 승인·승격 gate를 만들지 않는다. 비유한 수치/shape/입력·자산 identity/commit 불일치/자원 cap 위반 등 기술 오류는 명시적으로 중단·보고하며 자동 재제출하지 않는다. 준비 자산의 CPU PASS를 method GPU PASS로 표현하지 않는다.

가격 분포/최저가 층·request support/종료 상태/expansion stage/cap/spend, 유효 ratio 분모·zero-target leakage·tiny endpoint 교정을 모두 남긴다. 비용은 logical output×input update 기준 C0/H 진단과 norm으로 보고한다. MEMIT weighted C0에는 **20000 배율**을 표시한다. Alpha 비대칭 A0를 손상 에너지 행렬로 쓰지 않는다. native stored delta의 전치를 누락해 비용을 계산하지 않는다.

current/pre·current/post 및 W5/10/15/20 all_seen/post를 분리하고 PS acquisition·cohort retention/lost/gained, R/P/N success·TF ACC·NLL·조화평균을 기존 evaluator 정의대로 보고한다. `control/wandb-method-metric-schema.json`과 `control/wandb-policy.json`을 적용한다. edits 축과 별도 fit/global_candidate 축, 실제 job ID/배열 index/signed step, model=gpt2xl/model_family=gpt2/writer/role=scientific/schema, immutable run receipt를 포함한다. W0_first2000은 정확한 GPT2 cold cohort만 뜻한다. 업로드 접수와 remote readback 상태는 구분한다. helper와 caller의 CPU 계약 검사를 마친 후 함께 freeze한다.

## 자원·순서·인계

SH1은 현재 same-owner server1 allocation/admitted queue와 실제 GPU/메모리·disk/inode reserve를 확인한다. **server1 합산 cap2**, cell당 GPU1, 두 writer lane에서 CAP075→CAP100→FREE100 afterany 직렬화를 기본으로 한다. native input READY가 없으면 ALPHA_CAP075도 최초 MEMIT_CAP075 종료 뒤 시작하도록 실제 dependency를 연결해 GPU를 점유한 채 file polling하지 않는다. READY가 이미 완성돼 있으면 두 lane을 바로 허용할 수 있다. 기술적 입력 준비 실패는 READY 검증으로 차단하고 선행 성능 PASS는 요구하지 않는다. CPU collector는 own six afterany/GPU0이다.

source/config/YAML/model/stats/P/input/runtime/helper/reducer identity와 source closure를 새 archive로 봉인한다. 여섯 job 전량 held 검사 후 release하며 구현·입력 준비가 끝나면 별도 승인 없이 제출한다. 기존 job/source/raw는 보존한다. 새로운 recurring monitor, checkpoints, 원본 raw 복제, historical W&B backfill은 포함하지 않는다. NoCP/raw local KEEP/compact Git 정책을 유지한다.

고정 N FP32 약0.763GiB, 다섯 FP64 factor 약1.526GiB 등의 **계획치**와 실제 owner graph/head workspace를 포함해 server1 자원을 잡는다. Llama의 GPU/host request를 그대로 복사하지 않는다. 실제 B1 fit/solve/telemetry 시간과 peak RAM/VRAM을 분리 기록한다. speedup/ETA나 성능 향상을 선험적으로 주장하지 않는다.

GH는 정본 task/envelope를 게시하고 SH1에 official app-server direct로 전달해 nonce·accepted turn·명시 owner ACK를 회수한다. SH1의 현재 공통 W&B helper 작업은 본 task의 직접 dependency이므로 관련 steer로 후속 task를 pending 등록하고 기존 helper 작업을 마친 뒤 구현을 계속할 수 있다. 무관 active turn이면 idle까지 기다린다. 전달 성공·구현 단계·task pending·Slurm pending·실제 job ID를 구분한다.

허용 산출물: 새 namespace와 task별 proposal/plan/audit/run/status/report, ignored `local/jlz-price-gpt2xl-2k/**`; 권장 보고서 `experiment-reports/servers/server1/jlz-price-gpt2xl-2k/report-ko.md`. source/model/runtime/하이퍼파라미터가 다른 역사 Llama·GPT-J 결과는 참고이며 이번 여섯 cell만으로 stock baseline 우월성을 주장하지 않는다.
