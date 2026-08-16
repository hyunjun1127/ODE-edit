# P1R23 원본 B10 Progress-Simplex Atomic 재현 보고서

## 결론

전체 판정은 **NUMERIC_NONDETERMINISM**이다. 원본 outcome-selected B10, W0, 모델·토크나이저·평가기, 과학 소스와 수치 계약을 동일하게 고정했을 때 Neutral 4개 arm은 원본 job18697을 과학적 수치와 endpoint까지 정확히 재현했다. 반면 Soft 4개 arm은 k1의 동일한 q·물리 slope 뒤에 계산된 structural-P 정규화에서 `1.7e-13`(Llama)~`1.6e-10`(Qwen) 규모의 첫 수치 차이가 생겼고, simplex 해의 `pi/v`가 달라지면서 이후 BF16 상태와 endpoint가 갈라졌다.

최근 independent-B10×10의 case01은 request order가 `20933669...`로 원본 `984fe6ec...`와 다르다. 따라서 그 결과가 원본 Atomic을 재현하지 않은 주된 비교 경계는 **SOURCE_OR_INPUT_MISMATCH**이며, 원본 B10 재현 결과와 직접적인 동일-sample 재현 판정을 해서는 안 된다.

`scientific_promotion=false`. 이 샘플은 outcome-selected 재사용 B10이며, 본 보고서는 재현성/기계적 진단만 다룬다.

## 실행 및 고정 정체성

- 지시: `ODEEDIT-S05-P1R23-PROGRESS-SIMPLEX-ORIGINAL-B10-REPRO-V1`
- 과학 checkpoint: `a343d1f6967ef37763009b92d227ade85cd93de0`
- server1 launcher-only 최종 child: `baad69b298ff0d657c361a5f52d90161ed689547`
- child tree: `229248edac7523d868b463b3d809eb26cbdb5618`
- seal root: `3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628`
- request order: `984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b`
- 모델당 W0 endpoint는 원본과 exact: Llama `2/3/84`, Qwen `0/4/80` (Eff/Gen/Loc).
- 과학 closure(`progress_simplex_routing`, scalable runtime/experiment/panel, entry, numerical lock)는 `a343d1f`와 blob-byte 동일하다.
- valid job: Slurm array `18935_[0-3]%4`, devbox, 각 1 GPU/8 CPU/65000 MiB. 네 paired task가 Neutral+Soft 총 8 arm을 실행했다.
- task elapsed/MaxRSS: Llama BG `355 s/7,419,260 KiB`, Llama RS `337 s/7,387,532 KiB`, Qwen BG `420 s/10,623,396 KiB`, Qwen RS `421 s/10,519,052 KiB`.
- 8/8 arm 모두 `K8/tau1`, retry/backtracking 0, scientific-invalid 0, action-freeze 및 각 arm W0 pointer restore PASS.

### 순수 기술 실패 경계

최초 array `18931_[0-3]`은 6초 후 모델·결과·과학 동작 전에 동일한 argparse 오류로 종료됐다. launcher가 재현 전용 run token을 넘겼지만 원본 entry는 `p1r23-progress-simplex-router-v1`만 허용했다. 결과 root는 생성되지 않았다. run token만 원본 값으로 복원한 `baad69b`에서 `18935`를 실행했으며 과학 소스·방법·허용오차는 변경하지 않았다.

## Endpoint 비교

표의 연속값은 `new NLL / old NLL / margin(old-new)`이다. `재현`은 job18935, `원본`은 job18697이다.

| 모델 | 배분 | arm | 재현 E/G/L | 원본 E/G/L | 재현 E N/O/M | 원본 E N/O/M | 재현 G N/O/M | 원본 G N/O/M | 재현 L N/O/M | 원본 L N/O/M |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama | BG | Neutral | 10/20/81 | 10/20/81 | .031889/10.309375/10.277486 | .031889/10.309375/10.277486 | .844440/9.442188/8.597748 | .844440/9.442188/8.597748 | 8.196406/4.322847/3.873560 | 8.196406/4.322847/3.873560 |
| Llama | BG | Soft | 10/20/80 | 10/20/81 | .025787/10.512500/10.486713 | .021565/10.425000/10.403435 | .831058/9.448438/8.617380 | .807475/9.287500/8.480025 | 8.198125/4.323015/3.875110 | 8.211250/4.307705/3.903545 |
| Llama | RS | Neutral | 10/20/81 | 10/20/81 | .020693/16.187500/16.166807 | .020693/16.187500/16.166807 | .526387/12.743750/12.217363 | .526387/12.743750/12.217363 | 8.343594/4.297285/4.046309 | 8.343594/4.297285/4.046309 |
| Llama | RS | Soft | 10/20/82 | 10/20/81 | .020636/16.137500/16.116864 | .020660/16.137500/16.116840 | .541033/12.646875/12.105842 | .537638/12.667188/12.129550 | 8.342812/4.283970/4.058843 | 8.338594/4.288025/4.050569 |
| Qwen | BG | Neutral | 10/20/80 | 10/20/80 | .053979/12.965625/12.911646 | .053979/12.965625/12.911646 | 1.906824/10.566406/8.659583 | 1.906824/10.566406/8.659583 | 8.832812/5.177129/3.655684 | 8.832812/5.177129/3.655684 |
| Qwen | BG | Soft | 10/20/80 | 10/20/81 | .083437/12.071875/11.988438 | .051495/13.046875/12.995380 | 2.170728/9.836719/7.665991 | 2.095862/10.134375/8.038513 | 8.816875/5.181758/3.635117 | 8.814375/5.171914/3.642461 |
| Qwen | RS | Neutral | 10/20/81 | 10/20/81 | .037970/18.481250/18.443280 | .037970/18.481250/18.443280 | 1.465430/11.296875/9.831445 | 1.465430/11.296875/9.831445 | 8.847188/5.169692/3.677495 | 8.847188/5.169692/3.677495 |
| Qwen | RS | Soft | 10/20/79 | 10/20/81 | .038273/18.571875/18.533602 | .037681/18.912500/18.874819 | 1.351915/11.153125/9.801210 | 1.487311/11.139063/9.651752 | 8.846250/5.167715/3.678535 | 8.841875/5.179419/3.662456 |

Neutral은 네 cell 모두 count뿐 아니라 E/G/L의 연속 NLL·margin까지 exact 일치했다. Soft는 Eff/Gen count는 유지했지만 Loc가 원본 대비 각각 `-1,+1,-1,-2`로 변했고 연속값도 달라졌다.

## Receipt-chain 첫 차이

모든 새 run의 accepted-k1 `field_sha256`와 `routing_problem_sha256`는 원본 요약에 저장된 값과 달랐다. 이 identity는 재계산된 부동소수 telemetry/child provenance를 포함하므로 해시 차이만으로 과학 상태 차이를 판정하지 않고 수치 leaf를 대조했다.

| cell | k1 q/slope | k1 structural-P normalization 차이(재현-원본) | functional-P normalization 차이 | k1 `pi` 최대차 | 첫 BF16 materialization 차이 | 판정 |
|---|---:|---:|---:|---:|---:|---|
| Llama BG Neutral | exact | decision influence 0 | decision influence 0 | exact | 없음(K1–K8 exact) | EXACT_REPRODUCTION_PASS |
| Llama RS Neutral | exact | decision influence 0 | decision influence 0 | exact | 없음(K1–K8 exact) | EXACT_REPRODUCTION_PASS |
| Qwen BG Neutral | exact | decision influence 0 | decision influence 0 | exact | 없음(K1–K8 exact) | EXACT_REPRODUCTION_PASS |
| Qwen RS Neutral | exact | decision influence 0 | decision influence 0 | exact | 없음(K1–K8 exact) | EXACT_REPRODUCTION_PASS |
| Llama BG Soft | exact | `-1.658e-13` | `+8.674e-19` | `1.938e-9` | k1 | NUMERIC_NONDETERMINISM |
| Llama RS Soft | exact | `-1.889e-13` | `-1.735e-18` | `5.119e-10` | k7 | NUMERIC_NONDETERMINISM |
| Qwen BG Soft | exact | `+1.585e-10` | `-8.674e-19` | `2.106e-8` | k1 | NUMERIC_NONDETERMINISM |
| Qwen RS Soft | exact | `+3.707e-11` | exact | `3.885e-9` | k1 | NUMERIC_NONDETERMINISM |

추가 leaf 대조:

- Neutral: 4/4 cell에서 K1–K8의 q, alpha_req, signed slopes, pi, v, predicted progress가 exact였다. 선택 capacity telemetry만 최대 `2.39e-15` 차이가 있었고 라우팅 influence는 0이었다. 32/32 materialization identity와 endpoint가 exact였다.
- Soft: k1 q와 signed slopes는 4/4 exact였다. 즉 차이는 target/physical-progress 축이 아니라 P geometry/solver 축에서 시작했다. structural-P normalization의 미세 차이가 simplex 해를 바꿨고, Llama-RS를 제외한 세 cell은 k1에서 바로 BF16 materialization이 달라졌다. Llama-RS는 미세한 pi 차이가 k6까지 BF16 반올림에 흡수되어 k7에서 처음 materialization이 달라졌다.
- W0 endpoint, request order, 모델·토크나이저·평가기, source closure, K/h/tau, q/slope는 일치한다. paired initial gate는 양쪽 residual exact-zero와 metric equality를 확인했고, 각 arm의 W0 pointer restore가 통과했다. 따라서 RUNNER_STATE_LEAK 및 SOURCE_OR_INPUT_MISMATCH는 새 원본-B10 run 대 원본18697 비교에서 지지되지 않는다.
- 원본 package가 CUDA driver/kernel 및 완전한 scheduler 환경 fingerprint를 기록하지 않아 bit-level 차이의 하위 원인을 특정 GPU kernel까지 좁히지는 못했다. 따라서 상위 판정은 NUMERIC_NONDETERMINISM이며 SCHEDULER_ENVIRONMENT_DIFFERENCE는 NOT_RECORDED이다.

## 최근 independent-B10×10 case01과의 경계

case01 request order는 `20933669138b5090645da9ea81b662085081fbc7eb756f542ec90bb9f51435ac`로 원본 order와 모델 실행 전부터 다르다. 따라서 deterministic chain의 첫 차이는 request input이다.

| 모델/arm | case01 상태 또는 E/G/L |
|---|---|
| Llama BG Neutral | PASS, 10/18/92 |
| Llama BG Soft | typed failure, endpoint 없음 |
| Llama RS Neutral | typed failure, endpoint 없음 |
| Llama RS Soft | PASS, 10/19/92 |
| Qwen BG Neutral | PASS, 10/18/92 |
| Qwen BG Soft | PASS, 10/17/92 |
| Qwen RS Neutral | typed failure, endpoint 없음 |
| Qwen RS Soft | PASS, 10/18/92 |

case01의 높은 Loc와 낮은 Gen 또는 typed boundary는 다른 B10의 결과다. 이를 원본 sample의 재현 실패, state leak, 또는 동일 arm의 성능 변화로 해석할 수 없다.

## FACT / INFERENCE / NOT_RECORDED

### FACT

- job18935는 4/4 task COMPLETED `0:0`; 8/8 arm K8/tau1, retry/backtracking 0, scientific-invalid 0이다.
- Neutral 4/4는 원본의 모든 기록된 핵심 per-step 수치와 endpoint를 exact 재현했다.
- Soft 4/4는 동일 k1 q/slope 뒤 structural-P 정규화부터 미세하게 달랐고, 이후 trajectory 및 endpoint가 달라졌다.
- 모든 terminal manifest는 `W0_restored=true`; 8/8 endpoint restore가 pointer identity와 다섯 writer weight hash를 보존했다.
- 최근 case01과 원본 B10은 request order가 다르다.

### INFERENCE

- 원본-B10에서 Neutral path는 server1에서도 결정론적으로 재현된다.
- Soft path는 P geometry와 constrained solver의 아주 작은 부동소수 차이를 증폭시키며, BF16 경계를 넘은 뒤 endpoint count까지 바뀔 수 있다.
- 최근 case01 이상 현상의 원본-B10 대비 1차 원인은 다른 sample/input이며, Soft 자체의 별도 수치 민감성도 존재한다.

### NOT_RECORDED

- job18697 실행 당시의 완전한 CUDA driver/kernel/environment fingerprint.
- structural-P `1e-13~1e-10` 차이를 특정 단일 GPU 연산으로 환원하는 kernel-level trace.
- fresh-sample promotion 또는 동일 GPU/동일 process에서의 반복 재현 분포.

## Artifact hashes

- 원본 SH2 report: `/mnt/raid5/janghj/ODE-edit/local/source-handoff/P1R23_PROGRESS_SIMPLEX_RESULTS_SH2_V1/report-ko.md`, SHA-256 `c508242eb798fc0db4a3ac7c84e1e94eb7f656c5ab9e2dd4e7b23edbdf329c29`.
- 원본 stepwise: `/mnt/raid5/janghj/ODE-edit/local/source-handoff/P1R23_PROGRESS_SIMPLEX_RESULTS_SH2_V1/stepwise-analysis.json`, SHA-256 `64fa5e0205d640d83afd4e448c75dbf24b71fb230a644fe98697950f50aed212`.
- 최근 B10×10 report: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r23-progress-simplex-independent-b1x100-v1/local/odebf/reports/p1r23-independent-b10x10/p1r23-independent-b10x10-terminal-report-ko.md`, SHA-256 `e82eaf647e00de0f35e051dc487ae6602d447fbf2714f3a61aece3d511c0017e`.
- job18935 terminal SHA-256: Llama-BG `4ae2e470fd236afbb413b6885bc16feb3342c9a98f19dcf80ba2555c7561e769`; Llama-RS `24d41f46a4fe529785540e58702328b203d85954119e993c3515a48de8f1862f`; Qwen-BG `534a55c4f601cbf9210420895d34421c5a5f236762667e8f37c11c6af98585ec`; Qwen-RS `bfb57d98cb676d7590ea446e9394ed22d70886b39b129521cdfd8531a7d520f9`.
- corresponding manifest SHA-256: `4c0b28f3067b1f7568d238205524648df36b7eabcfddef513ea40c73829bfc42`, `f7c0c384e666b236a5cf4f078048ebce9f6e03e3d8d0dd7d76902b55d587f98d`, `c2687d68df3c833179744e73954fc92eb0eea8019b819cbe90e8dbb9d6780053`, `5936ebb79ba8f41e202354f62b90c25745ed3a7c79c127e4f2142cf2781185b4`.
- valid submission receipt SHA-256: `f4f7b1bec4f7a19e2e6e16ced3631d46be028ed26fd45cfa982882e4b58db210`.

No follow-up model/GPU/Slurm job was launched.
