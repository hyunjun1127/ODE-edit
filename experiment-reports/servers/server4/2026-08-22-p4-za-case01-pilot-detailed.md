# P4 ZA case01 target-side semantic barrier 상세 산출물 보고

## 1. 결론

- instruction:
  `ODEEDIT-S05-P4-TARGET-SIDE-SMOOTH-SEMANTIC-LOGODDS-BARRIER-V1`
- 실행 판정: `TECHNICAL_PILOT_PASS`
- 과학 판정: `SCIENCE_HOLD`
- 전체 ZA release 권고: `HOLD_FOR_TELEMETRY_REPAIR_AND_GH_REVIEW`
- Slurm array `22793_[0-1%2]`의 Llama/Qwen cell은 모두
  `COMPLETED`, exit `0:0`, elapsed `00:09:55`이다.
- 양 모델 모두 FULL-FP32/offline, same-W0, writer call 0, retry 0,
  nonfinite 0 조건에서 Z+, Z±, Native-Z를 완료했다.
- case01에서 Z±는 Z+ 대비 absolute efficacy를 유지했지만 held-out Gen을
  개선하지 못했다.
  - Llama: Gen `1.00 -> 1.00`, Gen new NLL mean `1.5234 -> 1.8376`
  - Qwen: Gen `0.95 -> 0.90`, Gen new NLL mean `2.5219 -> 2.6969`
- Native-Z의 Gen new NLL mean이 두 모델 모두 가장 낮았다.
- Llama target solver는 Z+/Z± 모두 모든 inner update에서 origin clamp에
  걸렸고, Qwen은 clamp hit가 없었다. 모델 간 target geometry 차이가 크다.
- authoritative contract §9의 new-only/PN gradient cosine과 request별 inner
  median/p90/max가 runtime receipt에 남지 않았다. 이 누락값은 저장된 SHA와
  norm만으로 사후 복원할 수 없다.

이 pilot은 실행 경계의 건전성을 확인하는 데에는 유효하다. 그러나 단일
B10 slice이며 writer0 target-injection 결과이므로 최종 weight edit의
`A±-A+` 인과효과로 해석하지 않는다. 자동 승격은 0이며 추가 submit은 HOLD다.

## 2. 식별 정보

| 항목 | 값 |
| --- | --- |
| server / agent | `server4` / `server4-server-head` |
| session | `codex://threads/01a028a7-9e3c-7541-81ba-efb40555d17d` |
| job | `22793_0` Llama, `22793_1` Qwen |
| start / end | `2026-08-22T20:22:40+09:00` / `20:32:35+09:00` |
| resource per cell | 1 GPU, 8 CPU, 65,000 MiB, limit 48h |
| project GPU cap | active 2 = cap 2 |
| source HEAD | `8229219c01f92e9c1d279b07504e88fc4b95f7ba` |
| source tree | `12f56a35b7f2c8bb074444fcd1c67ab94c8718e9` |
| seed | `1` (`COMMON_SEED`) |
| case | sealed independent stream slice 1, B10 |
| request order | `20933669138b5090645da9ea81b662085081fbc7eb756f542ec90bb9f51435ac` |

### Terminal identities

| 모델 | W0 SHA256 | terminal identity | terminal file SHA256 |
| --- | --- | --- | --- |
| Llama | `33a8ff4a1c57335b9c73107a758e0b31b2348e8795e89b13c0237158d81e7f78` | `f00cc764ae1fa40a43f952572a29feca050da6e99b4ff155992e001ca1ecf9a6` | `f01fa811cea4daa922d5a264607ad09b672f83306f28aa6377f7029ae60d88a5` |
| Qwen | `765190ef5405a7e9bc41297b37175ddbcae2f47abc6b80b3f0c10c9b279c124f` | `acf087c82bef1e1b3217fc865258e9dd79eb01cabe4c79329739b198c9e44f1d` | `37b4653ed54f28077c44ef86cfd6dc2ff62cdd231984bdad21c3d2bb84ff458a` |

Arm terminal identity, action-freeze raw SHA, 각 target outer receipt raw SHA를
부모 receipt와 독립 대조했다. 두 모델 모두 `ARM_LINKS_PASS`,
`ACTION_SHA_PASS`, `OUTER_RECEIPTS_PASS`다.

## 3. 봉인 입력과 실행 계약

### 입력 identity

- transfer archive SHA256:
  `748d9ff04ac584faf96af362b06fc9c34bc962062d8c1c72879f1c0311139756`
- extracted payload: 15 regular mode-0600 files, 163,211 bytes, 3,287 lines
- transfer receipt identity:
  `534b8a936f9dfdc2ab8f97dfcafa4d42f7523829cb7fc15e94b52522330cec50`
- member root:
  `28189ad33cdcda0c01d269216514508d62b15cd6c16ed053afd4683577169843`
- stream root:
  `74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6`
- all-order SHA:
  `abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c`
- evaluator identity:
  `72b8ecb737157a42d6a055ffd339dc3f907876a0a98165a00cca49ca9bbed07d`
- CounterFact dataset SHA:
  `d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f`
- FINAL_PRE_GPU identity:
  `257c74ec52cdfafd245ca14c10b61ad3b6cb366964f7c28af0e80cf3d4c14a71`

### 모델과 runtime

| 모델 | alias / native name | pinned revision | target layer | lr / KL / decay / clamp |
| --- | --- | --- | --- | --- |
| Llama | `llama3-8b-inst` / `meta-llama/Meta-Llama-3-8B-Instruct` | `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2` | `model.layers.8` | `0.1 / 0.0625 / 0.5 / 0.75` |
| Qwen | `qwen2.5-7b-inst` / `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | `model.layers.8` | `0.5 / 0.0625 / 0.001 / 4.0` |

- model parameter dtype: `torch.float32`
- autocast / CPU autocast / quantization: `false / false / false`
- model parameters requiring gradient: 0
- exact sealed snapshot, `local_files_only=true`, offline environment 3종 true
- EasyEdit root: `/data/janghj/EasyEdit`
- writer: 0회; W0는 모든 arm 전후 동일
- Z+/Z± 공통: 같은 requests, contexts, initialization, teacher, KL, decay,
  clamp, optimizer, M=5, K=8, final iterate only
- Native-Z: Official EasyEdit
  `easyeditor.models.alphaedit.compute_z.compute_z` 직접 10회 호출

## 4. 실제 실행 순서

각 array cell은 아래 순서를 독립 수행했다.

1. `stage-001`: FINAL_PRE_GPU와 모델별 sealed case01 binding을 model load 전에
   확인했다.
2. `stage-002`: pinned HF snapshot에서 모델/tokenizer를 offline으로 load하고
   모든 floating parameter가 FP32임을 확인했다.
3. `stage-003`: sealed request order와 runtime objective/capture plan을 모델
   alias에 결속했다. transferred BF16 plan hash는 observation-only다.
4. `stage-004`: W0 hash, activation capture, origin/current terminal z, KL teacher,
   Z+/Z± non-barrier identity를 고정했다.
5. Z+에서 outer `k=0..7`을 순서대로 수행했다. 각 outer는 독립 Adam moment로
   inner 5회를 전부 수행하고 항상 5번째 iterate를 다음 z로 넘겼다.
6. 첫 Z+ outer가 끝난 시점에 nonfinite 0, writer0, W0 unchanged/restored를
   `stage-005`로 기록했다.
7. W0가 같은지 다시 확인한 뒤 Z±에서 동일한 8×5 schedule을 수행했다.
8. 다시 W0를 확인한 뒤 Native-Z를 request별 1회, 총 10회 계산했다.
9. 각 arm action을 freeze한 뒤에만 terminal efficacy/generalization/locality
   evaluator를 열었다. inner held-out evaluation과 held-out decision influence는 0이다.
10. 세 arm terminal identity를 묶고 W0 restored=true인 model terminal을
    create-once로 기록했다.

## 5. z target의 정의와 산출물 의미

- `Z+`:
  `V+ = contexts_mean(-s_plus)`
- `Z±`:
  `V± = contexts_mean[-s_plus + softplus(s_minus-s_plus)]`
- `s_plus`: current request의 `target_new` length-normalized log probability
- `s_minus`: 같은 request의 `target_true` log probability
- 별도 margin threshold와 tuned barrier weight는 없다. barrier weight는 1이다.
- 아래 inner NLL/margin은 optimizer가 보고 있는 target-side activation
  intervention 값이다.
- terminal 지표는 최종 target z를 W0에 additive activation overlay로 주입해
  측정한 값이다. ZA는 writer0이므로 final edited W의 지표가 아니다.
- 각 outer receipt의 `z SHA`는 raw tensor를 Git에 넣지 않고 z 상태 전이를
  봉인하는 identity다.

## 6. Llama step별 z 진행

표의 `J1..J5`는 다섯 Adam update 직전의 total objective mean이다. `g J/V`는
마지막 inner에서 request별 combined/semantic gradient norm의 평균,
`step dz`는 해당 outer의 50개 request-update displacement norm 평균이다.
`clamp`는 50개 update 중 hit 수와 제거 norm 평균이다. `final-best`는 선택에
영향을 주지 않는 관측값이며 실제 선택은 항상 iterate 5다.

### Llama Z+

| k | J1 -> J2 -> J3 -> J4 -> J5 | final new/old NLL | margin | softplus / sigma | g J/V | step dz | clamp | final-best | z SHA prefix |
| ---: | --- | --- | ---: | --- | --- | ---: | --- | ---: | --- |
| 0 | 11.6208 -> 6.4069 -> 2.1550 -> 0.8513 -> 0.4445 | 0.3187 / 10.8284 | 10.5097 | 3.58e-4 / 3.58e-4 | 1.371 / 1.049 | 1.948 | 50/50, 2.876 | 0.0000 | `3d3ec0e56a5f -> c545cb3444c8` |
| 1 | 0.1958 -> 2.5017 -> 0.6371 -> 0.2052 -> 0.3789 | 0.2481 / 12.0669 | 11.8188 | 4.08e-4 / 4.07e-4 | 2.413 / 2.270 | 2.245 | 50/50, 2.754 | 0.1831 | `c545cb3444c8 -> 9b1414d6d3ac` |
| 2 | 0.2119 -> 2.2472 -> 1.3655 -> 0.2578 -> 0.1518 | 0.0449 / 12.3433 | 12.2984 | 1.35e-4 / 1.35e-4 | 0.489 / 0.323 | 2.169 | 50/50, 2.782 | 0.0000 | `9b1414d6d3ac -> 93a48556f215` |
| 3 | 0.1268 -> 3.3492 -> 2.1389 -> 1.1651 -> 0.2561 | 0.1379 / 12.3443 | 12.2064 | 9.57e-4 / 9.54e-4 | 0.841 / 0.829 | 2.313 | 50/50, 2.725 | 0.1293 | `93a48556f215 -> 85a7635499fc` |
| 4 | 0.1659 -> 2.8579 -> 1.2444 -> 0.2862 -> 0.2806 | 0.1720 / 11.6233 | 11.4513 | 0.01021 / 0.00947 | 3.099 / 3.062 | 2.301 | 50/50, 2.747 | 0.1148 | `85a7635499fc -> c4f6bf61ab2c` |
| 5 | 0.1396 -> 1.8398 -> 1.0447 -> 1.1905 -> 0.8773 | 0.7632 / 11.0915 | 10.3282 | 0.01278 / 0.01200 | 3.052 / 2.748 | 2.383 | 50/50, 2.717 | 0.7376 | `c4f6bf61ab2c -> b1fa854398e6` |
| 6 | 0.2969 -> 2.4720 -> 1.8081 -> 0.5411 -> 0.2860 | 0.1564 / 12.5883 | 12.4319 | 2.35e-4 / 2.35e-4 | 1.999 / 1.791 | 2.270 | 50/50, 2.799 | 0.0000 | `b1fa854398e6 -> 8ab215076d24` |
| 7 | 0.1864 -> 2.7374 -> 1.9490 -> 0.7362 -> 0.3886 | 0.2714 / 13.9215 | 13.6501 | 7.87e-4 / 7.83e-4 | 2.558 / 2.481 | 2.297 | 50/50, 2.717 | 0.2022 | `8ab215076d24 -> 9c26bf05fbb3` |

### Llama Z±

| k | J1 -> J2 -> J3 -> J4 -> J5 | final new/old NLL | margin | softplus / sigma | g J/V | step dz | clamp | final-best | z SHA prefix |
| ---: | --- | --- | ---: | --- | --- | ---: | --- | ---: | --- |
| 0 | 17.2390 -> 8.5741 -> 3.7740 -> 1.8033 -> 0.9505 | 0.8222 / 10.8210 | 9.9988 | 4.45e-4 / 4.45e-4 | 1.968 / 1.973 | 1.916 | 50/50, 2.907 | 0.0000 | `3d3ec0e56a5f -> 9d96bfe24564` |
| 1 | 0.3773 -> 3.2748 -> 0.6321 -> 0.1814 -> 0.1454 | 0.0222 / 12.5325 | 12.5103 | 1.15e-4 / 1.14e-4 | 0.429 / 0.252 | 2.104 | 50/50, 2.842 | 0.0000 | `9d96bfe24564 -> 51a34bb5e077` |
| 2 | 0.1270 -> 2.5737 -> 1.4344 -> 1.3691 -> 0.6161 | 0.4916 / 11.8868 | 11.3953 | 0.00945 / 0.00754 | 1.206 / 1.089 | 2.303 | 50/50, 2.711 | 0.4891 | `51a34bb5e077 -> 4b7b13e1b684` |
| 3 | 0.3989 -> 4.4107 -> 1.4032 -> 0.4482 -> 0.1907 | 0.0887 / 14.8253 | 14.7366 | 8.98e-6 / 8.98e-6 | 0.571 / 0.514 | 2.321 | 50/50, 2.731 | 0.0000 | `4b7b13e1b684 -> 79f7eac167cc` |
| 4 | 0.1267 -> 2.9313 -> 3.2748 -> 1.8203 -> 0.5986 | 0.4891 / 12.5807 | 12.0915 | 1.54e-4 / 1.53e-4 | 1.671 / 1.634 | 2.328 | 50/50, 2.721 | 0.4719 | `79f7eac167cc -> f4de9b8aa391` |
| 5 | 0.2835 -> 4.4414 -> 0.9003 -> 0.8361 -> 0.2642 | 0.1606 / 12.6137 | 12.4531 | 4.80e-4 / 4.79e-4 | 1.260 / 1.157 | 2.217 | 50/50, 2.770 | 0.0000 | `f4de9b8aa391 -> cd1166f70fc4` |
| 6 | 0.1891 -> 3.3263 -> 1.0657 -> 0.4195 -> 0.2047 | 0.0980 / 11.3606 | 11.2626 | 5.92e-4 / 5.91e-4 | 0.803 / 0.791 | 2.152 | 50/50, 2.806 | 0.0156 | `cd1166f70fc4 -> 2e588c10b527` |
| 7 | 0.1391 -> 3.7354 -> 0.8423 -> 0.5209 -> 0.2831 | 0.1650 / 14.0109 | 13.8459 | 4.32e-4 / 4.31e-4 | 1.490 / 1.408 | 2.272 | 50/50, 2.739 | 0.1440 | `2e588c10b527 -> 86b95d4e0ec4` |

Llama는 16개 outer-arm 모두 clamp 50/50이다. Z+에서 final iterate가
관측 best인 outer는 3/8, Z±는 4/8뿐이다. 계약대로 best를 선택하지 않았기
때문에 일부 outer의 `final-best`가 크며 특히 Z+ k5는 `0.7376`이다.

## 7. Qwen step별 z 진행

### Qwen Z+

| k | J1 -> J2 -> J3 -> J4 -> J5 | final new/old NLL | margin | softplus / sigma | g J/V | step dz | clamp | final-best | z SHA prefix |
| ---: | --- | --- | ---: | --- | --- | ---: | --- | ---: | --- |
| 0 | 12.5171 -> 7.6339 -> 3.7859 -> 1.4389 -> 0.7477 | 0.6670 / 12.1758 | 11.5088 | 0.00565 / 0.00537 | 0.172 / 0.159 | 19.509 | 0/50 | 0.0000 | `91ad27e317ce -> 5b1925ecd8de` |
| 1 | 0.3771 -> 0.4354 -> 0.0872 -> 0.0496 -> 0.0425 | 0.00524 / 16.4820 | 16.4768 | 3.81e-5 / 3.81e-5 | 0.0110 / 0.00627 | 18.499 | 0/50 | 0.0000 | `5b1925ecd8de -> 6d9454b7a193` |
| 2 | 0.0393 -> 0.0702 -> 0.0382 -> 0.0357 -> 0.0271 | 4.89e-4 / 19.8024 | 19.8019 | 2.46e-6 / 2.46e-6 | 0.00254 / 3.69e-4 | 18.664 | 0/50 | 0.0000 | `6d9454b7a193 -> ef7db2c9c89f` |
| 3 | 0.0403 -> 0.0544 -> 0.0197 -> 0.0188 -> 0.0169 | 8.84e-4 / 19.2764 | 19.2755 | 2.10e-6 / 2.10e-6 | 0.00144 / 6.27e-4 | 18.217 | 0/50 | 0.0000 | `ef7db2c9c89f -> e811b5f1c9b1` |
| 4 | 0.0155 -> 0.0448 -> 0.0169 -> 0.0123 -> 0.0109 | 1.65e-4 / 20.3207 | 20.3205 | 5.26e-6 / 5.26e-6 | 5.25e-4 / 1.06e-4 | 17.973 | 0/50 | 0.0000 | `e811b5f1c9b1 -> 6c438b7bf1a7` |
| 5 | 0.0106 -> 0.0168 -> 0.0105 -> 0.00854 -> 0.00761 | 8.60e-5 / 20.7414 | 20.7413 | 1.39e-7 / 1.39e-7 | 5.29e-4 / 6.70e-5 | 17.576 | 0/50 | 0.0000 | `6c438b7bf1a7 -> 5d6c01aa2933` |
| 6 | 0.00647 -> 0.0109 -> 0.00645 -> 0.00548 -> 0.00494 | 4.87e-5 / 19.9955 | 19.9955 | 1.36e-7 / 1.36e-7 | 2.97e-4 / 2.96e-5 | 16.910 | 0/50 | 0.0000 | `5d6c01aa2933 -> e0a508304447` |
| 7 | 0.00461 -> 0.00836 -> 0.00513 -> 0.00434 -> 0.00383 | 3.86e-5 / 19.7914 | 19.7914 | 5.80e-8 / 5.80e-8 | 2.70e-4 / 2.13e-5 | 16.834 | 0/50 | 0.0000 | `e0a508304447 -> fa65762a08c4` |

### Qwen Z±

| k | J1 -> J2 -> J3 -> J4 -> J5 | final new/old NLL | margin | softplus / sigma | g J/V | step dz | clamp | final-best | z SHA prefix |
| ---: | --- | --- | ---: | --- | --- | ---: | --- | ---: | --- |
| 0 | 19.2281 -> 10.2636 -> 4.9889 -> 2.6328 -> 1.3444 | 1.2364 / 12.4898 | 11.2534 | 0.00234 / 0.00230 | 0.266 / 0.255 | 19.340 | 0/50 | 0.0000 | `91ad27e317ce -> 3b7083b0e7ef` |
| 1 | 0.6307 -> 0.9142 -> 0.1923 -> 0.0881 -> 0.0663 | 0.00367 / 19.5461 | 19.5424 | 1.77e-5 / 1.77e-5 | 0.0131 / 0.00432 | 18.459 | 0/50 | 0.0000 | `3b7083b0e7ef -> 344b816dcf76` |
| 2 | 0.0586 -> 0.2765 -> 0.0561 -> 0.0487 -> 0.0417 | 0.00343 / 19.9227 | 19.9193 | 9.05e-6 / 9.05e-6 | 0.00412 / 0.00304 | 18.592 | 0/50 | 0.0000 | `344b816dcf76 -> 6da3de76b724` |
| 3 | 0.0367 -> 0.0539 -> 0.0310 -> 0.0216 -> 0.0180 | 3.95e-4 / 21.1417 | 21.1413 | 7.67e-7 / 7.67e-7 | 0.00208 / 3.28e-4 | 18.833 | 0/50 | 0.0000 | `6da3de76b724 -> a5fba1874678` |
| 4 | 0.0150 -> 0.0284 -> 0.0150 -> 0.0112 -> 0.00992 | 1.18e-4 / 20.3504 | 20.3502 | 2.85e-7 / 2.85e-7 | 6.18e-4 / 9.09e-5 | 18.219 | 0/50 | 0.0000 | `a5fba1874678 -> f0ee30a40e6f` |
| 5 | 0.00934 -> 0.0157 -> 0.00928 -> 0.00698 -> 0.00634 | 8.74e-5 / 20.6005 | 20.6004 | 2.08e-7 / 2.08e-7 | 4.19e-4 / 5.09e-5 | 17.439 | 0/50 | 0.0000 | `f0ee30a40e6f -> caf5274f7a0a` |
| 6 | 0.00613 -> 0.00970 -> 0.00677 -> 0.00552 -> 0.00472 | 8.34e-5 / 20.0190 | 20.0189 | 2.22e-7 / 2.22e-7 | 3.07e-4 / 5.77e-5 | 17.091 | 0/50 | 0.0000 | `caf5274f7a0a -> c4390b41bfb8` |
| 7 | 0.00441 -> 0.00748 -> 0.00445 -> 0.00393 -> 0.00359 | 5.60e-5 / 20.1165 | 20.1165 | 4.32e-8 / 4.32e-8 | 2.46e-4 / 3.61e-5 | 16.685 | 0/50 | 0.0000 | `c4390b41bfb8 -> 9209424d27c7` |

Qwen에서는 모든 outer의 최종 iterate가 관측 best였고 clamp hit는 0이다.
origin clamp factor와 lr가 Llama와 다르므로 `step dz`의 모델 간 절대 크기를
동일 scale의 성능 척도로 해석하지 않는다.

## 8. Z± barrier의 inner 활성도

`sigma = sigmoid(s_minus-s_plus)`다. 1에 가까우면 old가 new보다 강해 barrier
gradient가 활발하고, 0에 가까우면 new가 충분히 앞서 barrier가 사실상
비활성이다. 아래는 각 outer에서 inner 1→5의 평균 궤적이다.

### Llama Z±

| k | new NLL | new-old margin | sigma |
| ---: | --- | --- | --- |
| 0 | 11.6208 -> 7.4491 -> 3.6430 -> 1.6846 -> 0.8222 | -5.2346 -> 0.6029 -> 5.6547 -> 8.2507 -> 9.9988 | 0.82269 -> 0.46157 -> 0.02222 -> 0.00132 -> 0.00044 |
| 1 | 0.2411 -> 3.1065 -> 0.5208 -> 0.0702 -> 0.0222 | 11.8958 -> 6.5132 -> 9.8644 -> 11.3563 -> 12.5103 | 0.00004 -> 0.03577 -> 0.00127 -> 0.00028 -> 0.00011 |
| 2 | 0.0180 -> 2.3083 -> 1.2483 -> 1.0952 -> 0.4916 | 12.9846 -> 6.7517 -> 11.4220 -> 10.1977 -> 11.3953 | 0.00032 -> 0.06633 -> 0.01743 -> 0.07123 -> 0.00754 |
| 3 | 0.2922 -> 3.1253 -> 1.2553 -> 0.3232 -> 0.0887 | 12.9681 -> 6.4886 -> 11.6070 -> 11.0007 -> 14.7366 | 0.00024 -> 0.22065 -> 0.01088 -> 0.01309 -> 0.00001 |
| 4 | 0.0283 -> 2.4421 -> 2.9416 -> 1.6951 -> 0.4891 | 15.0074 -> 7.8858 -> 9.0442 -> 9.9416 -> 12.0915 | 0.00001 -> 0.15129 -> 0.11450 -> 0.01412 -> 0.00015 |
| 5 | 0.1798 -> 3.6836 -> 0.7678 -> 0.7241 -> 0.1606 | 12.9312 -> 3.5531 -> 11.8734 -> 11.1466 -> 12.4531 | 0.00003 -> 0.25620 -> 0.00102 -> 0.00015 -> 0.00048 |
| 6 | 0.0900 -> 2.6028 -> 0.9250 -> 0.3059 -> 0.0980 | 13.2045 -> 6.9040 -> 11.4109 -> 11.2110 -> 11.2626 | 0.00006 -> 0.12013 -> 0.00529 -> 0.00083 -> 0.00059 |
| 7 | 0.0263 -> 3.3249 -> 0.5982 -> 0.3756 -> 0.1650 | 12.1376 -> 7.0469 -> 11.2604 -> 10.4817 -> 13.8459 | 0.00037 -> 0.13174 -> 0.04236 -> 0.01955 -> 0.00043 |

### Qwen Z±

| k | new NLL | new-old margin | sigma |
| ---: | --- | --- | --- |
| 0 | 12.5171 -> 8.6336 -> 4.8682 -> 2.5220 -> 1.2364 | -6.6773 -> -0.4799 -> 4.8062 -> 8.5946 -> 11.2534 | 0.97014 -> 0.62367 -> 0.06104 -> 0.00904 -> 0.00230 |
| 1 | 0.5205 -> 0.7664 -> 0.0905 -> 0.0105 -> 0.00367 | 13.4514 -> 14.3484 -> 17.2100 -> 18.9164 -> 19.5424 | 0.00101 -> 0.00168 -> 0.00008 -> 0.00002 -> 0.00002 |
| 2 | 0.00474 -> 0.2092 -> 0.00378 -> 0.00165 -> 0.00343 | 19.6619 -> 16.8915 -> 18.1796 -> 20.0884 -> 19.9193 | 0.00001 -> 0.00014 -> 0.00002 -> 0.00001 -> 0.00001 |
| 3 | 0.00233 -> 0.00072 -> 0.00104 -> 0.00027 -> 0.00039 | 19.9523 -> 21.0693 -> 20.1464 -> 21.4319 -> 21.1413 | 0.00001 -> 0 -> 0 -> 0 -> 0 |
| 4 | 0.00035 -> 0.00010 -> 0.00155 -> 0.00014 -> 0.00012 | 20.5671 -> 21.5037 -> 20.2084 -> 20.4446 -> 20.3502 | 0 -> 0 -> 0.00001 -> 0 -> 0 |
| 5 | 0.00015 -> 0.00009 -> 0.00011 -> 0.00011 -> 0.00009 | 20.2148 -> 21.0364 -> 20.5704 -> 20.5245 -> 20.6004 | 0 -> 0 -> 0 -> 0 -> 0 |
| 6 | 0.00008 -> 0.00008 -> 0.00006 -> 0.00008 -> 0.00008 | 20.6901 -> 21.4731 -> 21.1084 -> 20.2223 -> 20.0189 | 0 -> 0 -> 0 -> 0 -> 0 |
| 7 | 0.00007 -> 0.00003 -> 0.00005 -> 0.00005 -> 0.00006 | 20.1762 -> 21.1213 -> 20.4144 -> 20.1914 -> 20.1165 | 0 -> 0 -> 0 -> 0 -> 0 |

두 모델 모두 최초 k0에서는 barrier가 강하게 작동해 음의 margin을 양으로
뒤집었다. 이후에는 sigma가 거의 0으로 수렴했다. 특히 Qwen k3 이후 barrier
항은 수치적으로 사실상 꺼져 있다. 따라서 terminal 차이는 지속적인 barrier
압력보다는 초반 궤적 분기와 이후 Adam state/reset의 누적 결과로 보는 것이
타당하다.

## 9. terminal z-injection 결과

NLL은 낮을수록 좋다. Gen은 모델별 10 requests × 2 prompts = 20이
denominator다. locality는 terminal z overlay가 locality rows에는 hook을 걸지
않는 W0 기준 관측값이므로 arm별 z quality를 구분하지 않는다.

| 모델 | arm | Eff | Gen | Loc | Eff new NLL mean | Gen new NLL mean / p90 / max | Gen margin mean |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| Llama | Z+ | 10/10 | 20/20 | 93/100 | 0.05924 | 1.5234 / 5.1491 / 11.1597 | 10.4130 |
| Llama | Z± | 10/10 | 20/20 | 93/100 | 0.06031 | 1.8376 / 5.3474 / 10.7554 | 8.3197 |
| Llama | Native-Z | 10/10 | 20/20 | 93/100 | 0.00098 | 1.0409 / 2.8104 / 6.8033 | 10.7380 |
| Qwen | Z+ | 10/10 | 19/20 | 93/100 | 0.000076 | 2.5219 / 6.7846 / 10.7849 | 7.9740 |
| Qwen | Z± | 10/10 | 18/20 | 93/100 | 0.000086 | 2.6969 / 7.9519 / 10.5151 | 8.3169 |
| Qwen | Native-Z | 10/10 | 20/20 | 93/100 | 0.00640 | 2.0096 / 7.0187 / 8.7351 | 11.0727 |

### Z± - Z+ observed delta

| 모델 | Eff delta | Gen delta | Gen new NLL mean delta | p90 delta | max delta | margin delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama | 0 | 0 | +0.3142 | +0.1984 | -0.4043 | -2.0933 |
| Qwen | 0 | -1/20 (-0.05) | +0.1750 | +1.1672 | -0.2698 | +0.3430 |

두 모델 모두 max 하나는 소폭 줄었지만 mean/p90이 악화했다. Qwen의 margin은
증가했으나 absolute Gen과 Gen NLL이 악화했으므로 계약의 “margin-only 성공
금지”에 따라 성공으로 판정하지 않는다.

## 10. case별 Gen 결과

각 셀은 `correct/2, 두 Gen prompt의 new NLL 평균`이다. case membership/order는
Llama와 Qwen에 동일하다.

### Llama

| ordinal | case_id | Z+ | Z± | Native-Z |
| ---: | ---: | --- | --- | --- |
| 1 | 19795 | 2/2, 5.6147 | 2/2, 5.3858 | 2/2, 3.4060 |
| 2 | 8489 | 2/2, 0.0850 | 2/2, 0.0129 | 2/2, 0.0035 |
| 3 | 12436 | 2/2, 0.1958 | 2/2, 1.4808 | 2/2, 0.6527 |
| 4 | 15031 | 2/2, 3.4252 | 2/2, 2.9993 | 2/2, 2.6800 |
| 5 | 20453 | 2/2, 0.0006 | 2/2, 0.2681 | 2/2, 0.0030 |
| 6 | 107 | 2/2, 2.4337 | 2/2, 2.8477 | 2/2, 1.4230 |
| 7 | 19312 | 2/2, 2.5639 | 2/2, 2.6954 | 2/2, 0.9639 |
| 8 | 11840 | 2/2, 0.8727 | 2/2, 0.3574 | 2/2, 0.0493 |
| 9 | 8439 | 2/2, 0.0109 | 2/2, 0.3143 | 2/2, 0.5026 |
| 10 | 16743 | 2/2, 0.0315 | 2/2, 2.0143 | 2/2, 0.7249 |

### Qwen

| ordinal | case_id | Z+ | Z± | Native-Z |
| ---: | ---: | --- | --- | --- |
| 1 | 19795 | 2/2, 4.0250 | 2/2, 4.0383 | 2/2, 5.0518 |
| 2 | 8489 | 2/2, 0.0084 | 2/2, 0.5924 | 2/2, 0.0228 |
| 3 | 12436 | 2/2, 0.5922 | 2/2, 1.4249 | 2/2, 1.2458 |
| 4 | 15031 | 2/2, 5.3943 | 2/2, 4.3953 | 2/2, 2.9855 |
| 5 | 20453 | 2/2, 0.3114 | 2/2, 0.0009 | 2/2, 0.0604 |
| 6 | 107 | 2/2, 5.1012 | 1/2, 5.1032 | 2/2, 4.8025 |
| 7 | 19312 | 2/2, 3.2062 | 2/2, 5.2636 | 2/2, 3.4975 |
| 8 | 11840 | 1/2, 3.3741 | 2/2, 0.5369 | 2/2, 0.1933 |
| 9 | 8439 | 2/2, 0.0870 | 1/2, 2.8411 | 2/2, 0.0382 |
| 10 | 16743 | 2/2, 3.1188 | 2/2, 2.7725 | 2/2, 2.1983 |

Qwen Z±는 case 11840을 회복했지만 case 107과 8439에서 각 한 prompt를
잃었다. 평균 NLL도 case 2, 3, 7, 9에서 뚜렷하게 악화해 순효과는 음수다.

## 11. 실행 overhead

### Arm wall-clock

`core`는 arm total에서 terminal evaluator 자체의 wall을 뺀 값이다. target
arm core에는 8×5 solver와 action-freeze 준비가 포함되며 Native core에는
Official compute_z 10회와 action-freeze 준비가 포함된다.

| 모델 | arm | total s | evaluator s | core s | cell wall 비율 |
| --- | --- | ---: | ---: | ---: | ---: |
| Llama | Z+ | 256.202 | 1.885 | 254.317 | 43.06% |
| Llama | Z± | 253.830 | 1.876 | 251.954 | 42.66% |
| Llama | Native-Z | 35.940 | 1.882 | 34.058 | 6.04% |
| Llama | preflight/load/plan/W0 guards/finalization | 49.028 | n/a | n/a | 8.24% |
| Qwen | Z+ | 267.725 | 1.874 | 265.851 | 45.00% |
| Qwen | Z± | 251.579 | 1.864 | 249.715 | 42.28% |
| Qwen | Native-Z | 24.187 | 1.874 | 22.313 | 4.07% |
| Qwen | preflight/load/plan/W0 guards/finalization | 51.510 | n/a | n/a | 8.66% |

Z±-Z+ observed wall delta는 Llama `-2.372 s (-0.93%)`, Qwen
`-16.146 s (-6.03%)`다. 이를 barrier의 계산 절감으로 해석하지 않는다.
두 arm은 공정한 telemetry를 위해 Z+에서도 old NLL forward를 수행하므로
model-call schedule이 동일하다. 차이는 warm-cache, scheduling 및
serialization 변동 범위의 관측값이다.

### Outer별 wall-clock

아래 값은 outer JSON mtime과 arm receipt total wall로부터 계산한 근사값이다.
`tail`은 k7 종료 후 action freeze, case materialization, terminal evaluator와
terminal write까지다. filesystem timestamp/serialization 오차를 포함한다.

| 모델/arm | k0 | k1 | k2 | k3 | k4 | k5 | k6 | k7 | tail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama Z+ | 19.458 | 28.778 | 29.675 | 30.091 | 30.746 | 29.160 | 29.211 | 28.196 | 30.887 |
| Llama Z± | 21.461 | 30.197 | 29.367 | 29.800 | 29.283 | 29.560 | 30.011 | 29.012 | 25.138 |
| Qwen Z+ | 19.383 | 30.019 | 31.170 | 31.035 | 31.039 | 31.649 | 30.145 | 28.637 | 34.648 |
| Qwen Z± | 21.575 | 31.127 | 30.416 | 30.303 | 30.692 | 31.054 | 30.789 | 28.619 | 17.003 |

### Model-call overhead

각 target arm의 receipt에서 직접 합산한 호출 수는 모델과 arm에 동일하다.

| 항목 | arm당 횟수 | 설명 |
| --- | ---: | --- |
| target_new semantic forward | 400 | 8 outer × 5 inner × 10 requests |
| target_true semantic forward | 400 | Z+에서는 telemetry-only, Z±에서는 barrier에 기여 |
| KL forward | 400 | pinned teacher KL |
| KL backward | 400 | request별 KL gradient |
| semantic gradient call | 400 | request별 target gradient |
| terminal evaluator forward | 10 | action freeze 후 1회/request |

Z+에서 old forward 400회는 old-source telemetry와 arm input symmetry를 위한
관측 overhead다. 별도 timer가 없어 이 부분만의 wall을 정확히 분리할 수 없다.

### 자원·storage overhead

- successful pilot allocation: `1,190 GPU-s = 0.3306 GPU-h`
- successful pilot CPU allocation: `9,520 CPU-s = 2.6444 CPU-h`
- requested memory-time: 약 `20.98 GiB-h`
- Slurm host MaxRSS:
  - Llama: `2,207,872 K`, 약 `2.106 GiB`
  - Qwen: `2,271,424 K`, 약 `2.166 GiB`
- GPU peak memory/utilization trace: 미기록. host MaxRSS로 대체 추정하지 않음.
- final result receipts: 56 files, 1,720,393 bytes
  - Llama: 28 files, 858,665 bytes
  - Qwen: 28 files, 861,728 bytes
- final stdout/stderr: 4 files, 3,173 bytes
- model/HF/EasyEdit artifacts는 PROTECTED_REUSABLE cache를 재사용했으며 per-job
  storage로 중복 생성하지 않았다.

### 사전 기술 실패 overhead

성공 namespace 이전 fail-close 배열은 `22785`, `22787`, `22789`, `22791`이다.
각각 session boundary, BF16 plan authority, context schema boundary, scalar device
binding에서 terminal science 전에 종료됐다. cell elapsed 합은 162 GPU-s,
즉 `0.045 GPU-h`이며 성공 pilot GPU-time의 약 13.6%다. 성공 run까지 포함한
총 project allocation은 약 `0.3756 GPU-h`다. formula, stream, evaluator,
tolerance의 과학 delta는 0이고 실패 namespace 결과를 metric에 합치지 않았다.

## 12. 경고와 계약상 한계

1. **필수 gradient cosine 누락**: 구현된
   `semantic_gradient_comparison`이 runtime solver binding에서 호출되지 않았다.
   receipt에는 arm별 semantic gradient SHA와 norm만 있어 동일 state의
   new-only/PN cosine을 사후 계산할 수 없다.
2. **request별 inner 분위수 누락**: inner semantic receipt의 median/p90/max는
   10×6 tensor 전체를 flatten한 summary다. contract의 request별 summary가 아니다.
3. **HF receipt 표현 불일치**: 실제 model load와 FP32 검증은 성공했지만
   post-model stage 안의 preflight-derived `hf.model_loaded`는 false다.
4. **telemetry warning**: `decay_values`의 requires-grad tensor를 detach 없이
   Python float로 바꾸는 경고가 양 모델에서 1회 발생했다. nonfinite나 결과
   오류 징후는 없지만 telemetry serialization 전 detach가 필요하다.
5. **deprecated argument warning**: Transformers의 `torch_dtype` deprecation
   warning이 있다. 현재 load dtype은 실제 FP32로 검증됐다.
6. **ZA 범위 한계**: writer0이므로 P/stats/cache update, `R_P(Delta W)`, final W
   Eff/Gen/Loc, historical forgetting과의 상관은 이 산출물에 존재하지 않는다.
7. **표본 한계**: 독립 slice 하나뿐이므로 모델/arm 평균과 causal claim을
   승격하지 않는다. retry/imputation은 0이다.

## 13. 판정과 다음 행동

- 기술 실행: `PASS`
- z solver finite/W0/final-only contract: `PASS`
- case01 barrier signal: `NO_POSITIVE_SIGNAL`
- report completeness versus authoritative §9: `INCOMPLETE_TELEMETRY`
- scientific promotion: `HOLD`
- 추가 submit: `HOLD`

전체 ZA 또는 ZB 전에 권고하는 최소 변경은 과학식·stream·tolerance를 건드리지
않는 telemetry-only repair다.

1. 동일 target state에서 new-only와 PN semantic gradient를 observation-only로
   함께 계산하고 request별 norm/cosine을 receipt에 연결한다.
2. new/old NLL, margin, softplus, sigma의 request별 median/p90/max를 보존한다.
3. post-load receipt의 `model_loaded=true` 사실값을 별도 필드로 기록한다.
4. decay telemetry scalar 변환에 명시적 detach를 적용한다.
5. focused negative/identity test 후 GH가 full ZA release 여부를 결정한다.

## 14. 로컬 산출물과 checksum

원시 tensor/model/data/log/result는 Git에 포함하지 않는다. 아래 ignored local
path만 참조한다.

- tracked machine-readable analysis:
  `experiment-reports/servers/server4/2026-08-22-p4-za-case01-pilot-analysis.json`
  - SHA256 `6edd2965a67e2042be70aaea9ae7a1dc0650b19b342d336900ba8fe839158aa0`
- result root:
  `/data/janghj/ODE-edit/local/odebf/results/p4-target-side-semantic-barrier-v1/za-case01-pilot-8229219-tech-r4`
- log root:
  `/data/janghj/ODE-edit/local/odebf/slurm/p4-za-case01-8229219-tech-r4`
- FINAL_PRE_GPU receipt:
  `/data/janghj/ODE-edit/local/state/p4-target-side-semantic-barrier-v1/readiness/za-case01-8229219-tech-r4/final-pre-gpu.json`
  - bytes 7,955, mode 0600
  - raw SHA256 `0cb10214cb32fef619a70d5c0f4bdc9aa3477389ce5d2f0bba0d84e995f041cc`
- Llama stdout/stderr SHA256:
  `3d4072782bdd44fccaf0497d27963eeb2ed3f1c8e7496e7f9b3f0364d946ac40` /
  `67f704b7da82647be6f4b779e83f5215a5f0bb7082c97b8d8b09a12baae34e50`
- Qwen stdout/stderr SHA256:
  `a47a9b9bf20336392717309f3428d41236ccd83367a3a0710e0a4e41e3eebacf` /
  `86a29a8e641a619626b193de7149b973addd4008f744228772ace1bf3717ad97`
- P4 ZA source manifest SHA/root:
  `36ecc9fc61e9bfaa0d857e461d4338626362b41e8aaa77987be9bcc5e1a88113` /
  `ce3b0cc4bfd1003c0c8dc5ff18dc001b184969428604dcb67d8940771dc481f9`
- P4 numerical lock SHA/root:
  `934137ec143a481ba69fe27053d35fb9db3b186ceab9c9843e004f838653e4e7` /
  `a7e8d51b0fcc8ac7576d4d16346f415fdca5c98904753cdcc2e9133dc6ba92c6`
